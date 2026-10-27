"""Resumable, time-budgeted experiment pipeline.

Each invocation runs pending work units (training epochs, counterfactual batches, ...) until the
time budget is used, persisting every unit to disk. Re-invoke until it prints PIPELINE COMPLETE.

  python -m trace_ids.pipeline --dataset unsw --path data/unsw --detector cnnbilstm --seed 0 \
      --out results/unsw_cnnbilstm_s0 --full --budget 240
"""
import argparse
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import f1_score

from . import cf, data, models
from .metrics import cf_quality, detection_metrics, per_class, softmax, subsample_alerts
from .space import FeasibleSpace, Prep

torch.set_num_threads(int(os.environ.get("TRACE_THREADS", "1")))
BS_CF = 1024


class Budget:
    def __init__(self, seconds):
        self.t0, self.s = time.time(), seconds

    def left(self):
        return self.s - (time.time() - self.t0)

    def ok(self, need=0.0):
        return self.left() > need


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["unsw", "cic"])
    ap.add_argument("--path", required=True)
    ap.add_argument("--detector", default="cnnbilstm", choices=["cnnbilstm", "mlp"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cap", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--n_tp", type=int, default=1500)
    ap.add_argument("--n_bench", type=int, default=1000)
    ap.add_argument("--n_sens", type=int, default=300)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=0.05)
    ap.add_argument("--full", action="store_true", help="benchmark, ablations, sensitivity, references")
    ap.add_argument("--det_only", action="store_true")
    ap.add_argument("--with_val", action="store_true", help="also explain validation alerts")
    ap.add_argument("--budget", type=float, default=1e9, help="seconds per invocation")
    a = ap.parse_args()
    B_ = Budget(a.budget)
    os.makedirs(a.out, exist_ok=True)
    P = lambda f: os.path.join(a.out, f)  # noqa: E731
    logf = open(P("log.txt"), "a")

    def log(*m):
        s = " ".join(str(x) for x in m)
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    def jsave(obj, fn):
        if isinstance(obj, pd.DataFrame):
            obj.to_csv(P(fn), index=False)
        else:
            with open(P(fn), "w") as f:
                json.dump(obj, f, indent=1, default=float)

    # ---------------- data (deterministic, recomputed each call) ----------------
    X, ylab, benign_name, stats = data.load(a.dataset, a.path, cap=a.cap, seed=0)
    names = sorted(ylab.unique())
    y = np.array([names.index(v) for v in ylab])
    B = names.index(benign_name)
    tr, va, te = data.split(X, ylab, seed=0)
    prep = Prep(a.dataset, X.iloc[tr])
    Utr, OHtr, Rtr = prep.transform(X.iloc[tr])
    Uva, OHva, Rva = prep.transform(X.iloc[va])
    Ute, OHte, Rte = prep.transform(X.iloc[te])
    space = FeasibleSpace(prep, Rtr)
    Xtr_in, Xva_in, Xte_in = (np.concatenate([U, O], 1) for U, O in ((Utr, OHtr), (Uva, OHva), (Ute, OHte)))
    ytr, yva, yte = y[tr], y[va], y[te]
    if not os.path.exists(P("data_stats.json")):
        stats.update({"train": len(tr), "val": len(va), "test": len(te), "classes": len(names),
                      "class_counts": ylab.value_counts().to_dict(), "benign": benign_name})
        jsave(stats, "data_stats.json")
        rep = pd.DataFrame(space.report)
        Dte = {c: Rte[:, i] for c, i in prep.ix.items()}
        rep["test_satisfaction"] = [float(np.mean(r.check(Dte))) for r in space.candidates]
        jsave(rep, "constraints.csv")
        jsave(pd.DataFrame({"feature": prep.num_cols, "immutable": space.imm, "derived": space.der,
                            "integer": space.int, "log1p": prep.log, "raw_min": prep.raw_lo,
                            "raw_max": prep.raw_hi, "sigma_u": prep.sigma}), "features.csv")
        log(f"[{a.dataset}/{a.detector}/s{a.seed}] {stats}")

    # ---------------- training (one epoch per unit, resumable) ----------------
    torch.manual_seed(a.seed)
    model = models.build(a.detector, Xtr_in.shape[1], len(names))
    if not os.path.exists(P("model.pt")):
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
        st = {"epoch": 0, "best": -1.0, "wait": 0, "hist": [], "best_state": None, "train_time": 0.0,
              "rng": torch.get_rng_state(), "done": False}
        if os.path.exists(P("train_state.pt")):
            ck = torch.load(P("train_state.pt"), weights_only=False)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            st = ck["st"]
            torch.set_rng_state(st["rng"])
        lossf = nn.CrossEntropyLoss(weight=models.class_weights(ytr, len(names)))
        Xt, yt = torch.from_numpy(Xtr_in), torch.from_numpy(ytr).long()
        ep_time = st["hist"][-1]["epoch_time"] if st["hist"] else 0.0
        while not st["done"] and B_.ok(ep_time * 1.15 + 5):
            t1 = time.time()
            model.train()
            perm = torch.randperm(len(Xt))
            tot = 0.0
            for i in range(0, len(Xt), 512):
                b = perm[i:i + 512]
                opt.zero_grad()
                loss = lossf(model(Xt[b]), yt[b])
                loss.backward()
                opt.step()
                tot += loss.item() * len(b)
            f1 = f1_score(yva, models.predict_logits(model, Xva_in).argmax(1), average="macro")
            sched.step(f1)
            st["epoch"] += 1
            ep_time = time.time() - t1
            st["train_time"] += ep_time
            st["hist"].append({"epoch": st["epoch"], "train_loss": tot / len(Xt), "val_macro_f1": f1,
                               "epoch_time": ep_time})
            log(f"  epoch {st['epoch']:02d} loss {tot / len(Xt):.4f} val macroF1 {f1:.4f} ({ep_time:.0f}s)")
            if f1 > st["best"] + 1e-4:
                st["best"], st["wait"] = f1, 0
                st["best_state"] = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                st["wait"] += 1
            if st["wait"] >= 5 or st["epoch"] >= a.epochs:
                st["done"] = True
            st["rng"] = torch.get_rng_state()
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "st": st}, P("train_state.pt"))
        if not st["done"]:
            log(f"  [budget] training paused at epoch {st['epoch']}")
            return
        model.load_state_dict(st["best_state"])
        torch.save(model.state_dict(), P("model.pt"))
        jsave(pd.DataFrame(st["hist"]), "train_history.csv")
        with open(P("train_meta.json"), "w") as f:
            json.dump({"train_time_s": st["train_time"], "epochs": st["epoch"], "best_val_macro_f1": st["best"]}, f)
    model.load_state_dict(torch.load(P("model.pt")))
    model.eval()

    # ---------------- detection ----------------
    if not os.path.exists(P("detection.json")):
        t1 = time.time()
        lg_te = models.predict_logits(model, Xte_in)
        t_inf = (time.time() - t1) / len(Xte_in)
        lg_va = models.predict_logits(model, Xva_in)
        pred_tr = models.predict_logits(model, Xtr_in).argmax(1)
        meta = json.load(open(P("train_meta.json")))
        det = detection_metrics(yte, softmax(lg_te), B)
        det.update({"model": a.detector, "seed": a.seed, "train_time_s": meta["train_time_s"],
                    "epochs": meta["epochs"], "infer_ms_per_flow": 1000 * t_inf,
                    "params": sum(p.numel() for p in model.parameters())})
        np.savez_compressed(P("preds.npz"), lg_te=lg_te, lg_va=lg_va, pred_tr=pred_tr)
        jsave(per_class(yte, lg_te.argmax(1), names), "per_class.csv")
        jsave(det, "detection.json")
        log(f"  detection: {det}")
    if a.det_only:
        log("PIPELINE COMPLETE")
        return
    pz = np.load(P("preds.npz"))
    P_te, P_va, pred_tr = softmax(pz["lg_te"]), softmax(pz["lg_va"]), pz["pred_tr"]

    # ---------------- reference models (chunked) ----------------
    if a.full and not os.path.exists(P("reference_models.csv")):
        import xgboost as xgb
        from sklearn.ensemble import RandomForestClassifier
        w = models.class_weights(ytr, len(names)).numpy()[ytr]
        refs = json.load(open(P("refs_partial.json"))) if os.path.exists(P("refs_partial.json")) else {}
        if "XGBoost" not in refs:
            dtr = xgb.DMatrix(Xtr_in, label=ytr, weight=w)
            params = {"objective": "multi:softprob", "num_class": len(names), "max_depth": 8, "eta": 0.1,
                      "tree_method": "hist", "nthread": 1, "seed": 0}
            bst = xgb.Booster(model_file=P("xgb.json")) if os.path.exists(P("xgb.json")) else None
            done = bst.num_boosted_rounds() if bst is not None else 0
            tt = json.load(open(P("xgb_time.json")))["t"] if os.path.exists(P("xgb_time.json")) else 0.0
            while done < 200 and B_.ok(60):
                t1 = time.time()
                bst = xgb.train(params, dtr, num_boost_round=20, xgb_model=bst)
                done = bst.num_boosted_rounds()
                tt += time.time() - t1
                bst.save_model(P("xgb.json"))
                json.dump({"t": tt}, open(P("xgb_time.json"), "w"))
                log(f"  XGBoost rounds {done}")
            if done < 200:
                return
            t1 = time.time()
            pr = bst.predict(xgb.DMatrix(Xte_in))
            m = detection_metrics(yte, pr, B)
            m.update({"model": "XGBoost", "train_time_s": tt, "infer_ms_per_flow": 1000 * (time.time() - t1) / len(Xte_in)})
            jsave(per_class(yte, pr.argmax(1), names), "per_class_XGBoost.csv")
            refs["XGBoost"] = m
            json.dump(refs, open(P("refs_partial.json"), "w"), default=float)
        if "RandomForest" not in refs:
            rf = pickle.load(open(P("rf.pkl"), "rb")) if os.path.exists(P("rf.pkl")) else \
                RandomForestClassifier(n_estimators=0, n_jobs=1, random_state=0, warm_start=True)
            tt = json.load(open(P("rf_time.json")))["t"] if os.path.exists(P("rf_time.json")) else 0.0
            while rf.n_estimators < 100 and B_.ok(60):
                t1 = time.time()
                rf.n_estimators += 10
                rf.fit(Xtr_in, ytr, sample_weight=w)
                tt += time.time() - t1
                pickle.dump(rf, open(P("rf.pkl"), "wb"))
                json.dump({"t": tt}, open(P("rf_time.json"), "w"))
                log(f"  RandomForest trees {rf.n_estimators}")
            if rf.n_estimators < 100:
                return
            t1 = time.time()
            pr = rf.predict_proba(Xte_in)
            m = detection_metrics(yte, pr, B)
            m.update({"model": "RandomForest", "train_time_s": tt, "infer_ms_per_flow": 1000 * (time.time() - t1) / len(Xte_in)})
            jsave(per_class(yte, pr.argmax(1), names), "per_class_RandomForest.csv")
            refs["RandomForest"] = m
            json.dump(refs, open(P("refs_partial.json"), "w"), default=float)
            os.remove(P("rf.pkl"))
        jsave(pd.DataFrame(list(refs.values())), "reference_models.csv")
        log(f"  references: {list(refs.values())}")

    # ---------------- alerts, benign pool, NUN ----------------
    rng = np.random.RandomState(a.seed)
    if not os.path.exists(P("alerts.npz")):
        fp_te, tp_te = subsample_alerts(yte, P_te.argmax(1), B, a.n_tp, rng)
        fp_va, tp_va = subsample_alerts(yva, P_va.argmax(1), B, a.n_tp, rng)
        btr = np.flatnonzero((ytr == B) & (pred_tr == B))
        if len(btr) > 30000:
            btr = np.sort(rng.choice(btr, 30000, replace=False))
        ref = btr if len(btr) <= 20000 else np.sort(rng.choice(btr, 20000, replace=False))
        bva = np.flatnonzero(yva == B)
        bva = bva if len(bva) <= 2000 else np.sort(rng.choice(bva, 2000, replace=False))
        A_te, A_va = np.concatenate([fp_te, tp_te]), np.concatenate([fp_va, tp_va])
        bench = np.sort(rng.choice(len(A_te), min(a.n_bench, len(A_te)), replace=False))
        sens = np.sort(rng.choice(len(A_te), min(a.n_sens, len(A_te)), replace=False))
        np.savez_compressed(P("alerts.npz"), A_te=A_te, A_va=A_va, nfp_te=len(fp_te), nfp_va=len(fp_va),
                            btr=btr, ref=ref, bva=bva, bench=bench, sens=sens)
        log(f"  alerts test: FP {len(fp_te)} TP(sub) {len(tp_te)} | val: FP {len(fp_va)} TP(sub) {len(tp_va)}")
    al = np.load(P("alerts.npz"))
    A_te, A_va, btr, ref, bva = al["A_te"], al["A_va"], al["btr"], al["ref"], al["bva"]
    isfp_te = np.r_[np.ones(int(al["nfp_te"]), bool), np.zeros(len(A_te) - int(al["nfp_te"]), bool)]
    Upool, OHpool, Rpool = Utr[btr], OHtr[btr], Rtr[btr]
    Uref = Utr[ref]
    split = {"te": (A_te, Ute, OHte, Rte), "va": (A_va, Uva, OHva, Rva)}
    for sp in (("va", "te") if a.with_val else ("te",)):
        if not os.path.exists(P(f"nun_{sp}.npz")):
            A, U, OH, R = split[sp]
            Un, Rn, cn, tn, nb = cf.nun(model, space, U[A], OH[A], R[A], B, Upool, OHpool, Rpool)
            np.savez_compressed(P(f"nun_{sp}.npz"), U=Un, R=Rn, c=cn, t=tn, nb=nb)
            log(f"  NUN {sp}: valid {np.isfinite(cn).mean():.3f}")

    # ---------------- batched CF units ----------------
    def batched(name, sp, fn, idx_sel=None, bs=BS_CF):
        A, U, OH, R = split[sp]
        nz = np.load(P(f"nun_{sp}.npz"))
        idx = np.arange(len(A)) if idx_sel is None else idx_sel
        nbat = (len(idx) + bs - 1) // bs
        est_f = P(f"_est_{name}.json")
        for k in range(nbat):
            fn_k = P(f"cf_{name}_{sp}_{k}.npz")
            if os.path.exists(fn_k):
                continue
            est = json.load(open(est_f))["t"] if os.path.exists(est_f) else 30.0
            sel = idx[k * bs:(k + 1) * bs]
            if not B_.ok(est * 1.2 + 5):
                return False
            t1 = time.time()
            Uc, Rc, cc, tt, ex = fn(A[sel], U, OH, R, nz["U"][sel], Upool[nz["nb"][sel]], sel, nz["c"][sel])
            np.savez_compressed(fn_k, U=Uc, R=Rc, c=cc, t=tt, **ex)
            json.dump({"t": time.time() - t1}, open(est_f, "w"))
            log(f"  {name}/{sp} batch {k + 1}/{nbat}: valid {np.isfinite(cc).mean():.3f} ({time.time() - t1:.0f}s)")
        return True

    def gather(name, sp, n, bs=BS_CF):
        parts = [np.load(P(f"cf_{name}_{sp}_{k}.npz")) for k in range((n + bs - 1) // bs)]
        out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files if k != "t"}
        out["t"] = float(np.mean([float(p["t"]) for p in parts]))
        return out

    def f_trace(**kw):
        def fn(A, U, OH, R, Un, S, sel, Nc):
            init = np.where(np.isfinite(Nc)[:, None], Un, U[A].astype(np.float64)) if kw.get("warm") else None
            Uc, Rc, cc, tt, ex = cf.trace(model, space, U[A], OH[A], R[A], B, nbrs=S if kw.get("gamma", a.gamma) > 0 else None,
                                          steps=a.steps, beta=kw.get("beta", a.beta), gamma=kw.get("gamma", a.gamma),
                                          use_rel=kw.get("use_rel", True), revert=kw.get("revert", True), init=init)
            return Uc, Rc, cc, tt, {"c_pre": ex["cost_pre"], "iters": ex["iters"]}
        return fn

    def f_wachter(A, U, OH, R, Un, S, sel, Nc):
        Uc, Rc, cc, tt = cf.wachter(model, prep, U[A], OH[A], R[A], B, steps=a.steps, beta=a.beta)
        return Uc, Rc, cc, tt, {}

    def f_gs(A, U, OH, R, Un, S, sel, Nc):
        Uc, Rc, cc, tt = cf.growing_spheres(model, prep, U[A], OH[A], R[A], B, seed=a.seed)
        return Uc, Rc, cc, tt, {}

    jobs = [("TRACE", "te", f_trace(), None, BS_CF), ("TRACEw", "te", f_trace(warm=True), None, BS_CF),
            ("Wachter", "te", f_wachter, None, BS_CF)]
    if a.with_val:
        jobs = [("TRACE", "va", f_trace(), None, BS_CF), ("Wachter", "va", f_wachter, None, BS_CF)] + jobs
    if a.full:
        bench = al["bench"]
        jobs += [("TRACE-noplaus", "te", f_trace(gamma=0.0), bench, BS_CF),
                 ("TRACE-noplausw", "te", f_trace(gamma=0.0, warm=True), bench, BS_CF),
                 ("TRACE-norel", "te", f_trace(use_rel=False), bench, BS_CF),
                 ("TRACE-norelw", "te", f_trace(use_rel=False, warm=True), bench, BS_CF),
                 ("TRACE-norevert", "te", f_trace(revert=False), bench, BS_CF),
                 ("TRACE-norevertw", "te", f_trace(revert=False, warm=True), bench, BS_CF),
                 ("GrowingSpheres", "te", f_gs, bench, 100)]
        for be in (0.01, 0.05, 0.25):
            for ga in (0.0, 0.05, 0.25):
                jobs.append((f"sens_b{be}_g{ga}", "te", f_trace(beta=be, gamma=ga), al["sens"], BS_CF))
    for name, sp, fn, sel, bs in jobs:
        if not batched(name, sp, fn, sel, bs):
            log(f"  [budget] paused in {name}/{sp}")
            return

    # ---------------- assemble results ----------------
    if os.path.exists(P("cf_quality.csv")) and os.path.exists(P("ig_agreement.json")):
        log("PIPELINE COMPLETE")
        return
    bva_d = cf.knn_dist(prep, Uva[bva], Uref)
    knn_norm = {"median": float(np.median(bva_d)), "p95": float(np.percentile(bva_d, 95))}
    res = {"te": {}}
    nz = np.load(P("nun_te.npz"))
    res["te"]["NUN"] = {"U": nz["U"], "R": nz["R"], "c": nz["c"], "t": float(nz["t"])}
    sig_np = prep.sigma
    mut_np = ~space.imm

    def combine(name, sel=None, lam=a.gamma / a.beta, bs=BS_CF):
        n = len(A_te) if sel is None else len(sel)
        ra, rb = gather(name, "te", n, bs), gather(name + "w", "te", n, bs)
        nbi = nz["nb"] if sel is None else nz["nb"][sel]

        def J(r):
            if lam == 0:
                return r["c"]
            P = np.zeros(n)
            for s0 in range(0, n, 512):
                S = Upool[nbi[s0:s0 + 512]].astype(np.float64)
                dk = (np.abs(r["U"][s0:s0 + 512, None, :] - S) / sig_np)[:, :, mut_np].sum(2)
                P[s0:s0 + 512] = np.sort(dk, 1)[:, :5].mean(1)
            return np.where(np.isfinite(r["c"]), r["c"] + lam * P, np.inf)
        pick = J(rb) < J(ra)
        out = {k: np.where(pick[:, None], rb[k], ra[k]) for k in ("U", "R")}
        out["c"] = np.where(pick, rb["c"], ra["c"])
        out["t"] = ra["t"] + rb["t"]
        out["warm_share"] = float(pick.mean())
        return out, ra

    res["te"]["TRACE"], res["te"]["TRACE-alertstart"] = combine("TRACE")
    res["te"]["Wachter"] = gather("Wachter", "te", len(A_te))

    q_rows = []
    for k in ("NUN", "TRACE", "TRACE-alertstart", "Wachter"):
        r = res["te"][k]
        q_rows.append(cf_quality(k, space, prep, Ute[A_te], Rte[A_te], r["U"], r["R"], r["c"], r["t"], Uref, knn_norm))
        if k == "TRACE":
            q_rows[-1]["warm_share"] = r["warm_share"]
        fpm = isfp_te
        for grp_name, m in (("false_alarms", fpm), ("true_alerts", ~fpm)):
            if m.any():
                row = cf_quality(f"{k}|{grp_name}", space, prep, Ute[A_te][m], Rte[A_te][m], r["U"][m], r["R"][m],
                                 r["c"][m], r["t"], Uref, knn_norm)
                q_rows.append(row)
    if a.full:
        bench = al["bench"]
        idx = A_te[bench]
        for k in ("TRACE-noplaus", "TRACE-norel", "TRACE-norevert"):
            r, _ = combine(k, sel=bench, lam=0.0 if k == "TRACE-noplaus" else a.gamma / a.beta)
            q_rows.append(cf_quality(k, space, prep, Ute[idx], Rte[idx], r["U"], r["R"], r["c"], r["t"], Uref, knn_norm))
        r = gather("GrowingSpheres", "te", len(bench), 100)
        q_rows.append(cf_quality("GrowingSpheres", space, prep, Ute[idx], Rte[idx], r["U"], r["R"], r["c"], r["t"], Uref, knn_norm))
        for k in ("TRACE", "TRACE-alertstart", "Wachter", "NUN"):
            r = res["te"][k]
            q_rows.append(cf_quality(k + "@bench", space, prep, Ute[idx], Rte[idx], r["U"][bench], r["R"][bench],
                                     r["c"][bench], r["t"], Uref, knn_norm))
        rows = []
        sidx = al["sens"]
        for be in (0.01, 0.05, 0.25):
            for ga in (0.0, 0.05, 0.25):
                r = gather(f"sens_b{be}_g{ga}", "te", len(sidx))
                row = cf_quality(f"beta={be},gamma={ga}", space, prep, Ute[A_te[sidx]], Rte[A_te[sidx]], r["U"], r["R"],
                                 r["c"], r["t"], Uref, knn_norm)
                row.update({"beta": be, "gamma": ga})
                rows.append(row)
        jsave(pd.DataFrame(rows), "sensitivity.csv")
    jsave(pd.DataFrame(q_rows), "cf_quality.csv")
    jsave(knn_norm, "knn_norm.json")
    log(pd.DataFrame(q_rows).to_string())

    # change profiles, case studies, IG agreement
    rT = res["te"]["TRACE"]
    U0, R0 = Ute[A_te].astype(np.float64), Rte[A_te]
    rel = np.abs(rT["U"] - U0) / prep.sigma
    ok = np.isfinite(rT["c"])
    grp = np.array([("FALSE ALARM" if f else names[t]) for f, t in zip(isfp_te, yte[A_te])])
    prof = pd.DataFrame(rel[ok], columns=prep.num_cols)
    prof["group"] = grp[ok]
    cnt = prof.groupby("group").size()
    prof = prof.groupby("group").mean()
    prof["n"] = cnt
    jsave(prof.reset_index(), "change_profile_TRACE.csv")
    rW = res["te"]["Wachter"]
    okW = np.isfinite(rW["c"])
    profW = pd.DataFrame((np.abs(rW["U"] - U0) / prep.sigma)[okW], columns=prep.num_cols)
    profW["group"] = grp[okW]
    jsave(profW.groupby("group").mean().reset_index(), "change_profile_Wachter.csv")
    pred_te = P_te.argmax(1)
    cases = []
    for labc, mask in (("false_alarm", isfp_te & ok), ("true_alert", ~isfp_te & ok)):
        ii = np.flatnonzero(mask)
        if len(ii) == 0:
            continue
        j = ii[np.argsort(rT["c"][ii])[len(ii) // 2]]
        for f in np.argsort(-rel[j])[:6]:
            if rel[j, f] < 1e-6:
                continue
            cases.append({"case": labc, "true_class": names[yte[A_te][j]], "pred_class": names[pred_te[A_te][j]],
                          "p_benign": float(P_te[A_te][j, B]), "cost": float(rT["c"][j]), "feature": prep.num_cols[f],
                          "original": float(R0[j, f]), "counterfactual": float(rT["R"][j, f]), "norm_change": float(rel[j, f])})
    jsave(pd.DataFrame(cases), "case_studies.csv")
    ig_idx = np.flatnonzero(ok)
    if len(ig_idx) > 2000:
        ig_idx = np.sort(rng.choice(ig_idx, 2000, replace=False))
    att = cf.integrated_gradients(model, Ute[A_te][ig_idx], OHte[A_te][ig_idx], pred_te[A_te][ig_idx], B,
                                  Utr[ytr == B].mean(0))
    mut = ~space.imm
    agg = []
    for k_, i in enumerate(ig_idx):
        a_ig, a_cf = np.abs(att[k_])[mut], rel[i][mut]
        rho = spearmanr(a_ig, a_cf)[0] if a_cf.std() > 0 and a_ig.std() > 0 else np.nan
        agg.append({"is_fp": bool(isfp_te[i]),
                    "top5_overlap": len(set(np.argsort(-a_ig)[:5]) & set(np.argsort(-a_cf)[:5])) / 5.0,
                    "top1_match": int(np.argmax(a_ig) == np.argmax(a_cf)), "spearman": rho})
    agg = pd.DataFrame(agg)
    jsave({"n": len(agg), "top5_overlap_mean": agg.top5_overlap.mean(), "top1_match_rate": agg.top1_match.mean(),
           "spearman_mean": agg.spearman.mean(), "random_top5_overlap": 5.0 / int(mut.sum()), "d_mutable": int(mut.sum()),
           "top5_overlap_fp": agg[agg.is_fp].top5_overlap.mean() if agg.is_fp.any() else np.nan,
           "top5_overlap_tp": agg[~agg.is_fp].top5_overlap.mean()}, "ig_agreement.json")
    log("PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
