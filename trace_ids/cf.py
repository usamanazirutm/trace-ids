"""Counterfactual generators for flow-based IDS alerts.

TRACE          constraint-aware, prototype-guided gradient search in the verified feasible space (proposed)
Wachter        unconstrained gradient search on all numeric inputs (Wachter et al., 2017; hinge form)
GrowingSpheres model-agnostic random search in growing L2 layers + sparsification (Laugel et al., 2018)
NUN            nearest benign training flow with matching immutable attributes (grafted)
"""
import time

import numpy as np
import torch


def _margin(logits, b):
    other = logits.clone()
    other[:, b] = -float("inf")
    return other.max(1).values - logits[:, b]  # > 0  -> not benign


def _freeze(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)


def _raw_from(prep, U, U0, R0):
    """Raw values of a counterfactual given in u-space; untouched coordinates keep exact raw values."""
    R = prep.raw_np(U.astype(np.float64))
    same = np.abs(U - U0) < 1e-9
    return np.where(same, R0, R)


def mad_l1(prep, U, U0):
    return (np.abs(U - U0) / prep.sigma).sum(1)


# ----------------------------------------------------------------------------------------
def trace(model, space, U0, OH, R0, benign, nbrs=None, use_rel=True, steps=300, lr=0.02,
          beta=0.05, gamma=0.05, kappa=0.5, k_plaus=5, revert=True, bs=1024, log=None, init=None):
    """TRACE: projected gradient search over the verified feasible space.

    Objective (valid region):  beta * d(x, x') + gamma * d_kNN(x', S(x)),
    d = MAD-normalised L1 over numeric inputs, S(x) = benign training flows nearest to x that
    share its immutable attributes. Every iterate is repaired (immutables, bounds, verified
    relations), integer-rounded and re-scored; the best valid iterate is kept and then
    greedily sparsified. Returns (U_cf, R_cf, cost, t_per_alert, extras) where cost is the
    proximity d(x, x') of the returned counterfactual (inf when none was found)."""
    _freeze(model)
    prep = space.prep
    sigma = torch.as_tensor(prep.sigma, dtype=torch.float64)
    free_np = space.free_rel if use_rel else space.free_norel
    free_t = torch.as_tensor(free_np)
    mut_t = torch.as_tensor(~space.imm)
    n = len(U0)
    U_out, R_out = U0.astype(np.float64).copy(), R0.copy()
    cost, cost_pre = np.full(n, np.inf), np.full(n, np.inf)
    U_pre = U0.astype(np.float64).copy()
    iters = np.zeros(n)
    lam = (gamma / beta) if (nbrs is not None and gamma > 0) else 0.0
    t0 = time.time()

    def plaus(u, S):
        dk = ((u[:, None, :] - S).abs() / sigma)[:, :, mut_t].sum(2)
        return dk.topk(min(k_plaus, dk.shape[1]), dim=1, largest=False).values.mean(1)

    def score(r, r0, oh, u0, S):
        u = prep.u_t(r).clamp(0, 1)
        ok = model(torch.cat([u.float(), oh], 1)).argmax(1) == benign
        d = ((u - u0).abs() / sigma).sum(1)
        crit = d + lam * plaus(u, S) if lam > 0 else d
        return u, ok, d, crit

    for s in range(0, n, bs):
        u0 = torch.as_tensor(U0[s:s + bs], dtype=torch.float64)
        r0 = torch.as_tensor(R0[s:s + bs], dtype=torch.float64)
        oh = torch.as_tensor(OH[s:s + bs])
        S = None if lam == 0 else torch.as_tensor(nbrs[s:s + bs], dtype=torch.float64)
        v0 = u0 if init is None else torch.as_tensor(init[s:s + bs], dtype=torch.float64)
        v = v0.clone().requires_grad_(True)  # warm start: search from a feasible benign anchor
        opt = torch.optim.Adam([v], lr=lr)
        m = len(u0)
        best_crit = torch.full((m,), float("inf"), dtype=torch.float64)
        best_d = best_crit.clone()
        best_u, best_r = u0.clone(), r0.clone()
        hist = []
        for it in range(steps):
            u = torch.where(free_t, v.clamp(0, 1), u0)
            r = space.repair(prep.raw_t(u), r0, use_rel=use_rel, round_int=False)
            uin = prep.u_t(r).clamp(0, 1)
            logits = model(torch.cat([uin.float(), oh], 1))
            hinge = (_margin(logits, benign) + kappa).clamp(min=0)
            dist = ((uin - u0).abs() / sigma).sum(1)
            loss = hinge + beta * dist.float()
            if lam > 0:
                loss = loss + gamma * plaus(uin, S).float()
            with torch.no_grad():  # score the rounded, fully repaired candidate
                rr = space.repair(r.detach(), r0, use_rel=use_rel, round_int=True)
                ur, ok, dr, cr = score(rr, r0, oh, u0, S)
                upd = ok & (cr < best_crit)
                best_crit = torch.where(upd, cr, best_crit)
                best_d = torch.where(upd, dr, best_d)
                best_u[upd], best_r[upd] = ur[upd], rr[upd]
            opt.zero_grad()
            loss.sum().backward()
            opt.step()
            hist.append(best_crit.clone())
            if it >= 120 and it % 20 == 0 and torch.isfinite(best_crit).all():
                prev = hist[-60]
                if ((prev - best_crit) / best_crit.clamp(min=1e-9)).max() < 5e-3:
                    break
        iters[s:s + bs] = it + 1
        U_pre[s:s + bs], cost_pre[s:s + bs] = best_u.numpy(), best_d.numpy()
        if revert:  # greedy reversion of the smallest changes (sparsity), never increasing the objective
            with torch.no_grad():
                vi = torch.isfinite(best_crit).nonzero().flatten()
                if len(vi):
                    rc, r0v, u0v, ohv = best_r[vi].clone(), r0[vi], u0[vi], oh[vi]
                    Sv = None if S is None else S[vi]
                    crit_c = best_crit[vi].clone()
                    delta = (best_u[vi] - u0v).abs() / sigma
                    delta[:, ~free_t] = float("inf")
                    order = delta.argsort(1)
                    ar = torch.arange(len(vi))
                    for kk in range(int(free_np.sum())):
                        j = order[:, kk]
                        trial = rc.clone()
                        trial[ar, j] = r0v[ar, j]
                        trial = space.repair(trial, r0v, use_rel=use_rel, round_int=True)
                        _, ok, _, cr = score(trial, r0v, ohv, u0v, Sv)
                        acc = ok & (cr <= crit_c + 1e-9)
                        rc[acc], crit_c[acc] = trial[acc], cr[acc]
                    uf, okf, df, _ = score(rc, r0v, ohv, u0v, Sv)
                    keep = okf  # safety: keep the reverted solution only if still valid
                    best_r[vi[keep]], best_u[vi[keep]], best_d[vi[keep]] = rc[keep], uf[keep], df[keep]
        U_out[s:s + bs], R_out[s:s + bs] = best_u.numpy(), best_r.numpy()
        cost[s:s + bs] = best_d.numpy()
        if log:
            log(f"    TRACE batch {s // bs + 1}: valid {np.isfinite(cost[s:s + bs]).mean():.3f} it {it + 1}")
    t = (time.time() - t0) / max(n, 1)
    return U_out, R_out, cost, t, {"U_pre": U_pre, "cost_pre": cost_pre, "iters": iters}


# ----------------------------------------------------------------------------------------
def wachter(model, prep, U0, OH, R0, benign, steps=300, lr=0.02, beta=0.05, kappa=0.5, bs=1024):
    _freeze(model)
    sigma = torch.as_tensor(prep.sigma, dtype=torch.float32)
    n = len(U0)
    U_out = U0.astype(np.float64).copy()
    cost = np.full(n, np.inf)
    t0 = time.time()
    for s in range(0, n, bs):
        u0 = torch.as_tensor(U0[s:s + bs])
        oh = torch.as_tensor(OH[s:s + bs])
        v = u0.clone().requires_grad_(True)
        opt = torch.optim.Adam([v], lr=lr)
        best_c = torch.full((len(u0),), float("inf"))
        best_u = u0.clone()
        hist = []
        for it in range(steps):
            u = v.clamp(0, 1)
            logits = model(torch.cat([u, oh], 1))
            dist = ((u - u0).abs() / sigma).sum(1)
            with torch.no_grad():
                upd = (logits.argmax(1) == benign) & (dist < best_c)
                best_c = torch.where(upd, dist, best_c)
                best_u[upd] = u[upd].detach()
            loss = (_margin(logits, benign) + kappa).clamp(min=0) + beta * dist
            opt.zero_grad()
            loss.sum().backward()
            opt.step()
            hist.append(best_c.clone())
            if it >= 120 and it % 20 == 0 and torch.isfinite(best_c).all():
                prev = hist[-60]
                if ((prev - best_c) / best_c.clamp(min=1e-9)).max() < 5e-3:
                    break
        U_out[s:s + bs] = best_u.double().numpy()
        cost[s:s + bs] = best_c.double().numpy()
    R_out = _raw_from(prep, U_out, U0.astype(np.float64), R0)
    return U_out, R_out, cost, (time.time() - t0) / max(n, 1)


# ----------------------------------------------------------------------------------------
@torch.no_grad()
def growing_spheres(model, prep, U0, OH, R0, benign, n_samp=256,
                    radii=(0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2), bs=32, seed=0):
    _freeze(model)
    g = torch.Generator().manual_seed(seed)
    sigma = torch.as_tensor(prep.sigma, dtype=torch.float32)
    n, d = U0.shape
    U_out = U0.astype(np.float64).copy()
    cost = np.full(n, np.inf)
    t0 = time.time()
    for s in range(0, n, bs):
        u0 = torch.as_tensor(U0[s:s + bs])
        oh = torch.as_tensor(OH[s:s + bs])
        m = len(u0)
        best_u, best_c = u0.clone(), torch.full((m,), float("inf"))
        found = torch.zeros(m, dtype=torch.bool)
        r_prev = 0.0
        for r in radii:
            act = (~found).nonzero().flatten()
            if len(act) == 0:
                break
            dirs = torch.randn(len(act), n_samp, d, generator=g)
            dirs = dirs / dirs.norm(dim=2, keepdim=True)
            rad = r_prev + (r - r_prev) * torch.rand(len(act), n_samp, 1, generator=g)
            cand = (u0[act][:, None, :] + dirs * rad).clamp(0, 1)
            ohr = oh[act][:, None, :].expand(-1, n_samp, -1)
            lg = model(torch.cat([cand, ohr], 2).reshape(len(act) * n_samp, -1))
            ok = (lg.argmax(1) == benign).reshape(len(act), n_samp)
            dist = ((cand - u0[act][:, None, :]).abs() / sigma).sum(2)
            dist[~ok] = float("inf")
            c, j = dist.min(1)
            hit = torch.isfinite(c)
            idx = act[hit]
            best_c[idx], best_u[idx] = c[hit], cand[hit, j[hit]]
            found[idx] = True
            r_prev = r
        # sparsification: revert the smallest changes while validity is preserved
        if found.any():
            fi = found.nonzero().flatten()
            cur = best_u[fi].clone()
            order = (cur - u0[fi]).abs().argsort(1)
            for k in range(d):
                j = order[:, k]
                trial = cur.clone()
                ar = torch.arange(len(fi))
                trial[ar, j] = u0[fi][ar, j]
                ok = model(torch.cat([trial, oh[fi]], 1)).argmax(1) == benign
                cur[ok] = trial[ok]
            best_u[fi] = cur
            best_c[fi] = ((cur - u0[fi]).abs() / sigma).sum(1)
        U_out[s:s + bs], cost[s:s + bs] = best_u.double().numpy(), best_c.double().numpy()
    R_out = _raw_from(prep, U_out, U0.astype(np.float64), R0)
    return U_out, R_out, cost, (time.time() - t0) / max(n, 1)


# ----------------------------------------------------------------------------------------
def immutable_key(space, R, OH):
    prep = space.prep
    if prep.name == "unsw":
        imm = R[:, space.imm]
        return [tuple(a) + tuple(np.flatnonzero(o)) for a, o in zip(imm, OH)]
    cols = [prep.ix[c] for c in ("Protocol Type", "Number") if c in prep.ix]
    return [tuple(a) for a in R[:, cols]]


@torch.no_grad()
def nun(model, space, U0, OH, R0, benign, Upool, OHpool, Rpool, k=10, k_nbrs=32):
    """Nearest benign training flow (predicted benign) with matching immutable key, grafted
    with the alert's immutable attributes; first valid of the k nearest is returned."""
    _freeze(model)
    prep = space.prep
    t0 = time.time()
    n = len(U0)
    sig = torch.as_tensor(prep.sigma, dtype=torch.float32)
    mut = torch.as_tensor(~space.imm)
    kq, kp = immutable_key(space, R0, OH), immutable_key(space, Rpool, OHpool)
    groups = {}
    for i, key in enumerate(kp):
        groups.setdefault(key, []).append(i)
    all_idx = np.arange(len(Upool))
    U_out, R_out = U0.astype(np.float64).copy(), R0.copy()
    cost = np.full(n, np.inf)
    nbrs = np.zeros((n, k_nbrs), dtype=np.int64)
    by_key = {}
    for i, key in enumerate(kq):
        by_key.setdefault(key, []).append(i)
    for key, qi in by_key.items():
        qi = np.array(qi)
        pi = np.array(groups.get(key, all_idx))
        for s in range(0, len(qi), 256):
            q = qi[s:s + 256]
            A = torch.as_tensor(U0[q])[:, mut] / sig[mut]
            B = torch.as_tensor(Upool[pi])[:, mut] / sig[mut]
            D = torch.cdist(A, B, p=1)
            kn = min(k_nbrs, len(pi))
            nb = D.topk(kn, largest=False).indices.numpy()
            if kn < k_nbrs:
                nb = np.concatenate([nb, np.repeat(nb[:, -1:], k_nbrs - kn, 1)], 1)
            nbrs[q] = pi[nb]
            kk = min(k, len(pi))
            nn_idx = nb[:, :kk]
            for t in range(kk):
                todo = ~np.isfinite(cost[q])
                if not todo.any():
                    break
                qq = q[todo]
                cand = pi[nn_idx[todo, t]]
                Rg = np.where(space.imm, R0[qq], Rpool[cand])
                Ug = np.clip(prep.u_np(Rg), 0, 1)
                pred = model(torch.cat([torch.as_tensor(Ug, dtype=torch.float32),
                                        torch.as_tensor(OH[qq])], 1)).argmax(1).numpy()
                ok = pred == benign
                U_out[qq[ok]], R_out[qq[ok]] = Ug[ok], Rg[ok]
                cost[qq[ok]] = mad_l1(prep, Ug[ok], U0[qq[ok]].astype(np.float64))
    return U_out, R_out, cost, (time.time() - t0) / max(n, 1), nbrs


# ----------------------------------------------------------------------------------------
@torch.no_grad()
def knn_dist(prep, Uq, Uref, k=5, chunk=1024):
    sig = torch.as_tensor(prep.sigma, dtype=torch.float32)
    B = torch.as_tensor(Uref, dtype=torch.float32) / sig
    out = []
    for s in range(0, len(Uq), chunk):
        A = torch.as_tensor(Uq[s:s + chunk], dtype=torch.float32) / sig
        out.append(torch.cdist(A, B, p=1).topk(k, largest=False).values.mean(1))
    return torch.cat(out).numpy() if out else np.zeros(0)


def integrated_gradients(model, U, OH, target, benign, baseline, steps=32, bs=512):
    _freeze(model)
    base = torch.as_tensor(baseline, dtype=torch.float32)
    out = np.zeros_like(U, dtype=np.float64)
    for s in range(0, len(U), bs):
        u = torch.as_tensor(U[s:s + bs])
        oh = torch.as_tensor(OH[s:s + bs])
        tg = torch.as_tensor(target[s:s + bs]).long()
        acc = torch.zeros_like(u)
        for a in torch.linspace(1.0 / steps, 1.0, steps):
            ui = (base + a * (u - base)).requires_grad_(True)
            lg = model(torch.cat([ui, oh], 1))
            obj = lg[torch.arange(len(u)), tg] - lg[:, benign]
            acc += torch.autograd.grad(obj.sum(), ui)[0]
        out[s:s + bs] = ((u - base) * acc / steps).double().numpy()
    return out
