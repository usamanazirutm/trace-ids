"""Turn experiment outputs into LaTeX table bodies, number macros and figures.

  python -m trace_ids.report --results results --out paper_assets
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DS = {"unsw": "UNSW-NB15", "cic": "CICIoT2023"}
TAG = {"unsw": "U", "cic": "C"}
DET = {"cnnbilstm": "CNN-BiLSTM", "mlp": "MLP"}
METHODS = [("GrowingSpheres", "Growing Spheres", "GS"), ("Wachter@bench", "Wachter-style", "Wachter"),
           ("NUN@bench", "Nearest benign (NUN)", "NUN"), ("TRACE-norel", "TRACE w/o relations", "TRACEnorel"),
           ("TRACE-noplaus", "TRACE w/o plausibility", "TRACEnoplaus"),
           ("TRACE-norevert", "TRACE w/o reversion", "TRACEnorevert"),
           ("TRACE-alertstart@bench", "TRACE alert start only", "TRACEalertstart"), ("TRACE@bench", "TRACE (full)", "TRACE")]


def rd(p):
    with open(p) as f:
        return json.load(f)


def nan(x):
    return x is None or (isinstance(x, float) and np.isnan(x))


def f2(x, d=2):
    return "--" if nan(x) else f"{x:.{d}f}"


def pct(x, d=1):
    return "--" if nan(x) else f"{100 * x:.{d}f}"


def tex_escape(s):
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


DIG = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven",
       "8": "eight", "9": "nine"}


def macro_name(k):
    return "".join(DIG.get(ch, ch) for ch in k if ch.isalnum())


class Macros(dict):
    def tex(self):
        return "\n".join(f"\\newcommand{{\\{macro_name(k)}}}{{{v}}}" for k, v in sorted(self.items()))


def run_dir(R, ds, det, seed):
    p = os.path.join(R, f"{ds}_{det}_s{seed}")
    return p if os.path.exists(os.path.join(p, "detection.json")) else None


def done_dir(R, ds, det, seed):
    p = run_dir(R, ds, det, seed)
    return p if p and os.path.exists(os.path.join(p, "cf_quality.csv")) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper_assets")
    a = ap.parse_args()
    R, O = a.results, a.out
    os.makedirs(os.path.join(O, "figures"), exist_ok=True)
    M, T = Macros(), {}

    # ----------------------------------------------------------- data & constraints
    rows, crow = [], []
    for ds in DS:
        d = run_dir(R, ds, "cnnbilstm", 0) or run_dir(R, ds, "mlp", 0)
        if not d:
            continue
        t = TAG[ds]
        st = rd(os.path.join(d, "data_stats.json"))
        ft = pd.read_csv(os.path.join(d, "features.csv"))
        ben = st.get("benign", "Normal" if ds == "unsw" else "BENIGN")
        cc = st["class_counts"]
        rows.append(f"{DS[ds]} & {st['raw']:,} & {st['after_dedup']:,} & {st['after_conflict']:,} & {st['used']:,} & "
                    f"{st['train']:,}/{st['val']:,}/{st['test']:,} & {st['classes']} & {cc[ben]:,} \\\\")
        for k, v in {"raw": st["raw"], "dedup": st["after_dedup"], "clean": st["after_conflict"], "used": st["used"],
                     "ntrain": st["train"], "nval": st["val"], "ntest": st["test"], "benign": cc[ben],
                     "dupremoved": st["raw"] - st["after_dedup"],
                     "conflictremoved": st["after_dedup"] - st["after_conflict"]}.items():
            M[f"{t}{k}"] = f"{v:,}"
        M[f"{t}classes"] = str(st["classes"])
        M[f"{t}nnum"] = str(len(ft))
        M[f"{t}nimm"] = str(int(ft.immutable.sum()))
        M[f"{t}nder"] = str(int(ft.derived.sum()))
        M[f"{t}nint"] = str(int(ft.integer.sum()))
        M[f"{t}nlog"] = str(int(ft.log1p.sum()))
        M[f"{t}nfree"] = str(int((~ft.immutable & ~ft.derived).sum()))
        if ds == "unsw":
            M["Ugeneric"] = f"{cc.get('Generic', 0):,}"
            M["Udos"] = f"{cc.get('DoS', 0):,}"
        c = pd.read_csv(os.path.join(d, "constraints.csv"))
        M[f"{t}nrelcand"] = str(len(c))
        M[f"{t}nrelenf"] = str(int(c.enforced.sum()))
        first = True
        for _, r in c.iterrows():
            rel = (r.relation.replace("_", r"\_").replace("*", r"$\cdot$").replace("<=", r"$\le$")
                   .replace("^2", r"$^2$"))
            crow.append(f"{DS[ds] if first else ''} & {rel} & {'derived' if r.kind == 'derive' else 'relational'} & "
                        f"{pct(r.train_satisfaction, 2)} & {pct(r.test_satisfaction, 2)} & {'yes' if r.enforced else 'no'} \\\\")
            first = False
        crow.append("\\midrule")
    T["datastats"] = "\n".join(rows)
    T["constraints"] = "\n".join(crow[:-1])

    # ----------------------------------------------------------- detection
    lines = []
    for ds in DS:
        t = TAG[ds]
        first = True
        for det in ("cnnbilstm", "mlp"):
            vals = [rd(os.path.join(run_dir(R, ds, det, s), "detection.json")) for s in range(3) if run_dir(R, ds, det, s)]
            if not vals:
                continue
            df = pd.DataFrame(vals)
            n = len(df)

            def ms(k, scale=100, d=2):
                if n > 1:
                    return f"{scale * df[k].mean():.{d}f}$\\pm${scale * df[k].std(ddof=1):.{d}f}"
                return f"{scale * df[k].mean():.{d}f}"
            lines.append(f"{DS[ds] if first else ''} & {DET[det]} ({n} seed{'s' if n > 1 else ''}) & {ms('accuracy')} & "
                         f"{ms('macro_f1')} & {ms('weighted_f1')} & {ms('alert_DR')} & {ms('alert_FPR')} & "
                         f"{ms('alert_AUROC', 1, 3)} \\\\")
            first = False
            tt = t + ("cnn" if det == "cnnbilstm" else "mlp")
            s0 = df.iloc[0]
            for k in ("accuracy", "macro_f1", "alert_DR", "alert_FPR", "weighted_f1"):
                M[f"{tt}{k.replace('_', '')}"] = f"{100 * df[k].mean():.2f}"
                M[f"{tt}{k.replace('_', '')}sz"] = f"{100 * s0[k]:.2f}"
            M[f"{tt}alertAUROC"] = f"{df['alert_AUROC'].mean():.3f}"
            M[f"{tt}infer"] = f"{s0['infer_ms_per_flow']:.3f}"
            M[f"{tt}params"] = f"{int(s0['params']):,}"
            M[f"{tt}nseeds"] = str(n)
            M[f"{tt}epochs"] = str(int(s0.get("epochs", 0)))
            M[f"{tt}traintime"] = f"{s0['train_time_s'] / 60:.1f}"
        for dd in (run_dir(R, ds, "cnnbilstm", 0), run_dir(R, ds, "mlp", 0)):
            if dd and os.path.exists(os.path.join(dd, "reference_models.csv")):
                ref = pd.read_csv(os.path.join(dd, "reference_models.csv"))
                for _, r in ref.iterrows():
                    lines.append(f" & {r.model} & {100 * r.accuracy:.2f} & {100 * r.macro_f1:.2f} & {100 * r.weighted_f1:.2f} & "
                                 f"{100 * r.alert_DR:.2f} & {100 * r.alert_FPR:.2f} & {r.alert_AUROC:.3f} \\\\")
                    tt = t + {"XGBoost": "xgb", "RandomForest": "rf"}[r.model]
                    M[f"{tt}macrof1"] = f"{100 * r.macro_f1:.2f}"
                    M[f"{tt}alertFPR"] = f"{100 * r.alert_FPR:.2f}"
                    M[f"{tt}alertDR"] = f"{100 * r.alert_DR:.2f}"
                    M[f"{tt}accuracy"] = f"{100 * r.accuracy:.2f}"
                break
        lines.append("\\midrule")
    T["detection"] = "\n".join(lines[:-1])

    # ----------------------------------------------------------- CF quality (benchmark subset, CNN-BiLSTM)
    lines, fb = [], []
    for ds in DS:
        d = done_dir(R, ds, "cnnbilstm", 0)
        if not d:
            continue
        t = TAG[ds]
        q = pd.read_csv(os.path.join(d, "cf_quality.csv")).set_index("method")
        M[f"{t}nbench"] = f"{int(q.loc['TRACE@bench'].n):,}"
        first = True
        for key, lab, mk in METHODS:
            if key not in q.index:
                continue
            r = q.loc[key]
            lines.append(f"{DS[ds] if first else ''} & {lab} & {pct(r.validity)} & {pct(r.feasible_all)} & "
                         f"{pct(r.valid_and_feasible)} & {f2(r.get('proximity_median'))} & {f2(r.get('sparsity_mean'), 1)} & "
                         f"{f2(r.get('plaus_rel_knn_median'))} & {pct(r.get('in_distribution'))} & {f2(r.time_ms, 1)} \\\\")
            first = False
            p = t + mk
            M[f"{p}validity"] = pct(r.validity)
            M[f"{p}feasible"] = pct(r.feasible_all)
            M[f"{p}vf"] = pct(r.valid_and_feasible)
            M[f"{p}prox"] = f2(r.get("proximity_median"))
            M[f"{p}spars"] = f2(r.get("sparsity_mean"), 1)
            M[f"{p}plaus"] = f2(r.get("plaus_rel_knn_median"))
            M[f"{p}indist"] = pct(r.get("in_distribution"))
            M[f"{p}time"] = f2(r.time_ms, 1)
            for k in ("immutable", "integer", "bounds", "relational", "derived"):
                M[f"{p}feas{k}"] = pct(r[f"feasible_{k}"])
        lines.append("\\midrule")
        first = True
        for key, lab in [("GrowingSpheres", "Growing Spheres"), ("Wachter@bench", "Wachter-style"),
                         ("TRACE-norel", "TRACE w/o relations"), ("TRACE@bench", "TRACE (full)")]:
            if key not in q.index:
                continue
            r = q.loc[key]
            fb.append(f"{DS[ds] if first else ''} & {lab} & {pct(r.feasible_immutable)} & {pct(r.feasible_integer)} & "
                      f"{pct(r.feasible_bounds)} & {pct(r.feasible_relational)} & {pct(r.feasible_derived)} & "
                      f"{pct(r.feasible_all)} \\\\")
            first = False
        fb.append("\\midrule")
        for k in ("TRACE", "Wachter", "NUN"):
            r = q.loc[k]
            M[f"{t}{k}nall"] = f"{int(r.n):,}"
            M[f"{t}{k}validityall"] = pct(r.validity)
            M[f"{t}{k}feasall"] = pct(r.feasible_all)
            M[f"{t}{k}vfall"] = pct(r.valid_and_feasible)
            M[f"{t}{k}proxall"] = f2(r.get("proximity_median"))
            M[f"{t}{k}sparsall"] = f2(r.get("sparsity_mean"), 1)
            M[f"{t}{k}indistall"] = pct(r.get("in_distribution"))
            M[f"{t}{k}timeall"] = f2(r.time_ms, 1)
    T["cfquality"] = "\n".join(lines[:-1])
    T["feasbreak"] = "\n".join(fb[:-1])

    # by alert type (false alarms vs true alerts), CNN-BiLSTM, all explained test alerts
    lines = []
    for ds in DS:
        d = done_dir(R, ds, "cnnbilstm", 0)
        if not d:
            continue
        t = TAG[ds]
        q = pd.read_csv(os.path.join(d, "cf_quality.csv")).set_index("method")
        first = True
        for grp, glab in (("false_alarms", "False alarms"), ("true_alerts", "True alerts")):
            for k, lab in (("Wachter", "Wachter-style"), ("NUN", "NUN"), ("TRACE", "TRACE")):
                key = f"{k}|{grp}"
                if key not in q.index:
                    continue
                r = q.loc[key]
                lines.append(f"{DS[ds] if first else ''} & {glab} ({int(r.n):,}) & {lab} & {pct(r.validity)} & "
                             f"{pct(r.valid_and_feasible)} & {f2(r.get('proximity_median'))} & {f2(r.get('sparsity_mean'), 1)} & "
                             f"{pct(r.get('in_distribution'))} \\\\")
                first = False
                g = "fa" if grp == "false_alarms" else "ta"
                M[f"{t}{k}{g}validity"] = pct(r.validity)
                M[f"{t}{k}{g}prox"] = f2(r.get("proximity_median"))
                M[f"{t}{k}{g}spars"] = f2(r.get("sparsity_mean"), 1)
                M[f"{t}{k}{g}n"] = f"{int(r.n):,}"
        lines.append("\\midrule")
    T["bygroup"] = "\n".join(lines[:-1])

    # ----------------------------------------------------------- MLP robustness across seeds
    lines = []
    for ds in DS:
        t = TAG[ds]
        qs = [pd.read_csv(os.path.join(done_dir(R, ds, "mlp", s), "cf_quality.csv")).set_index("method")
              for s in range(3) if done_dir(R, ds, "mlp", s)]
        if not qs:
            continue
        first = True
        for k, lab in (("Wachter", "Wachter-style"), ("NUN", "NUN"), ("TRACE", "TRACE")):
            X = pd.DataFrame([q.loc[k] for q in qs])

            def ms(c, scale=100, d=1):
                v = X[c].astype(float)
                return f"{scale * v.mean():.{d}f}" + (f"$\\pm${scale * v.std(ddof=1):.{d}f}" if len(v) > 1 else "")
            lines.append(f"{DS[ds] if first else ''} & {lab} & {ms('validity')} & {ms('valid_and_feasible')} & "
                         f"{ms('proximity_median', 1, 2)} & {ms('sparsity_mean', 1, 1)} & {ms('plaus_rel_knn_median', 1, 2)} & "
                         f"{ms('in_distribution')} \\\\")
            first = False
            M[f"{t}mlp{k}vf"] = f"{100 * X['valid_and_feasible'].astype(float).mean():.1f}"
            M[f"{t}mlp{k}validity"] = f"{100 * X['validity'].astype(float).mean():.1f}"
            M[f"{t}mlp{k}prox"] = f"{X['proximity_median'].astype(float).mean():.2f}"
            M[f"{t}mlp{k}spars"] = f"{X['sparsity_mean'].astype(float).mean():.1f}"
        M[f"{t}mlpnseedscf"] = str(len(qs))
        lines.append("\\midrule")
    T["mlpcf"] = "\n".join(lines[:-1])

    # ----------------------------------------------------------- sensitivity
    lines = []
    for ds in ("unsw",):
        d = done_dir(R, ds, "cnnbilstm", 0)
        if not d or not os.path.exists(os.path.join(d, "sensitivity.csv")):
            continue
        s = pd.read_csv(os.path.join(d, "sensitivity.csv"))
        M[f"{TAG[ds]}nsens"] = f"{int(s.n.iloc[0]):,}"
        first = True
        for _, r in s.iterrows():
            lines.append(f"{DS[ds] if first else ''} & {r.beta:g} & {r.gamma:g} & {pct(r.validity)} & {f2(r.get('proximity_median'))} & "
                         f"{f2(r.get('sparsity_mean'), 1)} & {f2(r.get('plaus_rel_knn_median'))} & {pct(r.get('in_distribution'))} \\\\")
            first = False
            key = f"{TAG[ds]}sens{'abc'[[0.01, 0.05, 0.25].index(r.beta)]}{'abc'[[0.0, 0.05, 0.25].index(r.gamma)]}"
            M[f"{key}validity"] = pct(r.validity)
            M[f"{key}prox"] = f2(r.get("proximity_median"))
            M[f"{key}indist"] = pct(r.get("in_distribution"))
            M[f"{key}spars"] = f2(r.get("sparsity_mean"), 1)
        lines.append("\\midrule")
    T["sens"] = "\n".join(lines[:-1])

    # ----------------------------------------------------------- IG agreement
    lines = []
    for ds in DS:
        d = done_dir(R, ds, "cnnbilstm", 0)
        if not d or not os.path.exists(os.path.join(d, "ig_agreement.json")):
            continue
        g = rd(os.path.join(d, "ig_agreement.json"))
        t = TAG[ds]
        lines.append(f"{DS[ds]} & {g['n']:,} & {g['top5_overlap_mean']:.3f} & {g['random_top5_overlap']:.3f} & "
                     f"{100 * g['top1_match_rate']:.1f} & {g['spearman_mean']:.3f} & {f2(g['top5_overlap_fp'], 3)} & "
                     f"{g['top5_overlap_tp']:.3f} \\\\")
        M[f"{t}IGn"] = f"{g['n']:,}"
        M[f"{t}IGtopfive"] = f"{g['top5_overlap_mean']:.3f}"
        M[f"{t}IGrandom"] = f"{g['random_top5_overlap']:.3f}"
        M[f"{t}IGtopone"] = f"{100 * g['top1_match_rate']:.1f}"
        M[f"{t}IGrho"] = f"{g['spearman_mean']:.3f}"
        M[f"{t}IGfa"] = f2(g["top5_overlap_fp"], 3)
        M[f"{t}IGta"] = f"{g['top5_overlap_tp']:.3f}"
    T["ig"] = "\n".join(lines)

    # ----------------------------------------------------------- case studies
    lines = []
    for ds in DS:
        d = done_dir(R, ds, "cnnbilstm", 0)
        if not d:
            continue
        c = pd.read_csv(os.path.join(d, "case_studies.csv"))
        for case, cdf in c.groupby("case", sort=False):
            r0 = cdf.iloc[0]
            what = "false alarm" if case == "false_alarm" else "true alert"
            lines.append(f"\\multicolumn{{4}}{{@{{}}l}}{{\\textit{{{DS[ds]}, {what}: true {tex_escape(r0.true_class)}, "
                         f"predicted {tex_escape(r0.pred_class)}, $p_b={r0.p_benign:.3f}$, $d={r0.cost:.2f}$}}}} \\\\")
            for _, r in cdf.head(5).iterrows():
                def fm(v):
                    if float(v).is_integer():
                        return f"{v:,.0f}"
                    return f"{v:.4g}"
                lines.append(f"\\quad {tex_escape(r.feature)} & {fm(r.original)} & {fm(r.counterfactual)} & {r.norm_change:.2f} \\\\")
            lines.append("\\midrule")
    T["cases"] = "\n".join(lines[:-1])

    # ----------------------------------------------------------- figures
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "legend.fontsize": 6.5, "pdf.fonttype": 42})
    marks = {"GrowingSpheres": ("o", "#7f7f7f"), "Wachter@bench": ("s", "#ff7f0e"), "NUN@bench": ("^", "#8c564b"),
             "TRACE-norel": ("v", "#9467bd"), "TRACE-noplaus": ("D", "#17becf"), "TRACE-norevert": ("P", "#bcbd22"),
             "TRACE-alertstart@bench": ("X", "#e377c2"),
             "TRACE@bench": ("*", "#d62728")}
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.1))
    for ax, ds in zip(axs, DS):
        d = done_dir(R, ds, "cnnbilstm", 0)
        if not d:
            ax.set_visible(False)
            continue
        q = pd.read_csv(os.path.join(d, "cf_quality.csv")).set_index("method")
        for key, lab, _ in METHODS:
            if key not in q.index:
                continue
            r = q.loc[key]
            mk, col = marks[key]
            feas = r.feasible_all >= 0.999
            ax.scatter(r.proximity_median, r.plaus_rel_knn_median, s=25 + 6 * r.sparsity_mean, marker=mk,
                       facecolors=col if feas else "none", edgecolors=col, linewidths=1.2, label=lab, zorder=3)
        ax.set_xlabel("Proximity (median MAD-$L_1$)")
        ax.set_ylabel("Relative 5-NN distance to benign")
        ax.set_title(DS[ds])
        ax.grid(alpha=0.3)
    h, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(os.path.join(O, "figures", "tradeoff.pdf"))
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.6))
    im = None
    for ax, ds in zip(axs, DS):
        d = done_dir(R, ds, "cnnbilstm", 0)
        if not d:
            ax.set_visible(False)
            continue
        p = pd.read_csv(os.path.join(d, "change_profile_TRACE.csv")).set_index("group")
        if "n" in p.columns:
            p = p[p["n"] >= 10].drop(columns=["n"])
        order = ["FALSE ALARM"] + sorted(g for g in p.index if g != "FALSE ALARM")
        p = p.loc[[g for g in order if g in p.index]]
        top = p.mean(0).sort_values(ascending=False).index[:10]
        P_ = p[top].values
        P_ = P_ / np.maximum(P_.max(1, keepdims=True), 1e-9)
        im = ax.imshow(P_, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(top)))
        ax.set_xticklabels([str(x) for x in top], rotation=60, ha="right", fontsize=6)
        ax.set_yticks(range(len(p.index)))
        ax.set_yticklabels([g if g == "FALSE ALARM" else g.replace("_", "-").title() for g in p.index], fontsize=5.5)
        ax.set_title(DS[ds])
    if im is not None:
        fig.colorbar(im, ax=list(axs), fraction=0.025, pad=0.01, label="Mean normalised change (row-scaled)")
    fig.savefig(os.path.join(O, "figures", "change_profiles.pdf"), bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(O, "generated_macros.tex"), "w") as f:
        f.write(M.tex() + "\n")
    with open(os.path.join(O, "generated_tables.json"), "w") as f:
        json.dump(T, f, indent=1)
    print(f"{len(M)} macros; tables: {[k for k, v in T.items() if v]}")


if __name__ == "__main__":
    main()
