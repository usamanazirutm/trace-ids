"""Detection and counterfactual-quality metrics."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score

from . import cf


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def detection_metrics(y, proba, benign):
    pred = proba.argmax(1)
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    is_att, alert = y != benign, pred != benign
    return {
        "accuracy": accuracy_score(y, pred), "macro_precision": p, "macro_recall": r, "macro_f1": f,
        "weighted_f1": f1_score(y, pred, average="weighted"),
        "alert_DR": float(alert[is_att].mean()), "alert_FPR": float(alert[~is_att].mean()),
        "alert_AUROC": roc_auc_score(is_att, 1 - proba[:, benign]),
    }


def per_class(y, pred, names):
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=range(len(names)), zero_division=0)
    return pd.DataFrame({"class": names, "precision": p, "recall": r, "f1": f, "support": s})


def subsample_alerts(y, pred, benign, n_tp, rng):
    alert = np.flatnonzero(pred != benign)
    fp = alert[y[alert] == benign]
    tp = alert[y[alert] != benign]
    if len(tp) > n_tp:  # stratified by true class
        cls, cnt = np.unique(y[tp], return_counts=True)
        take = np.maximum(1, np.round(cnt / cnt.sum() * n_tp)).astype(int)
        tp = np.concatenate([rng.choice(tp[y[tp] == c], min(k, (y[tp] == c).sum()), replace=False)
                             for c, k in zip(cls, take)])
    return np.sort(fp), np.sort(tp)


def cf_quality(name, space, prep, U0, R0, Ucf, Rcf, cost, t_alert, Uref, knn_norm):
    valid = np.isfinite(cost)
    chk = space.check(Rcf, R0)
    row = {"method": name, "n": len(U0), "validity": valid.mean(), "time_ms": 1000 * t_alert}
    for k in ("all", "immutable", "integer", "bounds", "relational", "derived"):
        row[f"feasible_{k}"] = chk[k][valid].mean() if valid.any() else np.nan
    row["valid_and_feasible"] = (valid & chk["all"]).mean()
    if valid.any():
        dist = cf.mad_l1(prep, Ucf[valid], U0[valid].astype(np.float64))
        ch = (np.abs(Ucf[valid] - U0[valid]) > 1e-6).sum(1)
        kd = cf.knn_dist(prep, Ucf[valid].astype(np.float32), Uref) / knn_norm["median"]
        row.update({"proximity_median": np.median(dist), "proximity_mean": dist.mean(),
                    "sparsity_mean": ch.mean(), "sparsity_median": np.median(ch),
                    "plaus_rel_knn_median": np.median(kd),
                    "in_distribution": (kd <= knn_norm["p95"] / knn_norm["median"]).mean()})
    return row
