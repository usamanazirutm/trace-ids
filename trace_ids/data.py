"""Data loading, cleaning, class capping and stratified splitting.

Supported datasets
  * UNSW-NB15 partitioned CSVs (UNSW_NB15_training-set.csv + UNSW_NB15_testing-set.csv)
  * CICIoT2023 (MERGED_CSV release or the original part-*.csv files; CSV directory or parquet)
"""
import glob
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

UNSW_DROP = ["id", "label", "stcpb", "dtcpb"]  # identifiers, binary label, random TCP ISNs


def load_unsw(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files in {data_dir}")
    df = pd.concat([pd.read_csv(f, encoding="utf-8-sig", low_memory=False) for f in files],
                   ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    y = (df["attack_cat"].astype(str).str.strip()
         .replace({"Backdoors": "Backdoor", "nan": "Normal", "": "Normal"}))
    X = df.drop(columns=[c for c in UNSW_DROP + ["attack_cat"] if c in df.columns])
    for c in ["proto", "service", "state"]:
        X[c] = X[c].astype(str).str.strip()
    return X.reset_index(drop=True), y.reset_index(drop=True), "Normal"


def load_ciciot(path):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.csv")))
        df = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    elif path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False)
    lab = "Label" if "Label" in df.columns else "label"
    y = df[lab].astype(str).str.strip()
    X = df.drop(columns=[c for c in [lab, "family", "is_attack"] if c in df.columns])
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    ok = X.notna().all(axis=1).values
    X, y = X[ok].reset_index(drop=True), y[ok].reset_index(drop=True)
    benign = "BENIGN" if "BENIGN" in set(y) else "BenignTraffic"
    return X, y, benign


def clean(X, y):
    """Drop exact duplicates, then drop feature vectors that carry more than one label."""
    n0 = len(X)
    hf = pd.util.hash_pandas_object(X, index=False).values
    key = pd.DataFrame({"h": hf, "y": y.values})
    keep = ~key.duplicated().values
    X1, y1, h1 = X[keep].reset_index(drop=True), y[keep].reset_index(drop=True), hf[keep]
    n1 = len(X1)
    nlab = pd.Series(y1.values).groupby(h1).transform("nunique").values
    ok = nlab == 1
    X2, y2 = X1[ok].reset_index(drop=True), y1[ok].reset_index(drop=True)
    return X2, y2, {"raw": n0, "after_dedup": n1, "after_conflict": len(X2)}


def cap_classes(X, y, cap, benign, seed=0):
    """Keep all benign rows; randomly cap every attack class at `cap` rows."""
    if cap is None:
        return X, y
    rng = np.random.RandomState(seed)
    idx = []
    for c, ix in y.groupby(y).groups.items():
        ix = np.asarray(ix)
        if c != benign and len(ix) > cap:
            ix = rng.choice(ix, cap, replace=False)
        idx.append(ix)
    idx = np.sort(np.concatenate(idx))
    return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)


def split(X, y, seed=0, val=0.15, test=0.15):
    idx = np.arange(len(X))
    tr, tmp = train_test_split(idx, test_size=val + test, stratify=y, random_state=seed)
    va, te = train_test_split(tmp, test_size=test / (val + test), stratify=y.iloc[tmp],
                              random_state=seed)
    return tr, va, te


def load(name, path, cap=None, seed=0):
    if name == "unsw":
        X, y, benign = load_unsw(path)
    elif name == "cic":
        X, y, benign = load_ciciot(path)
    else:
        raise ValueError(name)
    X, y, stats = clean(X, y)
    X, y = cap_classes(X, y, cap, benign, seed)
    stats["used"] = len(X)
    return X, y, benign, stats
