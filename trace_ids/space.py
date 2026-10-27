"""Preprocessing, verified flow-feature constraints and the differentiable feasible space.

Every relation listed for a dataset is a *candidate*: it is enforced only if it holds on at
least `VERIFY_MIN` of the training flows (verify-then-enforce). This keeps the framework
usable on other releases of the same benchmark (e.g. the 46-feature CICIoT2023 CSVs).
"""
import numpy as np
import pandas as pd
import torch

VERIFY_MIN = 0.995

SCHEMAS = {
    "unsw": {
        "categorical": ["proto", "service", "state"],
        "immutable": ["is_sm_ips_ports"],
    },
    "cic": {
        "categorical": [],
        "immutable": ["Protocol Type", "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC",
                      "TCP", "UDP", "DHCP", "ARP", "ICMP", "IGMP", "IPv", "LLC", "Number"],
    },
}


# ----------------------------------------------------------------------------------------
# Relations (each has a numpy check and a functional torch implementation on column lists)
# ----------------------------------------------------------------------------------------
class Relation:
    kind = "derive"  # or "clamp"
    child = None

    def cols(self):
        raise NotImplementedError


class Ratio(Relation):
    """child = num / den if den >= 1 else 0 (e.g. smean = sbytes/spkts, AVG = Tot sum/Number)."""

    def __init__(self, child, num, den, tol_abs):
        self.child, self.num, self.den, self.tol = child, num, den, tol_abs
        self.name = f"{child} = {num} / {den}"

    def cols(self):
        return [self.child, self.num, self.den]

    def check(self, D):
        den = D[self.den]
        val = np.where(den >= 0.5, D[self.num] / np.maximum(den, 1e-12), 0.0)
        return np.abs(D[self.child] - val) <= self.tol + 1e-9 * np.abs(val)

    def apply(self, C, ix):
        den = C[ix[self.den]]
        C[ix[self.child]] = torch.where(den >= 0.5, C[ix[self.num]] / den.clamp(min=1.0),
                                        torch.zeros_like(den))


class Sum(Relation):
    """child = a + b (e.g. tcprtt = synack + ackdat)."""

    def __init__(self, child, a, b):
        self.child, self.a, self.b = child, a, b
        self.name = f"{child} = {a} + {b}"

    def cols(self):
        return [self.child, self.a, self.b]

    def check(self, D):
        val = D[self.a] + D[self.b]
        return np.abs(D[self.child] - val) <= 1e-6 + 1e-6 * np.abs(val)

    def apply(self, C, ix):
        C[ix[self.child]] = C[ix[self.a]] + C[ix[self.b]]


class PacketRate(Relation):
    """child = (a + b - 1) / dur if dur > 0 else 0 (UNSW-NB15 `rate`)."""

    def __init__(self, child, a, b, dur):
        self.child, self.a, self.b, self.dur = child, a, b, dur
        self.name = f"{child} = ({a} + {b} - 1) / {dur}"

    def cols(self):
        return [self.child, self.a, self.b, self.dur]

    def check(self, D):
        d = D[self.dur]
        val = np.where(d > 0, np.maximum(D[self.a] + D[self.b] - 1, 0) / np.maximum(d, 1e-12), 0.0)
        return np.abs(D[self.child] - val) <= 1e-6 + 1e-2 * np.abs(val)

    def apply(self, C, ix):
        d = C[ix[self.dur]]
        num = (C[ix[self.a]] + C[ix[self.b]] - 1.0).clamp(min=0.0)
        C[ix[self.child]] = torch.where(d > 0, num / d.clamp(min=1e-9), torch.zeros_like(d))


class Square(Relation):
    """child = a^2 (CICIoT2023 MERGED release: Variance = Std^2)."""

    def __init__(self, child, a):
        self.child, self.a = child, a
        self.name = f"{child} = {a}^2"

    def cols(self):
        return [self.child, self.a]

    def check(self, D):
        val = D[self.a] ** 2
        return np.abs(D[self.child] - val) <= 1e-2 + 1e-3 * np.abs(val)

    def apply(self, C, ix):
        C[ix[self.child]] = C[ix[self.a]] ** 2


class LessEq(Relation):
    """a <= b (enforced by a := min(a, b))."""
    kind = "clamp"

    def __init__(self, a, b):
        self.a, self.b = a, b
        self.name = f"{a} <= {b}"

    def cols(self):
        return [self.a, self.b]

    def check(self, D):
        return D[self.a] <= D[self.b] + 1e-9

    def apply(self, C, ix, rounding=False):
        C[ix[self.a]] = torch.minimum(C[ix[self.a]], C[ix[self.b]])


class ScaledRange(Relation):
    """b*lo <= a <= b*hi, lo/hi constants fitted on training data (mean packet size range)."""
    kind = "clamp"

    def __init__(self, a, b):
        self.a, self.b = a, b
        self.lo, self.hi = None, None
        self.name = f"{b}*m_lo <= {a} <= {b}*m_hi"

    def cols(self):
        return [self.a, self.b]

    def fit(self, D):
        m = D[self.b] > 0
        r = D[self.a][m] / D[self.b][m]
        self.lo, self.hi = float(np.min(r)), float(np.max(r))
        self.name = f"{self.lo:g}*{self.b} <= {self.a} <= {self.hi:g}*{self.b}"

    def check(self, D):
        b = D[self.b]
        return (D[self.a] >= b * self.lo - 1e-6) & (D[self.a] <= b * self.hi + 1e-6)

    def apply(self, C, ix, rounding=False):
        b = C[ix[self.b]]
        lo, hi = b * self.lo, b * self.hi
        if rounding:  # integer-safe bounds
            lo, hi = torch.ceil(lo - 1e-9), torch.floor(hi + 1e-9)
        C[ix[self.a]] = torch.minimum(torch.maximum(C[ix[self.a]], lo), hi)


class ColRange(Relation):
    """n*lo_col <= a <= n*hi_col (CICIoT2023: Number*Min <= Tot sum <= Number*Max)."""
    kind = "clamp"

    def __init__(self, a, n, lo_col, hi_col):
        self.a, self.n, self.lo_col, self.hi_col = a, n, lo_col, hi_col
        self.name = f"{n}*{lo_col} <= {a} <= {n}*{hi_col}"

    def cols(self):
        return [self.a, self.n, self.lo_col, self.hi_col]

    def check(self, D):
        n = D[self.n]
        return (D[self.a] >= n * D[self.lo_col] - 1e-6) & (D[self.a] <= n * D[self.hi_col] + 1e-6)

    def apply(self, C, ix, rounding=False):
        n = C[ix[self.n]]
        C[ix[self.a]] = torch.minimum(torch.maximum(C[ix[self.a]], n * C[ix[self.lo_col]]),
                                      n * C[ix[self.hi_col]])


def candidate_relations(name):
    if name == "unsw":
        return [
            LessEq("sloss", "spkts"), LessEq("dloss", "dpkts"),
            ScaledRange("sbytes", "spkts"), ScaledRange("dbytes", "dpkts"),
            Ratio("smean", "sbytes", "spkts", tol_abs=1.0), Ratio("dmean", "dbytes", "dpkts", tol_abs=1.0),
            PacketRate("rate", "spkts", "dpkts", "dur"), Sum("tcprtt", "synack", "ackdat"),
        ]
    if name == "cic":
        return [
            LessEq("Min", "Max"),
            LessEq("ack_count", "Number"), LessEq("syn_count", "Number"),
            LessEq("fin_count", "Number"), LessEq("rst_count", "Number"), LessEq("urg_count", "Number"),
            ColRange("Tot sum", "Number", "Min", "Max"),
            Ratio("AVG", "Tot sum", "Number", tol_abs=0.006), Square("Variance", "Std"),
        ]
    raise ValueError(name)


# ----------------------------------------------------------------------------------------
# Preprocessing: raw -> (log1p for skewed non-negative features) -> min-max -> u in [0,1]
# ----------------------------------------------------------------------------------------
class Prep:
    def __init__(self, name, Xtr, min_cat_count=20):
        sch = SCHEMAS[name]
        self.name = name
        self.cat_cols = [c for c in sch["categorical"] if c in Xtr.columns]
        self.num_cols = [c for c in Xtr.columns if c not in self.cat_cols]
        R = Xtr[self.num_cols].to_numpy(np.float64)
        self.raw_lo, self.raw_hi = R.min(0), R.max(0)
        skew = pd.DataFrame(R).skew().fillna(0).values
        self.log = (self.raw_lo >= 0) & (skew > 2)
        T = self._t(R)
        self.t_lo, self.t_hi = T.min(0), T.max(0)
        self.span = np.where(self.t_hi - self.t_lo > 0, self.t_hi - self.t_lo, 1.0)
        U = self.u_np(R)
        med = np.median(U, 0)
        mad = np.median(np.abs(U - med), 0)
        self.sigma = np.where(mad > 1e-3, mad, np.maximum(U.std(0), 1e-3))
        self.integer = np.all(np.mod(R, 1) == 0, axis=0)
        self.cats = {}
        for c in self.cat_cols:
            vc = Xtr[c].value_counts()
            self.cats[c] = vc[vc >= min_cat_count].index.tolist()
        self.oh_names = [f"{c}={v}" for c in self.cat_cols for v in self.cats[c] + ["other"]]
        self.ix = {c: i for i, c in enumerate(self.num_cols)}

    def _t(self, R):
        T = R.copy()
        T[:, self.log] = np.log1p(np.maximum(R[:, self.log], 0))
        return T

    def u_np(self, R):
        return (self._t(R) - self.t_lo) / self.span

    def raw_np(self, U):
        T = U * self.span + self.t_lo
        R = T.copy()
        R[:, self.log] = np.expm1(T[:, self.log])
        return R

    def u_t(self, R):
        lg = torch.as_tensor(self.log)
        T = torch.where(lg, torch.log1p(R.clamp(min=0)), R)
        return (T - torch.as_tensor(self.t_lo)) / torch.as_tensor(self.span)

    def raw_t(self, U):
        T = U * torch.as_tensor(self.span) + torch.as_tensor(self.t_lo)
        return torch.where(torch.as_tensor(self.log), torch.expm1(T), T)

    def onehot(self, X):
        blocks = []
        for c in self.cat_cols:
            vals = X[c].astype(str).values
            cats = self.cats[c]
            M = np.zeros((len(X), len(cats) + 1), np.float32)
            pos = {v: i for i, v in enumerate(cats)}
            j = np.array([pos.get(v, len(cats)) for v in vals])
            M[np.arange(len(X)), j] = 1.0
            blocks.append(M)
        return np.concatenate(blocks, 1) if blocks else np.zeros((len(X), 0), np.float32)

    def transform(self, X):
        R = X[self.num_cols].to_numpy(np.float64)
        U = np.clip(self.u_np(R), 0.0, 1.0).astype(np.float32)
        return U, self.onehot(X), R

    @property
    def d_num(self):
        return len(self.num_cols)


# ----------------------------------------------------------------------------------------
# Feasible space: immutability, integrality, bounds, verified relations
# ----------------------------------------------------------------------------------------
class FeasibleSpace:
    def __init__(self, prep, Rtr):
        self.prep = prep
        ix = prep.ix
        D = {c: Rtr[:, i] for c, i in ix.items()}
        self.relations, self.report, self.candidates = [], [], []
        for r in candidate_relations(prep.name):
            if not all(c in ix for c in r.cols()):
                continue
            if isinstance(r, ScaledRange):
                r.fit(D)
            self.candidates.append(r)
            sat = float(np.mean(r.check(D)))
            self.report.append({"relation": r.name, "kind": r.kind, "train_satisfaction": sat,
                                "enforced": sat >= VERIFY_MIN})
            if sat >= VERIFY_MIN:
                self.relations.append(r)
        d = prep.d_num
        imm = set(SCHEMAS[prep.name]["immutable"])
        self.imm = np.array([c in imm for c in prep.num_cols])
        self.der = np.zeros(d, bool)
        for r in self.relations:
            if r.kind == "derive":
                self.der[ix[r.child]] = True
        self.free_rel = ~self.imm & ~self.der
        self.free_norel = ~self.imm
        self.int = prep.integer.copy()
        self.lo = torch.as_tensor(prep.raw_lo)
        self.hi = torch.as_tensor(prep.raw_hi)

    def repair(self, R, R0, use_rel=True, round_int=False):
        """Map an arbitrary raw candidate onto the feasible set (differentiable a.e.)."""
        imm = torch.as_tensor(self.imm)
        der = torch.as_tensor(self.der) if use_rel else torch.zeros_like(imm)
        intm = torch.as_tensor(self.int)
        R = torch.where(imm, R0, R)
        if round_int:
            R = torch.where(intm & ~imm, torch.round(R), R)
        R = torch.where(der, R, torch.maximum(torch.minimum(R, self.hi), self.lo))
        if use_rel:
            C = list(R.unbind(1))
            for r in self.relations:
                if r.kind == "clamp":
                    r.apply(C, self.prep.ix, rounding=round_int)
            if round_int:
                C = [torch.round(c) if (self.int[i] and not self.imm[i]) else c for i, c in enumerate(C)]
            for r in self.relations:
                if r.kind == "derive":
                    r.apply(C, self.prep.ix)
            if round_int:
                C = [torch.round(c) if (self.int[i] and self.der[i]) else c for i, c in enumerate(C)]
            R = torch.stack(C, 1)
        return R

    def check(self, Rcf, R0):
        """Constraint satisfaction of counterfactuals (numpy, raw space)."""
        out = {}
        imm_ok = np.all(np.abs(Rcf[:, self.imm] - R0[:, self.imm]) <= 1e-6 * (1 + np.abs(R0[:, self.imm])), 1)
        ints = self.int & ~self.imm
        int_ok = np.all(np.abs(Rcf[:, ints] - np.round(Rcf[:, ints])) <= 1e-3, 1)
        nd = ~self.der
        tol = 1e-6 * (1 + np.abs(self.prep.raw_hi[nd]))
        bnd_ok = np.all((Rcf[:, nd] >= self.prep.raw_lo[nd] - tol) & (Rcf[:, nd] <= self.prep.raw_hi[nd] + tol), 1)
        D = {c: Rcf[:, i] for c, i in self.prep.ix.items()}
        rel_ok = np.ones(len(Rcf), bool)
        der_ok = np.ones(len(Rcf), bool)
        for r in self.relations:
            ok = r.check(D)
            if r.kind == "clamp":
                rel_ok &= ok
            else:
                der_ok &= ok
        out["immutable"], out["integer"], out["bounds"] = imm_ok, int_ok, bnd_ok
        out["relational"], out["derived"] = rel_ok, der_ok
        out["all"] = imm_ok & int_ok & bnd_ok & rel_ok & der_ok
        return out
