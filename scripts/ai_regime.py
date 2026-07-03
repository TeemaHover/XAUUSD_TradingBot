"""
HMM Regime Brain for the GOLD bot.
Pure-numpy — no hmmlearn / scipy needed (works on Windows with numpy alone).

Ported and adapted from the regime_trader project. Adds three things the
TypeScript bot did not have before:

  1. A Gaussian-HMM market-regime detector (bull / bear / neutral / crash /
     euphoria ...), with the number of regimes chosen automatically by BIC.
  2. A regime + confidence -> position-size multiplier (regime-scaled sizing).
  3. CSV annotation so the backtester and analytics can use the regime.

No look-ahead: live/annotate inference uses the FORWARD algorithm only, so the
regime at time t is computed from data up to and including the last CLOSED
higher-timeframe bar at time t.

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
  # 1) Train the regime model on the 1h gold series (recommended TF):
  python scripts/ai_regime.py train --csv data/gold_1h.csv \
      --out models/regime_hmm.npz --bar-seconds 3600

  # 2) Annotate an AI-predictions CSV with regime / confidence / sizeMult:
  python scripts/ai_regime.py annotate --preds data/preds_baseline.csv \
      --model models/regime_hmm.npz --csv data/gold_1h.csv \
      --out data/preds_baseline_regime.csv --bar-seconds 3600

The annotated CSV keeps the original columns and appends:
      regime, regimeConf, sizeMult
"""

from __future__ import annotations

import argparse
import csv
import sys
import numpy as np

# ----------------------------------------------------------------- config

MIN_REGIMES      = 3
MAX_REGIMES      = 7
STABILITY_BARS   = 3     # a new regime must persist this many bars before we act
FLICKER_WINDOW   = 20    # look-back window for flicker detection
FLICKER_MAX      = 4     # more changes than this in the window => "uncertain"
FEATURE_COLS     = ["log_ret", "vol_10", "vol_ratio", "atr_norm", "volume_z"]

REGIME_LABELS = {
    3: ["bear", "neutral", "bull"],
    4: ["crash", "bear", "bull", "euphoria"],
    5: ["crash", "bear", "neutral", "bull", "euphoria"],
    6: ["crash", "bear", "neutral", "bull", "strong_bull", "euphoria"],
    7: ["crash", "bear", "weak_bear", "neutral", "weak_bull", "bull", "euphoria"],
}

# Regime -> base size multiplier. Direction-agnostic: it scales conviction by
# how tradeable / risky the regime is, NOT by bull-vs-bear (the AI picks the
# side). Trending regimes get full size; chop and blow-off tops get less.
# Tune these freely — this table is the single source of truth for sizing.
REGIME_SIZE = {
    "crash":       0.60,   # violent, wide stops
    "bear":        1.00,
    "weak_bear":   0.85,
    "neutral":     0.70,   # chop / mean-reversion
    "weak_bull":   0.85,
    "bull":        1.00,
    "strong_bull": 1.10,
    "euphoria":    0.60,   # blow-off tops are dangerous
    "unknown":     0.50,
}
CONFIDENCE_FLOOR = 0.55    # below this filtered confidence -> shrink size
SIZE_MIN, SIZE_MAX = 0.10, 1.25


def size_multiplier(label: str, confidence: float, uncertain: bool) -> float:
    """Regime + confidence -> position-size multiplier (clamped)."""
    base = REGIME_SIZE.get(label, 0.5)
    # confidence factor: full weight at high conf, down to 0.5 at/under floor
    if confidence >= CONFIDENCE_FLOOR:
        conf_factor = 1.0
    else:
        conf_factor = 0.5 + 0.5 * (confidence / max(CONFIDENCE_FLOOR, 1e-9))
    mult = base * conf_factor
    if uncertain:               # flickering regime -> trade smaller
        mult *= 0.5
    return float(np.clip(mult, SIZE_MIN, SIZE_MAX))


# ----------------------------------------------------------- data / features

def load_csv(path: str):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "time":   int(float(r["time"])),
                "open":   float(r["open"]),
                "high":   float(r["high"]),
                "low":    float(r["low"]),
                "close":  float(r["close"]),
                "volume": float(r.get("volume", 1) or 1),
            })
    rows.sort(key=lambda c: c["time"])
    return rows


def compute_features(candles):
    """Return (times, X) where X[i] are the features usable AT bar i (data up
    to and including bar i only — rolling stats never peek forward). Leading
    rows with NaNs are dropped."""
    t = np.array([c["time"] for c in candles], dtype=np.int64)
    o = np.array([c["open"] for c in candles], dtype=float)
    h = np.array([c["high"] for c in candles], dtype=float)
    lo = np.array([c["low"] for c in candles], dtype=float)
    c = np.array([c["close"] for c in candles], dtype=float)
    v = np.array([c_["volume"] for c_ in candles], dtype=float)

    log_ret = np.zeros_like(c)
    log_ret[1:] = np.log(c[1:] / np.maximum(c[:-1], 1e-12))

    def roll_std(a, w):
        out = np.full_like(a, np.nan)
        for i in range(w - 1, len(a)):
            out[i] = a[i - w + 1:i + 1].std()
        return out

    def roll_mean(a, w):
        out = np.full_like(a, np.nan)
        csum = np.cumsum(np.insert(a, 0, 0.0))
        out[w - 1:] = (csum[w:] - csum[:-w]) / w
        return out

    vol_10 = roll_std(log_ret, 10)
    vol_50 = roll_std(log_ret, 50)
    vol_ratio = vol_10 / np.where(vol_50 == 0, np.nan, vol_50)

    # ATR(14) normalised by price
    prev_c = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum.reduce([h - lo, np.abs(h - prev_c), np.abs(lo - prev_c)])
    atr = roll_mean(tr, 14)
    atr_norm = atr / np.maximum(c, 1e-12)

    vmu = roll_mean(v, 50)
    vsd = roll_std(v, 50)
    volume_z = (v - vmu) / np.where((vsd == 0) | np.isnan(vsd), np.nan, vsd)

    feats = np.column_stack([log_ret, vol_10, vol_ratio, atr_norm, volume_z])
    ok = ~np.isnan(feats).any(axis=1)
    return t[ok], feats[ok]


# --------------------------------------------------- pure-numpy Gaussian HMM

_LOG2PI = np.log(2.0 * np.pi)


def _lse(a, axis):
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


class GaussianHMM:
    """Diagonal-covariance Gaussian HMM trained with Baum-Welch EM."""

    def __init__(self, n_components, n_iter=120, tol=1e-4, random_state=42):
        self.n_components = n_components
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.startprob_ = self.transmat_ = self.means_ = self.covars_ = None

    def _log_emissions(self, X):
        var = np.maximum(self.covars_, 1e-10)
        diff2 = (X[:, None, :] - self.means_[None]) ** 2 / var[None]
        return -0.5 * (np.sum(np.log(var) + _LOG2PI, axis=1)[None] + diff2.sum(axis=2))

    def _fb(self, log_pi, log_A, log_B):
        T, K = log_B.shape
        la = np.empty((T, K)); lb = np.empty((T, K))
        la[0] = log_pi + log_B[0]
        for t in range(1, T):
            la[t] = log_B[t] + _lse(la[t - 1][:, None] + log_A, axis=0)
        lb[-1] = 0.0
        for t in range(T - 2, -1, -1):
            lb[t] = _lse(log_A + (log_B[t + 1] + lb[t + 1])[None, :], axis=1)
        return la, lb, float(_lse(la[-1], axis=0))

    def _init(self, X):
        rng = np.random.default_rng(self.random_state)
        T, d = X.shape; K = self.n_components
        order = np.argsort(X[:, 0])
        means = X[order[np.linspace(0, T - 1, K).astype(int)]].astype(float).copy()
        means += rng.normal(0, 1e-6, means.shape)
        for _ in range(10):
            assign = ((X[:, None, :] - means[None]) ** 2).sum(axis=2).argmin(axis=1)
            for k in range(K):
                pts = X[assign == k]
                if len(pts):
                    means[k] = pts.mean(axis=0)
        self.means_ = means
        self.covars_ = np.full((K, d), np.maximum(X.var(axis=0), 1e-6))
        self.startprob_ = np.full(K, 1.0 / K)
        A = np.full((K, K), 0.05 / max(K - 1, 1)); np.fill_diagonal(A, 0.95)
        self.transmat_ = A

    def fit(self, X):
        X = np.asarray(X, float)
        if X.ndim != 2 or len(X) < self.n_components * 5:
            raise ValueError("not enough rows to fit")
        self._init(X)
        prev = -np.inf
        for _ in range(self.n_iter):
            log_pi = np.log(np.maximum(self.startprob_, 1e-300))
            log_A = np.log(np.maximum(self.transmat_, 1e-300))
            log_B = self._log_emissions(X)
            la, lb, ll = self._fb(log_pi, log_A, log_B)
            log_gamma = la + lb; log_gamma -= _lse(log_gamma, axis=1)[:, None]
            gamma = np.exp(log_gamma)
            log_xi = (la[:-1, :, None] + log_A[None]
                      + (log_B[1:] + lb[1:])[:, None, :]) - ll
            xi = np.exp(log_xi)
            self.startprob_ = np.maximum(gamma[0], 1e-10)
            self.startprob_ /= self.startprob_.sum()
            denom = np.maximum(gamma[:-1].sum(axis=0), 1e-10)
            self.transmat_ = np.maximum(xi.sum(axis=0) / denom[:, None], 1e-10)
            self.transmat_ /= self.transmat_.sum(axis=1, keepdims=True)
            w = np.maximum(gamma.sum(axis=0), 1e-10)
            self.means_ = (gamma.T @ X) / w[:, None]
            diff2 = (X[:, None, :] - self.means_[None]) ** 2
            self.covars_ = np.maximum(np.einsum("tk,tkd->kd", gamma, diff2) / w[:, None], 1e-8)
            if abs(ll - prev) < self.tol * max(abs(prev), 1.0):
                break
            prev = ll
        return self

    def score(self, X):
        X = np.asarray(X, float)
        log_pi = np.log(np.maximum(self.startprob_, 1e-300))
        log_A = np.log(np.maximum(self.transmat_, 1e-300))
        return self._fb(log_pi, log_A, self._log_emissions(X))[2]


# --------------------------------------------------- regime engine (wrapper)

class RegimeHMM:
    def __init__(self):
        self.model = None
        self.n_regimes = 0
        self.labels = []
        self.order = None          # state index -> rank by mean return
        self.mu = None             # feature standardiser
        self.sd = None
        # forward-filter live state
        self._log_alpha = None
        self._hist = []
        self._acted = -1
        self._pending = -1
        self._pending_n = 0

    # -------- training with BIC model selection
    def fit(self, X):
        self.mu = X.mean(axis=0)
        self.sd = np.maximum(X.std(axis=0), 1e-9)
        Xs = (X - self.mu) / self.sd
        best, best_bic, best_n = None, np.inf, 0
        for n in range(MIN_REGIMES, MAX_REGIMES + 1):
            try:
                m = GaussianHMM(n_components=n).fit(Xs)
                ll = m.score(Xs)
                n_params = n * (n - 1) + (n - 1) + 2 * n * Xs.shape[1]
                bic = -2 * ll + n_params * np.log(len(Xs))
                print(f"  HMM n={n}  loglik={ll:,.1f}  bic={bic:,.1f}")
                if bic < best_bic:
                    best, best_bic, best_n = m, bic, n
            except Exception as e:
                print(f"  HMM n={n} failed: {e}")
        if best is None:
            raise RuntimeError("all HMM fits failed")
        self.model, self.n_regimes = best, best_n
        self.order = np.argsort(np.argsort(best.means_[:, 0]))  # by mean log_ret
        self.labels = REGIME_LABELS[best_n]
        print(f"Selected {best_n} regimes (BIC={best_bic:,.1f})")
        return self

    def label_of(self, state_idx):
        if state_idx < 0 or self.order is None:
            return "unknown"
        return self.labels[int(self.order[state_idx])]

    # -------- forward-filter one standardised feature vector
    def _log_emission(self, x):
        var = np.maximum(self.model.covars_, 1e-12)
        diff2 = (x[None, :] - self.model.means_) ** 2 / var
        return -0.5 * (np.sum(np.log(2 * np.pi * var), axis=1) + diff2.sum(axis=1))

    def reset(self):
        self._log_alpha = None
        self._hist = []
        self._acted = self._pending = -1
        self._pending_n = 0

    def update(self, x_raw):
        """Consume one raw feature vector; return (label, confidence, uncertain)."""
        x = (np.asarray(x_raw, float) - self.mu) / self.sd
        log_b = self._log_emission(x)
        if self._log_alpha is None:
            la = np.log(np.maximum(self.model.startprob_, 1e-300)) + log_b
        else:
            trans = np.log(np.maximum(self.model.transmat_, 1e-300))
            la = _lse(self._log_alpha[:, None] + trans, axis=0) + log_b
        la -= _lse(la, axis=0)
        self._log_alpha = la
        probs = np.exp(la)
        raw = int(np.argmax(probs))
        conf = float(probs[raw])

        self._hist.append(raw)
        if len(self._hist) > 500:
            self._hist = self._hist[-500:]

        # stability filter
        if self._acted == -1:
            self._acted = raw
        elif raw != self._acted:
            if raw == self._pending:
                self._pending_n += 1
            else:
                self._pending, self._pending_n = raw, 1
            if self._pending_n >= STABILITY_BARS:
                self._acted = raw
                self._pending, self._pending_n = -1, 0
        else:
            self._pending, self._pending_n = -1, 0

        # flicker / uncertainty
        w = self._hist[-FLICKER_WINDOW:]
        changes = sum(1 for a, b in zip(w, w[1:]) if a != b)
        uncertain = changes > FLICKER_MAX
        return self.label_of(self._acted), conf, uncertain

    # -------- persistence
    def save(self, path):
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(path,
                 n_regimes=np.array(self.n_regimes),
                 labels=np.array(self.labels),
                 order=self.order,
                 mu=self.mu, sd=self.sd,
                 startprob=self.model.startprob_,
                 transmat=self.model.transmat_,
                 means=self.model.means_,
                 covars=self.model.covars_)
        print(f"Regime model saved -> {path}")

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=True)
        self = cls()
        self.n_regimes = int(d["n_regimes"])
        self.labels = [str(s) for s in d["labels"]]
        self.order = d["order"]
        self.mu = d["mu"]; self.sd = d["sd"]
        m = GaussianHMM(self.n_regimes)
        m.startprob_ = d["startprob"]; m.transmat_ = d["transmat"]
        m.means_ = d["means"]; m.covars_ = d["covars"]
        self.model = m
        return self


# ----------------------------------------------------------------- CLI

def _filter_series(engine: RegimeHMM, times, X):
    """Forward-filter the whole feature series once. Returns arrays aligned to
    `times`: label(str), conf(float), uncertain(bool)."""
    engine.reset()
    labels, confs, unc = [], [], []
    for x in X:
        lb, cf, u = engine.update(x)
        labels.append(lb); confs.append(cf); unc.append(u)
    return times, np.array(labels), np.array(confs), np.array(unc, dtype=bool)


def cmd_train(args):
    candles = load_csv(args.csv)
    times, X = compute_features(candles)
    if len(X) < 500:
        sys.exit("Need at least 500 feature rows to train a regime model.")
    if args.max_rows and len(X) > args.max_rows:
        X = X[-args.max_rows:]      # most recent window (pure-numpy EM is O(T))
        print(f"Capped to most recent {args.max_rows:,} rows for speed.")
    print(f"Training on {len(X):,} {args.csv} feature rows "
          f"({', '.join(FEATURE_COLS)})")
    engine = RegimeHMM().fit(X)
    engine.save(args.out)


def cmd_annotate(args):
    engine = RegimeHMM.load(args.model)
    tf_candles = load_csv(args.csv)
    ftimes, X = compute_features(tf_candles)
    _, labels, confs, unc = _filter_series(engine, ftimes, X)

    # The regime that is KNOWN at time t is the last higher-TF bar fully closed
    # at or before t. ftimes are bar OPEN times, so a bar is closed once
    # t >= open + bar_seconds*1000.
    bar_ms = args.bar_seconds * 1000
    closed_at = ftimes + bar_ms  # time each regime reading becomes usable

    # read predictions
    with open(args.preds, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    tcol = header.index("time")

    import bisect
    out_rows = []
    n_missing = 0
    for r in rows:
        t = int(float(r[tcol]))
        j = bisect.bisect_right(closed_at, t) - 1
        if j < 0:
            lb, cf, u = "unknown", 0.0, True
            n_missing += 1
        else:
            lb, cf, u = labels[j], float(confs[j]), bool(unc[j])
        sm = size_multiplier(lb, cf, u)
        out_rows.append(r + [lb, f"{cf:.4f}", f"{sm:.4f}"])

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header + ["regime", "regimeConf", "sizeMult"])
        w.writerows(out_rows)

    # quick summary
    from collections import Counter
    cnt = Counter(r[-3] for r in out_rows)
    print(f"Annotated {len(out_rows):,} predictions -> {args.out}")
    if n_missing:
        print(f"  ({n_missing:,} early rows had no regime yet -> unknown)")
    print("  regime distribution:")
    for lb, n in cnt.most_common():
        print(f"    {lb:<12} {n:>7,}  ({100*n/len(out_rows):4.1f}%)")


def main():
    ap = argparse.ArgumentParser(description="HMM regime brain for the GOLD bot")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="fit the regime HMM on a higher-TF CSV")
    t.add_argument("--csv", default="data/gold_1h.csv")
    t.add_argument("--out", default="models/regime_hmm.npz")
    t.add_argument("--bar-seconds", type=int, default=3600)
    t.add_argument("--max-rows", type=int, default=6000,
                   help="cap training to the most recent N bars (0 = all)")
    t.set_defaults(func=cmd_train)

    a = sub.add_parser("annotate", help="add regime/sizeMult columns to a preds CSV")
    a.add_argument("--preds", required=True)
    a.add_argument("--model", default="models/regime_hmm.npz")
    a.add_argument("--csv", default="data/gold_1h.csv")
    a.add_argument("--out", required=True)
    a.add_argument("--bar-seconds", type=int, default=3600)
    a.set_defaults(func=cmd_annotate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
