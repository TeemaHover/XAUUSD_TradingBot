"""
Gaussian Hidden Markov Model for market regime detection. Pure numpy.

Fits a K-state HMM (diagonal covariance, Baum-Welch/EM) on per-bar
observations [log return, log volatility], then produces a CAUSAL regime
probability for every bar via forward filtering — the probability at bar t
uses only bars <= t, so it is safe for backtesting and live use.

The HMM is fit ONLY on the first (1 - val_frac) of the data (the same train
region as ai_train.py), then decoded over the whole series. Regime stats on
the holdout are therefore out-of-sample.

Output CSV (one row per bar):  time, state, p0, p1, ..., p{K-1}

Usage:
    python scripts/hmm_regime.py data/gold_5m.csv
    python scripts/hmm_regime.py data/gold_1h.csv --states 4 --out data/hmm_1h.csv

Options:
    --states     Number of hidden states (default 3)
    --val-frac   Fraction of data excluded from fitting (default 0.2)
    --iters      Max EM iterations (default 100)
    --out        Output CSV (default data/hmm_regimes.csv)
    --seed       RNG seed (default 42)

Next step:
    node dist/src/backtest/runBacktest.js data/gold_5m.csv        (rules)
    python scripts/regime_report.py backtest-results.json data/hmm_regimes.csv
"""

import sys, os, csv, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ai_train import load_csv


# ---------------------------------------------------------- observations
def build_observations(candles):
    """Per-bar features: [log return, log true-range volatility]."""
    closes = np.array([c["close"] for c in candles], dtype=np.float64)
    highs  = np.array([c["high"]  for c in candles], dtype=np.float64)
    lows   = np.array([c["low"]   for c in candles], dtype=np.float64)
    times  = np.array([c["time"]  for c in candles], dtype=np.int64)

    ret = np.diff(np.log(closes))
    tr = np.maximum(highs[1:] - lows[1:],
         np.maximum(np.abs(highs[1:] - closes[:-1]),
                    np.abs(lows[1:]  - closes[:-1])))
    vol = np.log(tr / closes[1:] + 1e-10)

    X = np.column_stack([ret, vol])          # (T, 2)
    return X, times[1:]                       # obs t belongs to candle t+1


# ------------------------------------------------------------ Gaussian HMM
class GaussianHMM:
    """Diagonal-covariance Gaussian HMM trained with scaled Baum-Welch."""

    def __init__(self, n_states: int, seed: int = 42):
        self.K = n_states
        self.rng = np.random.default_rng(seed)
        self.pi = None      # (K,)   initial state probs
        self.A = None       # (K,K)  transition matrix
        self.means = None   # (K,D)
        self.vars = None    # (K,D)

    # log emission probabilities, (T, K)
    def _log_b(self, X):
        d = X[:, None, :] - self.means[None, :, :]          # (T,K,D)
        return -0.5 * (np.sum(d * d / self.vars[None], axis=2)
                       + np.sum(np.log(2 * np.pi * self.vars), axis=1))

    def _init_params(self, X):
        T, D = X.shape
        # Initialise means by volatility quantiles so states start distinct
        order = np.argsort(X[:, 1])
        chunks = np.array_split(order, self.K)
        self.means = np.array([X[c].mean(axis=0) for c in chunks])
        self.vars = np.array([X[c].var(axis=0) + 1e-6 for c in chunks])
        self.A = np.full((self.K, self.K), 0.05 / max(1, self.K - 1))
        np.fill_diagonal(self.A, 0.95)                      # sticky regimes
        self.pi = np.full(self.K, 1.0 / self.K)

    def fit(self, X, iters: int = 100, tol: float = 1e-5, verbose: bool = True):
        self._init_params(X)
        T = X.shape[0]
        prev_ll = -np.inf
        for it in range(1, iters + 1):
            log_b = self._log_b(X)
            b = np.exp(log_b - log_b.max(axis=1, keepdims=True))  # scaled

            # forward (scaled)
            alpha = np.empty((T, self.K)); scale = np.empty(T)
            a = self.pi * b[0]
            scale[0] = a.sum() + 1e-300; alpha[0] = a / scale[0]
            for t in range(1, T):
                a = (alpha[t-1] @ self.A) * b[t]
                scale[t] = a.sum() + 1e-300
                alpha[t] = a / scale[t]

            # backward (scaled)
            beta = np.empty((T, self.K))
            beta[-1] = 1.0
            for t in range(T - 2, -1, -1):
                beta[t] = (self.A @ (b[t+1] * beta[t+1])) / scale[t+1]

            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

            # transition expectations
            xi_num = np.zeros((self.K, self.K))
            for t in range(T - 1):
                m = (alpha[t][:, None] * self.A) * (b[t+1] * beta[t+1])[None, :]
                xi_num += m / (m.sum() + 1e-300)

            # M-step
            self.pi = gamma[0]
            self.A = xi_num / (xi_num.sum(axis=1, keepdims=True) + 1e-300)
            w = gamma.sum(axis=0) + 1e-300
            self.means = (gamma.T @ X) / w[:, None]
            d2 = (X[:, None, :] - self.means[None]) ** 2
            self.vars = np.einsum("tk,tkd->kd", gamma, d2) / w[:, None] + 1e-8

            ll = float(np.sum(np.log(scale)))  # scaled log-likelihood
            if verbose and (it == 1 or it % 10 == 0):
                print(f"  EM iter {it:>3}  loglik/T = {ll / T:.4f}")
            if abs(ll - prev_ll) < tol * T:
                if verbose:
                    print(f"  Converged at iter {it}")
                break
            prev_ll = ll
        return self

    def filter_probs(self, X):
        """CAUSAL state probabilities: P(state_t | obs_1..t). (T, K)"""
        log_b = self._log_b(X)
        b = np.exp(log_b - log_b.max(axis=1, keepdims=True))
        T = X.shape[0]
        out = np.empty((T, self.K))
        a = self.pi * b[0]; a /= a.sum() + 1e-300
        out[0] = a
        for t in range(1, T):
            a = (a @ self.A) * b[t]
            a /= a.sum() + 1e-300
            out[t] = a
        return out

    def save(self, path):
        np.savez(path, pi=self.pi, A=self.A, means=self.means,
                 vars=self.vars, K=np.array(self.K))
        print(f"HMM saved -> {path}")


def describe_states(model: GaussianHMM, probs: np.ndarray, mu, sd):
    """Human-readable summary of each state (in raw units, not z-scores)."""
    K = model.K
    occ = probs.argmax(axis=1)
    raw_means = model.means * sd + mu   # unstandardize
    print("\nState summary (on decoded series):")
    print(f"  {'state':>5}  {'occupancy':>9}  {'mean ret/bar':>12}  "
          f"{'vol (log TR)':>12}  {'avg duration':>12}")
    for k in range(K):
        stay = model.A[k, k]
        dur = 1.0 / max(1e-9, 1.0 - stay)
        share = float(np.mean(occ == k))
        print(f"  {k:>5}  {share:>8.1%}  {raw_means[k,0]:>12.2e}  "
              f"{raw_means[k,1]:>12.2f}  {dur:>10.1f} bars")
    vol_rank = np.argsort(model.means[:, 1])
    names = ["calmest"] + ["mid"] * max(0, K - 2) + ["most volatile"]
    print("  Volatility ranking (low->high): "
          + " < ".join(f"state {k} ({n})" for k, n in zip(vol_rank, names)))


def main():
    ap = argparse.ArgumentParser(description="HMM regime detection")
    ap.add_argument("csv", help="Candle CSV (e.g. data/gold_5m.csv)")
    ap.add_argument("--states", type=int, default=3)
    ap.add_argument("--val-frac", type=float, default=0.2, dest="val_frac")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--out", default="data/hmm_regimes.csv")
    ap.add_argument("--model-out", default="models/hmm_model.npz")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Loading {args.csv} ...")
    candles = load_csv(args.csv)
    X, times = build_observations(candles)
    print(f"  {len(X):,} observations")

    # Standardize using TRAIN stats only (no peeking at holdout)
    n_train = int(len(X) * (1.0 - args.val_frac))
    mu, sd = X[:n_train].mean(axis=0), X[:n_train].std(axis=0) + 1e-12
    Xs = (X - mu) / sd

    print(f"Fitting {args.states}-state HMM on first {n_train:,} bars "
          f"(holdout: last {len(X) - n_train:,}) ...")
    model = GaussianHMM(args.states, seed=args.seed).fit(Xs[:n_train], iters=args.iters)

    print("Decoding full series (causal forward filter) ...")
    probs = model.filter_probs(Xs)
    states = probs.argmax(axis=1)

    describe_states(model, probs, mu, sd)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "state"] + [f"p{k}" for k in range(args.states)])
        for t in range(len(times)):
            w.writerow([int(times[t]), int(states[t])]
                       + [f"{p:.4f}" for p in probs[t]])
    print(f"\nWrote {len(times):,} rows -> {args.out}")

    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
    model.save(args.model_out)

    print("\nNext: run a backtest, then:")
    print(f"  python scripts/regime_report.py backtest-results.json {args.out}")


if __name__ == "__main__":
    main()
