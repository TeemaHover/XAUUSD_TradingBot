"""
Live HMM regime classification for the dashboard.

Reads JSON from stdin:
    { "modelPath": "models/hmm_model.npz",
      "candles": [ {time,open,high,low,close,volume}, ... ] }

Writes JSON to stdout:
    { "state": 1, "label": "volatile", "confidence": 0.87, "probs": [..] }

Uses the same observations and causal forward filter as scripts/hmm_regime.py,
so the live label matches what a backtest at this bar would have seen.
Labels are assigned by each state's volatility rank (calm < normal < volatile),
which is stable regardless of arbitrary state indexing.

Fails soft: any problem returns {"label": "unknown"} with a reason.
"""

import json
import os
import sys


def out(obj):
    print(json.dumps(obj))
    sys.exit(0)


def unknown(reason: str):
    out({"state": -1, "label": "unknown", "confidence": 0.0, "probs": [], "reason": reason})


try:
    import numpy as np
except ImportError as e:
    unknown(f"numpy not available: {e}")

try:
    payload = json.loads(sys.stdin.read() or "{}")
    model_path = payload.get("modelPath", "models/hmm_model.npz")
    candles = payload.get("candles", [])

    if not os.path.exists(model_path):
        unknown(f"model not found at {model_path} -- run hmm_regime.py first")
    if len(candles) < 50:
        unknown(f"need at least 50 candles, got {len(candles)}")

    data = np.load(model_path)
    pi, A = data["pi"], data["A"]
    means, vars_ = data["means"], data["vars"]
    K = int(data["K"])

    closes = np.array([c["close"] for c in candles], dtype=np.float64)
    highs = np.array([c["high"] for c in candles], dtype=np.float64)
    lows = np.array([c["low"] for c in candles], dtype=np.float64)

    ret = np.diff(np.log(closes))
    tr = np.maximum(highs[1:] - lows[1:],
                    np.maximum(np.abs(highs[1:] - closes[:-1]),
                               np.abs(lows[1:] - closes[:-1])))
    vol = np.log(tr / closes[1:] + 1e-10)
    X = np.column_stack([ret, vol])

    # Prefer the train-time standardization saved with the model; older models
    # lack it, so fall back to this window's stats (approximate but usable).
    if "mu" in data.files and "sd" in data.files:
        mu, sd = data["mu"], data["sd"]
    else:
        mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-12
    Xs = (X - mu) / sd

    # Causal forward filter: P(state_t | obs_1..t)
    d = Xs[:, None, :] - means[None, :, :]
    log_b = -0.5 * (np.sum(d * d / vars_[None], axis=2)
                    + np.sum(np.log(2 * np.pi * vars_), axis=1))
    b = np.exp(log_b - log_b.max(axis=1, keepdims=True))
    a = pi * b[0]
    a /= a.sum() + 1e-300
    for t in range(1, len(Xs)):
        a = (a @ A) * b[t]
        a /= a.sum() + 1e-300

    state = int(np.argmax(a))
    vol_rank = list(np.argsort(means[:, 1]))  # states ordered low -> high volatility
    if K == 2:
        names = ["calm", "volatile"]
    elif K == 3:
        names = ["calm", "normal", "volatile"]
    else:
        names = ["calm"] + [f"mid-{i}" for i in range(1, K - 1)] + ["volatile"]
    label = names[vol_rank.index(state)]

    out({
        "state": state,
        "label": label,
        "confidence": float(a[state]),
        "probs": [float(p) for p in a]
    })
except SystemExit:
    raise
except Exception as e:  # noqa: BLE001 — dashboard display must never crash the bot
    unknown(f"prediction error: {e}")
