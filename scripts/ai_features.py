"""
Sequence feature extraction for the XAUUSD multi-timeframe CNN model.
Pure numpy — no external dependencies.

FEATURE REGISTRY: every feature is a named function. Pick any combination at
training time with --features; the trained model remembers its feature list,
and all prediction paths extract exactly those features automatically.

Groups:
  base (7)      raw candle geometry — always recommended
      open_ret, body_ret, upper_wick, lower_wick, range_ret, vol_ratio, pos_20
  patterns (5)  per-bar candle/volume patterns
      vol_pressure  signed volume: sign(body) * vol_ratio (-1..1)
      engulfing     bullish engulfing +1, bearish -1, else 0
      pinbar        hammer +1, shooting star -1, else 0
      dbl_retest    retesting prior swing low +1 / swing high -1, else 0
      compression   10-bar range / (6*ATR), 0-1 (low = squeeze)
  context (5)   market-structure context, computed at the decision bar over up
                to CONTEXT_LOOKBACK bars of history, broadcast to all rows
      sr_dist       S/R proximity: +1 sitting on support, -1 at resistance,
                    0 = nothing near (within 3 ATR)
      ob_dist       unmitigated order-block proximity, same encoding as sr_dist
      slope_high    regression slope of recent swing HIGHS (ATR/bar, clipped)
      slope_low     regression slope of recent swing LOWS  (ATR/bar, clipped)
                    falling wedge: both < 0 converging; descending triangle:
                    slope_low ~ 0, slope_high < 0; rectangle: both ~ 0
      convergence   channel width now / width 40 bars ago (0-1 = contracting)

Feature spec strings (for --features): group names and/or feature names,
comma-separated. Examples: "base,patterns" (default), "all",
"base,patterns,sr_dist,slope_high,slope_low".

Main entry points:
  resolve_features(spec)                          -> list of feature names
  extract_sequence(candles, features=None)        -> (SEQ_LEN, n_features)
  extract_sequence_matrix(candles, features=None) -> (N, SEQ_LEN, n_features)
  extract_multi_tf_sequence(candles_dict, features=None)
"""

import numpy as np
from typing import List, Dict, Any, Optional

SEQ_LEN          = 30
CLIP             = 3.0
CONTEXT_LOOKBACK = 240   # max history context features may use
TIMEFRAMES       = ["5m", "1h", "4h"]   # must match MultiTFCNN.TIMEFRAMES

BASE_FEATURES    = ["open_ret", "body_ret", "upper_wick", "lower_wick",
                    "range_ret", "vol_ratio", "pos_20"]
PATTERN_FEATURES = ["vol_pressure", "engulfing", "pinbar", "dbl_retest", "compression"]
CONTEXT_FEATURES = ["sr_dist", "ob_dist", "slope_high", "slope_low", "convergence"]
ALL_FEATURES     = BASE_FEATURES + PATTERN_FEATURES + CONTEXT_FEATURES

FEATURE_GROUPS = {
    "base":     BASE_FEATURES,
    "patterns": PATTERN_FEATURES,
    "context":  CONTEXT_FEATURES,
    "all":      ALL_FEATURES,
}

# Default = the 12 features models have used so far (backward compatible)
DEFAULT_FEATURES = BASE_FEATURES + PATTERN_FEATURES
N_FEATURES = len(DEFAULT_FEATURES)   # of the DEFAULT set; actual models may differ


def resolve_features(spec) -> List[str]:
    """
    Turn a spec ("all", "base,patterns", ["base","sr_dist"], None) into an
    ordered, de-duplicated list of feature names. Order follows ALL_FEATURES
    so the same selection always produces the same column order.
    """
    if spec is None:
        return list(DEFAULT_FEATURES)
    if isinstance(spec, str):
        parts = [p.strip() for p in spec.split(",") if p.strip()]
    else:
        parts = [str(p).strip() for p in spec]
    chosen = []
    for p in parts:
        if p in FEATURE_GROUPS:
            chosen += FEATURE_GROUPS[p]
        elif p in ALL_FEATURES:
            chosen.append(p)
        else:
            raise ValueError(
                f"Unknown feature or group '{p}'. "
                f"Groups: {list(FEATURE_GROUPS)}; features: {ALL_FEATURES}"
            )
    return [f for f in ALL_FEATURES if f in set(chosen)]


# ----------------------------------------------------------------- helpers
def _rolling_atr_arrays(highs, lows, closes, period: int = 14) -> np.ndarray:
    n = len(highs)
    trs = np.empty(n)
    trs[0] = highs[0] - lows[0]
    if n > 1:
        pc = closes[:-1]
        trs[1:] = np.maximum(highs[1:] - lows[1:],
                             np.maximum(np.abs(highs[1:] - pc), np.abs(lows[1:] - pc)))
    atrs = np.empty(n)
    csum = np.cumsum(trs)
    for i in range(n):
        start = max(0, i - period + 1)
        total = csum[i] - (csum[start - 1] if start > 0 else 0.0)
        atrs[i] = total / (i - start + 1)
    return atrs


def _ema(values: np.ndarray, period: int = 20) -> np.ndarray:
    out = np.empty_like(values, dtype=float)
    k = 2.0 / (period + 1)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = values[i] * k + out[i - 1] * (1.0 - k)
    return out


def _pivots(values: np.ndarray, kind: str, wing: int = 2) -> List[int]:
    """Indices of local swing highs/lows using `wing`-bar pivots."""
    out = []
    for j in range(wing, len(values) - wing):
        seg = values[j - wing:j + wing + 1]
        if kind == "high" and values[j] >= seg.max():
            out.append(j)
        elif kind == "low" and values[j] <= seg.min():
            out.append(j)
    return out


def _proximity(price: float, levels: List[float], atr_v: float) -> float:
    """
    +1 = sitting on a level below (support), -1 = at a level above (resistance),
    0 = nothing within 3 ATR. Nearest level wins.
    """
    if not levels:
        return 0.0
    diffs = [price - lv for lv in levels]
    nearest = min(diffs, key=abs)
    closeness = max(0.0, 1.0 - abs(nearest) / (3.0 * atr_v))
    return closeness if nearest >= 0 else -closeness


# ----------------------------------------------------- per-bar feature fns
# Each takes (ctx, i) where ctx holds arrays over the window and i is the bar.
def _f_open_ret(c, i):
    return float(np.clip((c["opens"][i] - c["closes"][i-1]) / c["atr"][i], -CLIP, CLIP)) / CLIP

def _f_body_ret(c, i):
    return float(np.clip((c["closes"][i] - c["opens"][i]) / c["atr"][i], -CLIP, CLIP)) / CLIP

def _f_upper_wick(c, i):
    w = c["highs"][i] - max(c["opens"][i], c["closes"][i])
    return float(np.clip(w / c["atr"][i], 0., CLIP)) / CLIP

def _f_lower_wick(c, i):
    w = min(c["opens"][i], c["closes"][i]) - c["lows"][i]
    return float(np.clip(w / c["atr"][i], 0., CLIP)) / CLIP

def _f_range_ret(c, i):
    return float(np.clip((c["highs"][i] - c["lows"][i]) / c["atr"][i], 0., CLIP)) / CLIP

def _f_vol_ratio(c, i):
    ev = max(c["ema_vol"][i-1], 1e-8)
    return float(np.clip(c["volumes"][i] / ev, 0., 3.)) / 3.

def _f_pos_20(c, i):
    s = max(0, i - 20)
    p_high = float(np.max(c["highs"][s:i+1]))
    p_low  = float(np.min(c["lows"][s:i+1]))
    return float(np.clip((c["closes"][i] - p_low) / (p_high - p_low + 1e-8), 0., 1.))

def _f_vol_pressure(c, i):
    return float(np.sign(c["closes"][i] - c["opens"][i])) * _f_vol_ratio(c, i)

def _f_engulfing(c, i):
    body, prev = c["closes"][i] - c["opens"][i], c["closes"][i-1] - c["opens"][i-1]
    if body > 0 and prev < 0 and c["closes"][i] > c["opens"][i-1] and c["opens"][i] < c["closes"][i-1]:
        return 1.0
    if body < 0 and prev > 0 and c["closes"][i] < c["opens"][i-1] and c["opens"][i] > c["closes"][i-1]:
        return -1.0
    return 0.0

def _f_pinbar(c, i):
    body = abs(c["closes"][i] - c["opens"][i])
    uw = c["highs"][i] - max(c["opens"][i], c["closes"][i])
    lw = min(c["opens"][i], c["closes"][i]) - c["lows"][i]
    if lw > body * 1.5 and uw < body:
        return 1.0
    if uw > body * 1.5 and lw < body:
        return -1.0
    return 0.0

def _f_dbl_retest(c, i):
    tol = 0.25 * c["atr"][i]
    highs, lows = c["highs"], c["lows"]
    near_high = any(abs(highs[i] - highs[j]) <= tol
                    for j in _pivots(highs[:max(1, i-2)], "high", 1))
    near_low  = any(abs(lows[i] - lows[j]) <= tol
                    for j in _pivots(lows[:max(1, i-2)], "low", 1))
    if near_high and near_low:
        return 0.0
    return -1.0 if near_high else (1.0 if near_low else 0.0)

def _f_compression(c, i):
    s = max(0, i - 9)
    span = float(np.max(c["highs"][s:i+1]) - np.min(c["lows"][s:i+1]))
    return float(np.clip(span / (6.0 * c["atr"][i]), 0., 1.))

PER_BAR = {
    "open_ret": _f_open_ret, "body_ret": _f_body_ret,
    "upper_wick": _f_upper_wick, "lower_wick": _f_lower_wick,
    "range_ret": _f_range_ret, "vol_ratio": _f_vol_ratio, "pos_20": _f_pos_20,
    "vol_pressure": _f_vol_pressure, "engulfing": _f_engulfing,
    "pinbar": _f_pinbar, "dbl_retest": _f_dbl_retest, "compression": _f_compression,
}


# ----------------------------------------------------- context feature fns
# Each takes a history ctx (arrays over up to CONTEXT_LOOKBACK bars ending at
# the decision bar) and returns one scalar, broadcast to every window row.
def _c_sr_dist(h):
    hi_piv = [h["highs"][j] for j in _pivots(h["highs"], "high")]
    lo_piv = [h["lows"][j]  for j in _pivots(h["lows"], "low")]
    return _proximity(h["close"], hi_piv + lo_piv, h["atr_now"])

def _c_ob_dist(h):
    """Nearest unmitigated order block edge (simplified port of the TS detector)."""
    opens, highs, lows, closes = h["opens"], h["highs"], h["lows"], h["closes"]
    n = len(closes)
    bodies = np.abs(closes - opens)
    avg_body = float(np.mean(bodies)) if n else 0.0
    edges = []
    for j in range(1, n - 1):
        if bodies[j] <= avg_body * 1.5:
            continue
        impulse_up = closes[j] > opens[j]
        # last opposite-color candle before the impulse = the order block
        k = j - 1
        while k >= 0 and ((closes[k] > opens[k]) == impulse_up):
            k -= 1
        if k < 0:
            continue
        zlow, zhigh = lows[k], highs[k]
        after = slice(j + 1, n)
        touched = np.any((lows[after] <= zhigh) & (highs[after] >= zlow))
        if not touched:
            edges.append(zhigh if impulse_up else zlow)
    return _proximity(h["close"], edges, h["atr_now"])

def _slope_of(values: np.ndarray, idxs: List[int], atr_v: float) -> float:
    if len(idxs) < 2:
        return 0.0
    pts = idxs[-5:]
    x = np.array(pts, dtype=float)
    y = np.array([values[j] for j in pts], dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])          # price units per bar
    return float(np.clip(slope / (0.5 * atr_v), -1., 1.))

def _c_slope_high(h):
    lb = min(len(h["highs"]), 60)
    hs = h["highs"][-lb:]
    return _slope_of(hs, _pivots(hs, "high"), h["atr_now"])

def _c_slope_low(h):
    lb = min(len(h["lows"]), 60)
    ls = h["lows"][-lb:]
    return _slope_of(ls, _pivots(ls, "low"), h["atr_now"])

def _c_convergence(h):
    n = len(h["highs"])
    if n < 60:
        return 1.0
    now  = float(np.max(h["highs"][-20:]) - np.min(h["lows"][-20:]))
    past = float(np.max(h["highs"][-60:-40]) - np.min(h["lows"][-60:-40]))
    return float(np.clip(now / (past + 1e-8), 0., 2.)) / 2.

CONTEXT = {
    "sr_dist": _c_sr_dist, "ob_dist": _c_ob_dist,
    "slope_high": _c_slope_high, "slope_low": _c_slope_low,
    "convergence": _c_convergence,
}


# --------------------------------------------------------- single sequence
def extract_sequence(candles: List[Dict[str, Any]],
                     features: Optional[List[str]] = None) -> np.ndarray:
    """
    Uses the last SEQ_LEN+1 candles for per-bar features and up to
    CONTEXT_LOOKBACK candles of history for context features.
    Returns float32 (SEQ_LEN, len(features)). Zeros if too few candles.
    """
    names = features if features is not None else DEFAULT_FEATURES
    needed = SEQ_LEN + 1
    if len(candles) < needed:
        return np.zeros((SEQ_LEN, len(names)), dtype=np.float32)

    window = candles[-needed:]
    w_opens   = np.array([c["open"]           for c in window], dtype=float)
    w_highs   = np.array([c["high"]           for c in window], dtype=float)
    w_lows    = np.array([c["low"]            for c in window], dtype=float)
    w_closes  = np.array([c["close"]          for c in window], dtype=float)
    w_volumes = np.array([c.get("volume", 1.) for c in window], dtype=float)
    atrs = _rolling_atr_arrays(w_highs, w_lows, w_closes, 14)
    ctx = {
        "opens": w_opens, "highs": w_highs, "lows": w_lows,
        "closes": w_closes, "volumes": w_volumes,
        "atr": np.maximum(atrs, 1e-8), "ema_vol": _ema(w_volumes, 20),
    }

    per_bar_names = [f for f in names if f in PER_BAR]
    context_names = [f for f in names if f in CONTEXT]

    context_vals = {}
    if context_names:
        hist = candles[-min(len(candles), CONTEXT_LOOKBACK):]
        h_highs  = np.array([c["high"]  for c in hist], dtype=float)
        h_lows   = np.array([c["low"]   for c in hist], dtype=float)
        h_opens  = np.array([c["open"]  for c in hist], dtype=float)
        h_closes = np.array([c["close"] for c in hist], dtype=float)
        h = {
            "opens": h_opens, "highs": h_highs, "lows": h_lows, "closes": h_closes,
            "close": float(h_closes[-1]),
            "atr_now": float(max(ctx["atr"][-1], 1e-8)),
        }
        for f in context_names:
            context_vals[f] = CONTEXT[f](h)

    rows = []
    for i in range(1, needed):
        row = []
        for f in names:
            row.append(PER_BAR[f](ctx, i) if f in PER_BAR else context_vals[f])
        rows.append(row)
    return np.array(rows, dtype=np.float32)


# ------------------------------------------------------ matrix for training
def extract_sequence_matrix(candles: List[Dict[str, Any]],
                            window: int = SEQ_LEN,
                            features: Optional[List[str]] = None,
                            stride: int = 1) -> np.ndarray:
    """
    Slide over all candles and build (N, window, n_features).
    Context features see history BEFORE the window too (up to CONTEXT_LOOKBACK).
    `stride` > 1 skips samples (reduces overlap between training windows).
    """
    needed = window + 1
    samples = []
    for end in range(needed, len(candles) + 1, stride):
        start = max(0, end - CONTEXT_LOOKBACK)
        samples.append(extract_sequence(candles[start:end], features=features))
    return np.array(samples, dtype=np.float32)


# ----------------------------------------- multi-timeframe single predict
def extract_multi_tf_sequence(candles_dict: Dict[str, List[Dict[str, Any]]],
                              features: Optional[List[str]] = None
                              ) -> Dict[str, np.ndarray]:
    """candles_dict: {"5m": [...], "1h": [...], "4h": [...]}"""
    return {tf: extract_sequence(c, features=features) for tf, c in candles_dict.items()}


# --------------------------------- multi-TF alignment for training (fast)
def align_tf_index(candles_fast: List[Dict[str, Any]],
                   X_slow: np.ndarray,
                   window_end_times: List[int]) -> np.ndarray:
    """
    For each fast-TF sample at index i (>= SEQ_LEN), index of the most-recent
    slow-TF window whose last-candle timestamp <= candles_fast[i]["time"].
    """
    import bisect
    M_slow = len(X_slow)
    indices = []
    for i in range(SEQ_LEN, len(candles_fast)):
        t = candles_fast[i]["time"]
        idx = bisect.bisect_right(window_end_times, t) - 1
        indices.append(max(0, min(idx, M_slow - 1)))
    return np.array(indices, dtype=np.int32)
