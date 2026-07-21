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
  graded (3)    continuous replacements for the coarse patterns above — same
                meaning, magnitude = strength (all ATR/ratio-normalized, clipped)
      engulf_strength   engulf body / prev body, signed (+bull, -bear), 0 = none
      pinbar_strength   dominant wick / body, signed (+hammer, -star)
      dbl_retest_dist   signed ATR distance to nearest prior swing
                        (+ near swing low, - near swing high)
  time (2)      cyclical time-of-day (UTC), continuous intraday seasonality
      tod_sin, tod_cos  sin/cos of the fraction of the UTC day at the bar
  context (10)  market-structure context, computed at the decision bar over up
                to CONTEXT_LOOKBACK bars of history, broadcast to all rows
      sr_dist       S/R proximity: +1 sitting on support, -1 at resistance,
                    0 = nothing near (within 3 ATR)
      ob_dist       unmitigated order-block proximity, same encoding as sr_dist
      slope_high    regression slope of recent swing HIGHS (ATR/bar, clipped)
      slope_low     regression slope of recent swing LOWS  (ATR/bar, clipped)
                    falling wedge: both < 0 converging; descending triangle:
                    slope_low ~ 0, slope_high < 0; rectangle: both ~ 0
      convergence   channel width now / width 40 bars ago (0-1 = contracting)
      fvg_dist      unfilled fair-value-gap proximity, same encoding as sr_dist
      round_dist    psychological round-number level proximity ($25 grid)
      trendline_bounce  price sitting on a fitted swing-pivot trendline and
                    rejecting off it (+1 bounce up off support line, -1 reject
                    down off resistance line), scaled by wick-rejection strength
      trendline_break   current candle closed THROUGH a fitted trendline
                    (+1 broke above resistance line, -1 broke below support line)
      atr_percentile    percentile rank (0-1) of current ATR over last 100 bars
                    (one volatility-regime number; 0 = quiet, 1 = most volatile)

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
# Graded (continuous) replacements for the coarse -1/0/+1 pattern flags. Same
# trading meaning, but magnitude carries strength instead of a bare category.
# All clipped to CLIP so a near-zero denominator can't produce a fat tail.
GRADED_FEATURES  = ["engulf_strength", "pinbar_strength", "dbl_retest_dist"]
# Cyclical time-of-day (UTC): captures intraday seasonality (session rhythm)
# continuously, with no hard session boundaries. Works on any timeframe.
TIME_FEATURES    = ["tod_sin", "tod_cos"]
CONTEXT_FEATURES = ["sr_dist", "ob_dist", "slope_high", "slope_low", "convergence",
                    "fvg_dist", "round_dist", "trendline_bounce", "trendline_break",
                    "atr_percentile"]
ALL_FEATURES     = (BASE_FEATURES + PATTERN_FEATURES + GRADED_FEATURES
                    + TIME_FEATURES + CONTEXT_FEATURES)

FEATURE_GROUPS = {
    "base":      BASE_FEATURES,
    "patterns":  PATTERN_FEATURES,
    "graded":    GRADED_FEATURES,
    "time":      TIME_FEATURES,
    "context":   CONTEXT_FEATURES,
    "poi":       ["sr_dist", "ob_dist", "fvg_dist", "round_dist"],
    "trendline": ["trendline_bounce", "trendline_break"],
    "all":       ALL_FEATURES,
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

# ---- graded (continuous) pattern features -----------------------------------
def _f_engulf_strength(c, i):
    """
    Graded engulfing: magnitude = this body / previous body (clipped), signed by
    direction. +1-ish = strong bullish engulf, -1-ish = strong bearish engulf,
    0 = no engulf. Replaces the bare +1/-1 flag with engulf *strength*.
    """
    body, prev = c["closes"][i] - c["opens"][i], c["closes"][i-1] - c["opens"][i-1]
    bullish = body > 0 and prev < 0 and c["closes"][i] > c["opens"][i-1] and c["opens"][i] < c["closes"][i-1]
    bearish = body < 0 and prev > 0 and c["closes"][i] < c["opens"][i-1] and c["opens"][i] > c["closes"][i-1]
    if not (bullish or bearish):
        return 0.0
    ratio = abs(body) / (abs(prev) + 1e-8)
    mag = float(np.clip(ratio, 0., CLIP)) / CLIP
    return mag if bullish else -mag

def _f_pinbar_strength(c, i):
    """
    Graded pinbar: dominant-wick / body (clipped), signed. +1-ish = strong
    hammer (lower wick dominates), -1-ish = strong shooting star. Replaces the
    +1/-1 flag with rejection *strength*.
    """
    o, cl = c["opens"][i], c["closes"][i]
    body = abs(cl - o) + 1e-8
    uw = c["highs"][i] - max(o, cl)
    lw = min(o, cl) - c["lows"][i]
    dominant, sign = (lw, 1.0) if lw >= uw else (uw, -1.0)
    mag = float(np.clip(dominant / body, 0., CLIP)) / CLIP
    return sign * mag

def _f_dbl_retest_dist(c, i):
    """
    Graded double-retest: signed distance (in ATR) from the current close to the
    nearest prior swing. Positive = near a prior swing LOW (support retest),
    negative = near a prior swing HIGH (resistance retest), 0 = no prior swing.
    Replaces the +1/-1 flag with how *close* the retest is.
    """
    highs, lows = c["highs"], c["lows"]
    hi_piv = _pivots(highs[:max(1, i-2)], "high", 1)
    lo_piv = _pivots(lows[:max(1, i-2)], "low", 1)
    close = c["closes"][i]
    cands = []   # (abs_dist, signed_value)
    if hi_piv:
        nearest_hi = min((highs[j] for j in hi_piv), key=lambda p: abs(p - close))
        d = abs(nearest_hi - close) / c["atr"][i]
        cands.append((abs(nearest_hi - close), -float(np.clip(d, 0., CLIP)) / CLIP))
    if lo_piv:
        nearest_lo = min((lows[j] for j in lo_piv), key=lambda p: abs(p - close))
        d = abs(nearest_lo - close) / c["atr"][i]
        cands.append((abs(nearest_lo - close), float(np.clip(d, 0., CLIP)) / CLIP))
    if not cands:
        return 0.0
    return min(cands, key=lambda t: t[0])[1]

# ---- cyclical time-of-day (UTC) ---------------------------------------------
_MS_PER_DAY = 86_400_000.0

def _f_tod_sin(c, i):
    frac = (float(c["times"][i]) % _MS_PER_DAY) / _MS_PER_DAY
    return float(np.sin(2.0 * np.pi * frac))

def _f_tod_cos(c, i):
    frac = (float(c["times"][i]) % _MS_PER_DAY) / _MS_PER_DAY
    return float(np.cos(2.0 * np.pi * frac))

PER_BAR = {
    "open_ret": _f_open_ret, "body_ret": _f_body_ret,
    "upper_wick": _f_upper_wick, "lower_wick": _f_lower_wick,
    "range_ret": _f_range_ret, "vol_ratio": _f_vol_ratio, "pos_20": _f_pos_20,
    "vol_pressure": _f_vol_pressure, "engulfing": _f_engulfing,
    "pinbar": _f_pinbar, "dbl_retest": _f_dbl_retest, "compression": _f_compression,
    "engulf_strength": _f_engulf_strength, "pinbar_strength": _f_pinbar_strength,
    "dbl_retest_dist": _f_dbl_retest_dist,
    "tod_sin": _f_tod_sin, "tod_cos": _f_tod_cos,
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


# --------------------------------------------- trendline fitting helpers
TREND_LB = 80   # bars a fitted trendline may span (kept local so the line
                # tracks the current leg, not ancient history)

def _fit_line(values: np.ndarray, idxs: List[int]):
    """
    Least-squares line through the last up-to-5 pivot points.
    Returns (slope, intercept) in price-vs-bar-index units, or None if there
    are too few pivots to define a line.
    """
    if len(idxs) < 2:
        return None
    pts = idxs[-5:]
    x = np.array(pts, dtype=float)
    y = np.array([values[j] for j in pts], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def _project(line, x: float) -> float:
    return line[0] * x + line[1]


def _trend_lines(h):
    """
    Fit a resistance line through swing HIGHS and a support line through swing
    LOWS over the last TREND_LB bars. Returns (x_now, res_line, sup_line) where
    x_now is the current bar's index within that local window (so projections
    land on the live candle). res_line / sup_line may be None.
    """
    n  = len(h["highs"])
    lb = min(n, TREND_LB)
    highs = h["highs"][-lb:]
    lows  = h["lows"][-lb:]
    x_now = lb - 1
    res = _fit_line(highs, _pivots(highs, "high"))
    sup = _fit_line(lows,  _pivots(lows,  "low"))
    return x_now, res, sup

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

def _c_fvg_dist(h):
    """Nearest unfilled fair-value-gap edge (3-candle imbalance), sr_dist encoding."""
    highs, lows = h["highs"], h["lows"]
    n = len(highs)
    edges = []
    for j in range(2, n):
        if lows[j] > highs[j - 2]:                    # bullish FVG (below price)
            top, bottom = lows[j], highs[j - 2]
            later = lows[j + 1:]
            if later.size == 0 or float(np.min(later)) > bottom:   # not fully filled
                edges.append(top)
        elif highs[j] < lows[j - 2]:                  # bearish FVG (above price)
            top, bottom = lows[j - 2], highs[j]
            later = highs[j + 1:]
            if later.size == 0 or float(np.max(later)) < top:
                edges.append(bottom)
    return _proximity(h["close"], edges, h["atr_now"])

ROUND_GRID = 25.0   # gold psychological levels: 2650, 2675, 2700 ...

def _c_round_dist(h):
    """Proximity to the nearest round-number level, sr_dist encoding."""
    below = float(np.floor(h["close"] / ROUND_GRID)) * ROUND_GRID
    return _proximity(h["close"], [below, below + ROUND_GRID], h["atr_now"])


def _c_trendline_bounce(h):
    """
    Trendline BOUNCE: price is sitting on a fitted trendline and rejecting off
    it. sr_dist encoding — +1 = bouncing up off a support trendline (line below
    price), -1 = rejecting down off a resistance trendline (line above price),
    0 = no line nearby. The proximity magnitude is scaled by a rejection factor:
    full weight when the current candle shows a wick rejecting off the line
    (lower wick at support, upper wick at resistance), reduced weight otherwise.
    """
    x_now, res, sup = _trend_lines(h)
    levels = []
    if sup is not None:
        levels.append(_project(sup, x_now))
    if res is not None:
        levels.append(_project(res, x_now))
    if not levels:
        return 0.0

    prox = _proximity(h["close"], levels, h["atr_now"])
    if prox == 0.0:
        return 0.0

    o, c = float(h["opens"][-1]), float(h["closes"][-1])
    body = abs(c - o)
    upper_wick = float(h["highs"][-1]) - max(o, c)
    lower_wick = min(o, c) - float(h["lows"][-1])
    if prox > 0:      # near a support line -> want a lower-wick rejection
        conf = 1.0 if lower_wick >= body else 0.4
    else:             # near a resistance line -> want an upper-wick rejection
        conf = 1.0 if upper_wick >= body else 0.4
    return float(prox * conf)


def _c_atr_percentile(h):
    """
    Volatility regime as a single continuous number: percentile rank (0-1) of
    the current ATR within the last 100 bars of ATR history. 0 = quietest in
    memory, 1 = most volatile. One feature instead of three overlapping ones.
    """
    highs, lows, closes = h["highs"], h["lows"], h["closes"]
    if len(closes) < 5:
        return 0.5
    atrs = _rolling_atr_arrays(highs, lows, closes, 14)
    window = atrs[-100:]
    cur = atrs[-1]
    return float(np.mean(window <= cur))


def _c_trendline_break(h):
    """
    Trendline BREAK: the current candle CLOSED THROUGH a fitted trendline.
    +1 = closed above a (descending/flat) resistance line having been below it,
    -1 = closed below a (rising/flat) support line having been above it,
    0 = no break. Uses a 0.1-ATR tolerance so noise around the line doesn't
    register as a breakout.
    """
    x_now, res, sup = _trend_lines(h)
    closes = h["closes"]
    if len(closes) < 2:
        return 0.0
    tol        = 0.1 * h["atr_now"]
    close_now  = float(closes[-1])
    close_prev = float(closes[-2])

    if res is not None:
        r_now  = _project(res, x_now)
        r_prev = _project(res, x_now - 1)
        if close_prev <= r_prev + tol and close_now > r_now + tol:
            return 1.0
    if sup is not None:
        s_now  = _project(sup, x_now)
        s_prev = _project(sup, x_now - 1)
        if close_prev >= s_prev - tol and close_now < s_now - tol:
            return -1.0
    return 0.0

CONTEXT = {
    "sr_dist": _c_sr_dist, "ob_dist": _c_ob_dist,
    "slope_high": _c_slope_high, "slope_low": _c_slope_low,
    "convergence": _c_convergence,
    "fvg_dist": _c_fvg_dist, "round_dist": _c_round_dist,
    "trendline_bounce": _c_trendline_bounce, "trendline_break": _c_trendline_break,
    "atr_percentile": _c_atr_percentile,
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
    w_times   = np.array([c.get("time", 0)    for c in window], dtype=float)
    atrs = _rolling_atr_arrays(w_highs, w_lows, w_closes, 14)
    ctx = {
        "opens": w_opens, "highs": w_highs, "lows": w_lows,
        "closes": w_closes, "volumes": w_volumes, "times": w_times,
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
