"""
Sequence feature extraction for the XAUUSD multi-timeframe CNN model.
Pure numpy — no external dependencies.

For each candle in the window, 12 features are computed:

  0  open_ret     gap at open vs prev close, normalised by ATR
  1  body_ret     close - open, normalised by ATR
  2  upper_wick   high - max(open,close), normalised by ATR
  3  lower_wick   min(open,close) - low, normalised by ATR
  4  range_ret    candle total range (high - low), normalised by ATR
  5  vol_ratio    volume / 20-bar EMA volume  (capped at 3, normalised 0-1)
  6  pos_20       close position within 20-bar high/low range (0-1)
  7  vol_pressure signed volume: sign(body) * vol_ratio  (-1..1)
                  buying pressure > 0, selling pressure < 0
  8  engulfing    bullish engulfing +1, bearish engulfing -1, else 0
  9  pinbar       hammer/bullish pin +1, shooting star/bearish pin -1, else 0
 10  dbl_retest   bar retesting a prior swing low +1 (double-bottom zone),
                  prior swing high -1 (double-top zone), else 0
 11  compression  10-bar total range / (6 * ATR), clipped 0-1
                  low values = squeeze (triangle/flag), high = expansion

Main entry points:
  extract_sequence(candles)                    -> ndarray (SEQ_LEN, N_FEATURES)
  extract_sequence_matrix(candles)             -> ndarray (N, SEQ_LEN, N_FEATURES)
  extract_multi_tf_sequence(candles_dict)      -> dict {tf: ndarray (SEQ_LEN, N_FEATURES)}
"""

import numpy as np
from typing import List, Dict, Any

SEQ_LEN    = 30
N_FEATURES = 12
CLIP       = 3.0
TIMEFRAMES = ["5m", "1h", "4h"]   # must match MultiTFCNN.TIMEFRAMES


# ----------------------------------------------------------------- helpers
def _rolling_atr(candles: List[Dict[str, Any]], period: int = 14) -> List[float]:
    trs = [candles[0]["high"] - candles[0]["low"]]
    for i in range(1, len(candles)):
        h, l, pc = candles[i]["high"], candles[i]["low"], candles[i - 1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    atrs = []
    for i in range(len(trs)):
        start = max(0, i - period + 1)
        atrs.append(float(np.mean(trs[start:i + 1])))
    return atrs


def _ema_volume(volumes: np.ndarray, period: int = 20) -> np.ndarray:
    out = np.empty_like(volumes, dtype=float)
    k = 2.0 / (period + 1)
    out[0] = volumes[0]
    for i in range(1, len(volumes)):
        out[i] = volumes[i] * k + out[i - 1] * (1.0 - k)
    return out


def _swing_indices(values: np.ndarray, end: int, kind: str) -> List[int]:
    """Local swing highs/lows among bars [1, end-3] (3-bar pivots)."""
    out = []
    for j in range(1, max(1, end - 2)):
        if j - 1 < 0 or j + 1 >= len(values):
            continue
        if kind == "high" and values[j] >= values[j - 1] and values[j] >= values[j + 1]:
            out.append(j)
        if kind == "low" and values[j] <= values[j - 1] and values[j] <= values[j + 1]:
            out.append(j)
    return out


# --------------------------------------------------------- single sequence
def extract_sequence(candles: List[Dict[str, Any]]) -> np.ndarray:
    """
    Takes the last SEQ_LEN+1 candles.
    Returns float32 array (SEQ_LEN, N_FEATURES). Returns zeros if too few candles.
    """
    needed = SEQ_LEN + 1
    if len(candles) < needed:
        return np.zeros((SEQ_LEN, N_FEATURES), dtype=np.float32)

    window = candles[-needed:]
    atrs   = _rolling_atr(window, 14)

    opens   = np.array([c["open"]           for c in window], dtype=float)
    highs   = np.array([c["high"]           for c in window], dtype=float)
    lows    = np.array([c["low"]            for c in window], dtype=float)
    closes  = np.array([c["close"]          for c in window], dtype=float)
    volumes = np.array([c.get("volume", 1.) for c in window], dtype=float)
    ema_vol = _ema_volume(volumes, 20)

    rows = []
    for i in range(1, needed):
        atr_v = max(atrs[i], 1e-8)
        body  = closes[i] - opens[i]

        f0 = float(np.clip((opens[i] - closes[i-1]) / atr_v, -CLIP, CLIP)) / CLIP
        f1 = float(np.clip(body / atr_v, -CLIP, CLIP)) / CLIP
        f2 = float(np.clip((highs[i] - max(opens[i], closes[i])) / atr_v, 0., CLIP)) / CLIP
        f3 = float(np.clip((min(opens[i], closes[i]) - lows[i])  / atr_v, 0., CLIP)) / CLIP
        f4 = float(np.clip((highs[i] - lows[i]) / atr_v, 0., CLIP)) / CLIP
        ev = max(ema_vol[i-1], 1e-8)
        f5 = float(np.clip(volumes[i] / ev, 0., 3.)) / 3.
        start20 = max(0, i - 20)
        p_high = float(np.max(highs[start20:i+1]))
        p_low  = float(np.min(lows[start20:i+1]))
        f6 = float(np.clip((closes[i] - p_low) / (p_high - p_low + 1e-8), 0., 1.))

        # -- volume pressure: signed volume ratio (buying vs selling) --------
        f7 = float(np.sign(body)) * f5

        # -- candlestick patterns (needs prev bar) ---------------------------
        prev_body = closes[i-1] - opens[i-1]
        bull_engulf = (body > 0 and prev_body < 0
                       and closes[i] > opens[i-1] and opens[i] < closes[i-1])
        bear_engulf = (body < 0 and prev_body > 0
                       and closes[i] < opens[i-1] and opens[i] > closes[i-1])
        f8 = 1.0 if bull_engulf else (-1.0 if bear_engulf else 0.0)

        abs_body   = abs(body)
        upper_wick = highs[i] - max(opens[i], closes[i])
        lower_wick = min(opens[i], closes[i]) - lows[i]
        bull_pin = lower_wick > abs_body * 1.5 and upper_wick < abs_body
        bear_pin = upper_wick > abs_body * 1.5 and lower_wick < abs_body
        f9 = 1.0 if bull_pin else (-1.0 if bear_pin else 0.0)

        # -- double top/bottom retest: current bar tests a prior 3-bar swing --
        tol = 0.25 * atr_v
        near_swing_high = any(abs(highs[i] - highs[j]) <= tol
                              for j in _swing_indices(highs, i, "high"))
        near_swing_low  = any(abs(lows[i] - lows[j]) <= tol
                              for j in _swing_indices(lows, i, "low"))
        if near_swing_high and near_swing_low:
            f10 = 0.0
        elif near_swing_high:
            f10 = -1.0
        elif near_swing_low:
            f10 = 1.0
        else:
            f10 = 0.0

        # -- range compression over last 10 bars (triangle/flag squeeze) -----
        start10 = max(0, i - 9)
        span10  = float(np.max(highs[start10:i+1]) - np.min(lows[start10:i+1]))
        f11 = float(np.clip(span10 / (6.0 * atr_v), 0., 1.))

        rows.append([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11])

    return np.array(rows, dtype=np.float32)   # (SEQ_LEN, N_FEATURES)


# ------------------------------------------------------ matrix for training
def extract_sequence_matrix(
    candles: List[Dict[str, Any]],
    window: int = SEQ_LEN
) -> np.ndarray:
    """
    Slide over all candles and build a 3-D feature matrix.
    Returns shape (N_samples, window, N_FEATURES).
    """
    needed = window + 1
    samples = []
    for end in range(needed, len(candles) + 1):
        samples.append(extract_sequence(candles[end - needed:end]))
    return np.array(samples, dtype=np.float32)   # (N, window, N_FEATURES)


# ----------------------------------------- multi-timeframe single predict
def extract_multi_tf_sequence(
    candles_dict: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, np.ndarray]:
    """
    Extract a sequence for each timeframe in candles_dict.

    candles_dict: {"5m": [...], "1h": [...], "4h": [...]}
    Returns:      {"5m": (SEQ_LEN, N_FEATURES), "1h": ..., "4h": ...}
    """
    return {tf: extract_sequence(candles) for tf, candles in candles_dict.items()}


# --------------------------------- multi-TF alignment for training (fast)
def align_tf_index(
    candles_fast: List[Dict[str, Any]],
    X_slow: np.ndarray,
    window_end_times: List[int],
) -> np.ndarray:
    """
    For each fast-TF sample at index i (>= SEQ_LEN),
    return the index of the most-recent slow-TF window
    whose last-candle timestamp <= candles_fast[i]["time"].

    window_end_times[j] = candles_slow[j + SEQ_LEN]["time"]
                          (precomputed outside this function)

    Returns int32 array of shape (N_fast_samples,)
    where N_fast_samples = len(candles_fast) - SEQ_LEN
    """
    import bisect
    M_slow = len(X_slow)
    indices = []
    for i in range(SEQ_LEN, len(candles_fast)):
        t   = candles_fast[i]["time"]
        idx = bisect.bisect_right(window_end_times, t) - 1
        indices.append(max(0, min(idx, M_slow - 1)))
    return np.array(indices, dtype=np.int32)
