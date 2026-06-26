from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class TradeSignal:
    action: str
    reason: str
    entry: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    fvg_low: Optional[float] = None
    fvg_high: Optional[float] = None
    order_block_low: Optional[float] = None
    order_block_high: Optional[float] = None
    poc: Optional[float] = None
    liquidity_level: Optional[float] = None
    order_block_confluence: bool = False
    poc_confluence: bool = False


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def trend_bias(h1: pd.DataFrame) -> str:
    data = h1.copy()
    data["ema50"] = ema(data["close"], 50)
    data["ema200"] = ema(data["close"], 200)

    latest = data.iloc[-1]
    if latest["ema50"] > latest["ema200"]:
        return "bullish"
    if latest["ema50"] < latest["ema200"]:
        return "bearish"
    return "neutral"


def _swing_low(df: pd.DataFrame, lookback: int) -> float:
    return float(df["low"].tail(lookback).min())


def _swing_high(df: pd.DataFrame, lookback: int) -> float:
    return float(df["high"].tail(lookback).max())


def _average_range(df: pd.DataFrame, lookback: int = 20) -> float:
    ranges = df["high"].tail(lookback) - df["low"].tail(lookback)
    return float(ranges.mean())


def _zones_overlap(a_low: float, a_high: float, b_low: float, b_high: float) -> bool:
    return max(a_low, b_low) <= min(a_high, b_high)


def _zone_distance(a_low: float, a_high: float, b_low: float, b_high: float) -> float:
    if _zones_overlap(a_low, a_high, b_low, b_high):
        return 0.0
    if a_high < b_low:
        return b_low - a_high
    return a_low - b_high


def detect_liquidity_sweep(m5: pd.DataFrame, direction: str, lookback: int = 20):
    if len(m5) < lookback + 2:
        return None

    recent = m5.iloc[-1]
    previous = m5.iloc[-(lookback + 1):-1]

    if direction == "bullish":
        prior_low = previous["low"].min()
        if recent["low"] < prior_low and recent["close"] > prior_low:
            return {"direction": "bullish", "level": float(prior_low)}
        return None

    if direction == "bearish":
        prior_high = previous["high"].max()
        if recent["high"] > prior_high and recent["close"] < prior_high:
            return {"direction": "bearish", "level": float(prior_high)}
        return None

    return None


def detect_liquidity_zone(m5: pd.DataFrame, direction: str, lookback: int = 60):
    if len(m5) < lookback + 2:
        return None

    previous = m5.iloc[-(lookback + 1):-1]
    tolerance = max(_average_range(previous, 20) * 0.2, 0.01)

    if direction == "bullish":
        lows = previous["low"].sort_values().head(5)
        level = float(lows.median())
        touches = int((previous["low"].sub(level).abs() <= tolerance).sum())
        if touches >= 2:
            return {"type": "equal_lows", "level": level, "touches": touches}
        return {"type": "swing_low", "level": float(previous["low"].min()), "touches": 1}

    if direction == "bearish":
        highs = previous["high"].sort_values(ascending=False).head(5)
        level = float(highs.median())
        touches = int((previous["high"].sub(level).abs() <= tolerance).sum())
        if touches >= 2:
            return {"type": "equal_highs", "level": level, "touches": touches}
        return {"type": "swing_high", "level": float(previous["high"].max()), "touches": 1}

    return None


def detect_market_structure_shift(m5: pd.DataFrame, direction: str, lookback: int = 12) -> bool:
    if len(m5) < lookback + 3:
        return False

    recent_close = m5.iloc[-1]["close"]
    previous = m5.iloc[-(lookback + 1):-1]

    if direction == "bullish":
        return bool(recent_close > previous["high"].max())

    if direction == "bearish":
        return bool(recent_close < previous["low"].min())

    return False


def detect_fair_value_gap(m5: pd.DataFrame, direction: str):
    if len(m5) < 3:
        return None

    first = m5.iloc[-3]
    third = m5.iloc[-1]

    if direction == "bullish" and third["low"] > first["high"]:
        return {
            "direction": "bullish",
            "low": float(first["high"]),
            "high": float(third["low"]),
        }

    if direction == "bearish" and third["high"] < first["low"]:
        return {
            "direction": "bearish",
            "low": float(third["high"]),
            "high": float(first["low"]),
        }

    return None


def calculate_poc(m5: pd.DataFrame, lookback: int = 120, buckets: int = 48):
    if len(m5) < 20:
        return None

    data = m5.tail(lookback).copy()
    if "tick_volume" not in data.columns:
        return None

    low = float(data["low"].min())
    high = float(data["high"].max())
    if high <= low:
        return None

    typical_price = (data["high"] + data["low"] + data["close"]) / 3.0
    bucket_size = (high - low) / buckets
    data["poc_bucket"] = ((typical_price - low) / bucket_size).clip(0, buckets - 1).astype(int)
    volume_by_bucket = data.groupby("poc_bucket")["tick_volume"].sum()
    poc_bucket = int(volume_by_bucket.idxmax())
    return low + (poc_bucket + 0.5) * bucket_size


def detect_order_block(m5: pd.DataFrame, direction: str, lookback: int = 30):
    if len(m5) < lookback + 2:
        return None

    previous = m5.iloc[-(lookback + 1):-1]

    if direction == "bullish":
        candidates = previous[previous["close"] < previous["open"]]
    elif direction == "bearish":
        candidates = previous[previous["close"] > previous["open"]]
    else:
        return None

    if candidates.empty:
        return None

    candle = candidates.iloc[-1]
    return {
        "direction": direction,
        "low": float(candle["low"]),
        "high": float(candle["high"]),
        "time": candle.get("time"),
    }


def has_poc_confluence(entry: float, poc: Optional[float], m5: pd.DataFrame, multiplier: float = 3.0) -> bool:
    if poc is None:
        return False
    max_distance = max(_average_range(m5, 20) * multiplier, 0.01)
    return abs(entry - poc) <= max_distance


def has_order_block_confluence(fvg: dict, order_block: dict, m5: pd.DataFrame) -> bool:
    tolerance = max(_average_range(m5, 20), 0.01)
    distance = _zone_distance(fvg["low"], fvg["high"], order_block["low"], order_block["high"])
    return distance <= tolerance


def build_signal(
    m5: pd.DataFrame,
    h1: pd.DataFrame,
    risk_reward: float = 2.0,
    swing_lookback: int = 20,
) -> TradeSignal:
    if len(h1) < 210:
        return TradeSignal("WAIT", "Need at least 210 H1 candles for EMA200 warmup")
    if len(m5) < max(30, swing_lookback + 3):
        return TradeSignal("WAIT", "Need more M5 candles")

    bias = trend_bias(h1)
    if bias not in {"bullish", "bearish"}:
        return TradeSignal("WAIT", "H1 EMA50 and EMA200 are flat/neutral")

    sweep = detect_liquidity_sweep(m5, bias)
    if not sweep:
        return TradeSignal("WAIT", f"No {bias} liquidity sweep")

    liquidity_zone = detect_liquidity_zone(m5, bias)
    if liquidity_zone is None:
        return TradeSignal("WAIT", f"No {bias} liquidity zone")

    mss = detect_market_structure_shift(m5, bias)
    if not mss:
        return TradeSignal("WAIT", f"No {bias} market structure shift")

    fvg = detect_fair_value_gap(m5, bias)
    if fvg is None:
        return TradeSignal("WAIT", f"No {bias} fair value gap")

    entry = float(m5.iloc[-1]["close"])
    order_block = detect_order_block(m5, bias)
    order_block_confluence = (
        has_order_block_confluence(fvg, order_block, m5) if order_block is not None else False
    )
    poc = calculate_poc(m5)
    poc_confluence = has_poc_confluence(entry, poc, m5)

    if bias == "bullish":
        sl = _swing_low(m5.iloc[:-1], swing_lookback)
        risk = entry - sl
        if risk <= 0 or np.isclose(risk, 0):
            return TradeSignal("WAIT", "Invalid bullish stop distance")
        tp = entry + (risk * risk_reward)
        return TradeSignal(
            "BUY",
            "Bullish bias + liquidity + MSS + FVG",
            entry,
            sl,
            tp,
            fvg["low"],
            fvg["high"],
            order_block["low"] if order_block else None,
            order_block["high"] if order_block else None,
            poc,
            liquidity_zone["level"],
            order_block_confluence,
            poc_confluence,
        )

    sl = _swing_high(m5.iloc[:-1], swing_lookback)
    risk = sl - entry
    if risk <= 0 or np.isclose(risk, 0):
        return TradeSignal("WAIT", "Invalid bearish stop distance")
    tp = entry - (risk * risk_reward)
    return TradeSignal(
        "SELL",
        "Bearish bias + liquidity + MSS + FVG",
        entry,
        sl,
        tp,
        fvg["low"],
        fvg["high"],
        order_block["low"] if order_block else None,
        order_block["high"] if order_block else None,
        poc,
        liquidity_zone["level"],
        order_block_confluence,
        poc_confluence,
    )
