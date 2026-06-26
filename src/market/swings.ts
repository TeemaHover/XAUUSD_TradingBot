import { Candle, SwingPoint } from "../types";

export function detectSwings(candles: Candle[], lookback: number): SwingPoint[] {
  const swings: SwingPoint[] = [];
  if (candles.length < lookback * 2 + 1) return swings;

  for (let i = lookback; i < candles.length - lookback; i += 1) {
    const candle = candles[i];
    const left = candles.slice(i - lookback, i);
    const right = candles.slice(i + 1, i + lookback + 1);
    const isHigh = [...left, ...right].every((other) => candle.high >= other.high);
    const isLow = [...left, ...right].every((other) => candle.low <= other.low);

    if (isHigh) swings.push({ index: i, time: candle.time, price: candle.high, type: "high" });
    if (isLow) swings.push({ index: i, time: candle.time, price: candle.low, type: "low" });
  }

  return swings;
}

export function lastSwing(swings: SwingPoint[], type: "high" | "low"): SwingPoint | undefined {
  return [...swings].reverse().find((swing) => swing.type === type);
}
