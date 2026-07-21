import { Candle, Timeframe } from "../types";

export const TIMEFRAME_MS: Record<Timeframe, number> = {
  "5m": 300_000,
  "15m": 900_000,
  "1h": 3_600_000,
  "4h": 14_400_000,
  "1d": 86_400_000
};

/**
 * Aggregate candles into a higher timeframe (UTC-aligned buckets).
 * Mirrors scripts/resample_csv.py so backtests and data files agree.
 */
export function resampleCandles(candles: Candle[], timeframe: Timeframe): Candle[] {
  const tfMs = TIMEFRAME_MS[timeframe];
  const out: Candle[] = [];
  let current: Candle | undefined;

  for (const c of candles) {
    const bucket = Math.floor(c.time / tfMs) * tfMs;
    if (!current || current.time !== bucket) {
      if (current) out.push(current);
      current = { time: bucket, open: c.open, high: c.high, low: c.low, close: c.close, volume: c.volume ?? 0 };
    } else {
      current.high = Math.max(current.high, c.high);
      current.low = Math.min(current.low, c.low);
      current.close = c.close;
      current.volume = (current.volume ?? 0) + (c.volume ?? 0);
    }
  }
  if (current) out.push(current);
  return out;
}

/**
 * Candles fully CLOSED at `cutoffMs` (bucket start + duration <= cutoff),
 * windowed to the most recent `maxBars`. Keeps higher-timeframe series free
 * of lookahead when a backtest asks "what did the 4h chart show at bar i?".
 */
export function closedUpTo(
  candles: Candle[],
  timeframe: Timeframe,
  cutoffMs: number,
  maxBars: number
): Candle[] {
  const tfMs = TIMEFRAME_MS[timeframe];
  // Binary search: first index whose bucket is NOT closed at cutoff
  let lo = 0;
  let hi = candles.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (candles[mid].time + tfMs <= cutoffMs) lo = mid + 1;
    else hi = mid;
  }
  return candles.slice(Math.max(0, lo - maxBars), lo);
}
