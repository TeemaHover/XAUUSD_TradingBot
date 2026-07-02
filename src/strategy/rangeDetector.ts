import { atr } from "../indicators/atr";
import { detectSwings } from "../market/swings";
import { AppConfig, Candle, Timeframe } from "../types";

export interface RangeResult {
  detected: boolean;
  high: number;
  low: number;
  mid: number;
  /** 0 = at range low, 100 = at range high */
  pricePosition: number;
  /** width of range in ATR units */
  widthAtr: number;
  /** how many times price touched resistance */
  resistanceTouches: number;
  /** how many times price touched support */
  supportTouches: number;
  /** which timeframe the candles came from */
  timeframe: Timeframe;
  summary: string;
}

/**
 * Detects a confirmed consolidation range.
 *
 * A valid range requires:
 *   - At least 2 swing highs clustered near the same resistance level
 *   - At least 2 swing lows clustered near the same support level
 *   - Price alternated between the two levels (rejected from top, bounced from bottom)
 *   - Current price is still inside the range (no breakout)
 *   - Range width >= 1.5 ATR
 */
export function detectRange(candles: Candle[], config: AppConfig, timeframe: Timeframe = "5m"): RangeResult {
  const none: RangeResult = {
    detected: false, high: 0, low: 0, mid: 0,
    pricePosition: 50, widthAtr: 0,
    resistanceTouches: 0, supportTouches: 0,
    timeframe, summary: "No confirmed range"
  };

  const latest = candles.at(-1);
  if (!latest || candles.length < config.strategy.swingLookback * 3) return none;

  const atrValues = atr(candles, config.strategy.atrLength);
  const atrValue = atrValues.at(-1) ?? 1;
  // Slightly wider tolerance so touches at "the same area" cluster correctly
  const tolerance = atrValue * Math.max(config.strategy.srZoneToleranceAtr, 0.5);

  const swings = detectSwings(candles, config.strategy.swingLookback);
  // Use only recent swings — too many dilutes the signal
  const recentHighs = swings.filter((s) => s.type === "high").slice(-20);
  const recentLows  = swings.filter((s) => s.type === "low").slice(-20);

  if (recentHighs.length < 2 || recentLows.length < 2) {
    return { ...none, summary: "Not enough swing points" };
  }

  // Find best resistance cluster: the cluster of highs where most swing highs group together
  const resistanceCluster = bestCluster(recentHighs.map((s) => s.price), tolerance);
  if (resistanceCluster.count < 2) {
    return { ...none, summary: "No resistance cluster (need >= 2 touches)" };
  }

  // Find best support cluster: same for lows
  const supportCluster = bestCluster(recentLows.map((s) => s.price), tolerance);
  if (supportCluster.count < 2) {
    return { ...none, summary: "No support cluster (need >= 2 touches)" };
  }

  const rangeHigh = resistanceCluster.level;
  const rangeLow  = supportCluster.level;

  if (rangeHigh <= rangeLow) {
    return { ...none, summary: "Support above resistance - invalid" };
  }

  const width = rangeHigh - rangeLow;
  const widthAtr = width / atrValue;

  if (widthAtr < 1.5) {
    return { ...none, summary: "[" + timeframe + "] Range too narrow (" + widthAtr.toFixed(1) + " ATR)" };
  }

  // Confirm alternation: get all touches time-sorted and check they bounce between sides
  const resistanceTouchPoints = recentHighs
    .filter((s) => Math.abs(s.price - rangeHigh) <= tolerance)
    .map((s) => ({ time: s.index, side: "resistance" as const }));
  const supportTouchPoints = recentLows
    .filter((s) => Math.abs(s.price - rangeLow) <= tolerance)
    .map((s) => ({ time: s.index, side: "support" as const }));

  const allTouches = [...resistanceTouchPoints, ...supportTouchPoints]
    .sort((a, b) => a.time - b.time);

  // Count direction changes (resistance->support or support->resistance)
  let alternations = 0;
  for (let i = 1; i < allTouches.length; i++) {
    if (allTouches[i].side !== allTouches[i - 1].side) alternations++;
  }

  const minAlternations = config.strategy.rangeMinAlternations ?? 2;
  if (alternations < minAlternations) {
    return { ...none, summary: "Price not alternating between levels (need >= " + minAlternations + " bounces)" };
  }

  // Make sure price hasn't broken out of the range already
  const price = latest.close;
  const breakoutBuffer = atrValue * 0.3;
  if (price > rangeHigh + breakoutBuffer || price < rangeLow - breakoutBuffer) {
    return { ...none, summary: "[" + timeframe + "] Range broken (price outside bounds)" };
  }

  const mid = (rangeHigh + rangeLow) / 2;
  const pricePosition = Math.max(0, Math.min(100, ((price - rangeLow) / width) * 100));

  const zone = pricePosition >= 70 ? "upper third (near resistance)"
    : pricePosition <= 30         ? "lower third (near support)"
    : "middle of range";

  const summary = "[" + timeframe + "] Range " + rangeLow.toFixed(2) + " - " + rangeHigh.toFixed(2)
    + " | R:" + resistanceCluster.count + "x S:" + supportCluster.count + "x"
    + " | mid " + mid.toFixed(2)
    + " | price at " + pricePosition.toFixed(0) + "% (" + zone + ")";

  return {
    detected: true,
    high: rangeHigh,
    low: rangeLow,
    mid,
    pricePosition,
    widthAtr,
    resistanceTouches: resistanceCluster.count,
    supportTouches: supportCluster.count,
    timeframe,
    summary
  };
}

interface Cluster {
  level: number;
  count: number;
}

/**
 * Groups prices into clusters within tolerance and returns the cluster
 * with the most members (most-tested level).
 */
function bestCluster(prices: number[], tolerance: number): Cluster {
  if (prices.length === 0) return { level: 0, count: 0 };

  const sorted = [...prices].sort((a, b) => a - b);
  const clusters: number[][] = [[sorted[0]]];

  for (let i = 1; i < sorted.length; i++) {
    const last = clusters[clusters.length - 1];
    const avg = last.reduce((s, v) => s + v, 0) / last.length;
    if (Math.abs(sorted[i] - avg) <= tolerance) {
      last.push(sorted[i]);
    } else {
      clusters.push([sorted[i]]);
    }
  }

  const best = clusters.reduce((a, b) => (b.length > a.length ? b : a));
  const level = best.reduce((s, v) => s + v, 0) / best.length;
  return { level, count: best.length };
}
