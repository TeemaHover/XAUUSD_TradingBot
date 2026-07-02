import { atr } from "../indicators/atr";
import { detectSwings } from "../market/swings";
import { AppConfig, Candle, DetectorResult } from "../types";

export interface DoublePatternResult extends DetectorResult {
  doubleTop: boolean;
  doubleBottom: boolean;
  neckline?: number;
  patternHigh?: number;
  patternLow?: number;
}

/**
 * Detects double top and double bottom chart patterns using swing points.
 *
 * Double top:  two swing highs at similar price (+/- ATR tolerance),
 *              with a swing low (neckline) between them.
 *              Valid only while price is BELOW the double top level.
 *              If price closes above the pattern high, it is invalidated.
 *
 * Double bottom: two swing lows at similar price (+/- ATR tolerance),
 *                with a swing high (neckline) between them.
 *                Valid only while price is ABOVE the double bottom level.
 *                If price closes below the pattern low, it is invalidated.
 *
 * Scoring:
 * - Confirmed (price near or past neckline, within valid range) = full score
 * - Forming (pattern identified, price not yet at neckline)      = 50% score
 */
export function detectDoublePattern(candles: Candle[], config: AppConfig): DoublePatternResult {
  const empty: DoublePatternResult = {
    score: 0,
    reasons: ["No double top or double bottom detected"],
    doubleTop: false,
    doubleBottom: false,
  };

  const latest = candles.at(-1);
  if (!latest || candles.length < config.strategy.swingLookback * 4 + 1) {
    return { ...empty, reasons: ["Not enough candles for double pattern"] };
  }

  const atrValues = atr(candles, config.strategy.atrLength);
  const atrValue = atrValues.at(-1) ?? 1;
  const tolerance = atrValue * config.strategy.srZoneToleranceAtr;
  const proximity = atrValue * config.strategy.srProximityAtr;

  const swings = detectSwings(candles, config.strategy.swingLookback);
  const price = latest.close;
  const fullScore = config.scoring.doublePattern;

  // --- Double Top ---
  const highs = swings.filter((s) => s.type === "high").slice(-8);
  for (let i = 0; i < highs.length - 1; i++) {
    for (let j = i + 1; j < highs.length; j++) {
      const h1 = highs[i];
      const h2 = highs[j];
      if (Math.abs(h1.price - h2.price) > tolerance) continue;

      const between = swings.filter(
        (s) => s.type === "low" && s.index > h1.index && s.index < h2.index
      );
      if (between.length === 0) continue;

      const patternHigh = Math.max(h1.price, h2.price);
      const neckline = Math.min(...between.map((s) => s.price));

      // Invalidated: price broke above the double top
      if (price > patternHigh) continue;

      const atNeckline = Math.abs(price - neckline) <= proximity || price < neckline;
      const score = atNeckline ? fullScore : Math.round(fullScore * 0.5);
      const confirmed = atNeckline ? " (confirmed)" : " (forming)";

      return {
        score,
        reasons: [`Double top at ${patternHigh.toFixed(2)}, neckline ${neckline.toFixed(2)}${confirmed}`],
        doubleTop: true,
        doubleBottom: false,
        neckline,
        patternHigh,
      };
    }
  }

  // --- Double Bottom ---
  const lows = swings.filter((s) => s.type === "low").slice(-8);
  for (let i = 0; i < lows.length - 1; i++) {
    for (let j = i + 1; j < lows.length; j++) {
      const l1 = lows[i];
      const l2 = lows[j];
      if (Math.abs(l1.price - l2.price) > tolerance) continue;

      const between = swings.filter(
        (s) => s.type === "high" && s.index > l1.index && s.index < l2.index
      );
      if (between.length === 0) continue;

      const patternLow = Math.min(l1.price, l2.price);
      const neckline = Math.max(...between.map((s) => s.price));

      // Invalidated: price broke below the double bottom
      if (price < patternLow) continue;

      const atNeckline = Math.abs(price - neckline) <= proximity || price > neckline;
      const score = atNeckline ? fullScore : Math.round(fullScore * 0.5);
      const confirmed = atNeckline ? " (confirmed)" : " (forming)";

      return {
        score,
        reasons: [`Double bottom at ${patternLow.toFixed(2)}, neckline ${neckline.toFixed(2)}${confirmed}`],
        doubleTop: false,
        doubleBottom: true,
        neckline,
        patternLow,
      };
    }
  }

  return empty;
}
