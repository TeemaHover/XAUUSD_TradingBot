import { atr } from "../indicators/atr";
import { detectSwings } from "../market/swings";
import { AppConfig, Candle, DetectorResult } from "../types";

export interface FibLevel {
  ratio: number;
  price: number;
  label: string;
}

export interface FibResult extends DetectorResult {
  levels: FibLevel[];
  swingHigh: number;
  swingLow: number;
  nearestLevel?: FibLevel;
  inGoldenZone: boolean;
}

const FIB_RATIOS: { ratio: number; label: string }[] = [
  { ratio: 0.236, label: "23.6%" },
  { ratio: 0.382, label: "38.2%" },
  { ratio: 0.500, label: "50.0%" },
  { ratio: 0.618, label: "61.8%" },
  { ratio: 0.786, label: "78.6%" },
];

/**
 * Detects Fibonacci retracement levels from the most recent significant
 * swing high and low. Scores highest when price is in the golden zone
 * (50%-61.8%), which is the most respected pullback area on XAUUSD.
 *
 * Scoring:
 * - Golden zone (50-61.8%) = full score
 * - 38.2% or 78.6%         = 65% of score
 * - 23.6%                  = 40% of score
 * - Not near any level     = 0
 */
export function detectFib(candles: Candle[], config: AppConfig): FibResult {
  const empty: FibResult = {
    score: 0,
    reasons: ["Not enough candles for Fibonacci"],
    levels: [],
    swingHigh: 0,
    swingLow: 0,
    inGoldenZone: false,
  };

  const latest = candles.at(-1);
  if (!latest || candles.length < config.strategy.swingLookback * 2 + 1) return empty;

  const atrValues = atr(candles, config.strategy.atrLength);
  const atrValue = atrValues.at(-1) ?? 1;
  const proximity = atrValue * config.strategy.srProximityAtr;

  const swings = detectSwings(candles, config.strategy.swingLookback);
  if (swings.length < 2) return { ...empty, reasons: ["Not enough swings for Fibonacci"] };

  const highs = swings.filter((s) => s.type === "high");
  const lows  = swings.filter((s) => s.type === "low");
  if (highs.length === 0 || lows.length === 0) return { ...empty, reasons: ["Missing swing high or low"] };

  // Use the most recent significant swing high and low
  const swingHigh = Math.max(...highs.slice(-5).map((s) => s.price));
  const swingLow  = Math.min(...lows.slice(-5).map((s) => s.price));
  const range = swingHigh - swingLow;

  if (range < atrValue * 0.5) return { ...empty, reasons: ["Swing range too small for Fibonacci"] };

  const price = latest.close;

  // Retracement levels (measured from high down to low for bullish pullback)
  const levels: FibLevel[] = FIB_RATIOS.map(({ ratio, label }) => ({
    ratio,
    label,
    price: swingHigh - range * ratio,
  }));

  // Find nearest fib level
  let nearestLevel: FibLevel | undefined;
  let nearestDist = Infinity;
  for (const level of levels) {
    const dist = Math.abs(price - level.price);
    if (dist < nearestDist) {
      nearestDist = dist;
      nearestLevel = level;
    }
  }

  const atLevel = nearestDist <= proximity;
  const goldenHigh = swingHigh - range * 0.5;
  const goldenLow  = swingHigh - range * 0.618;
  const inGoldenZone = price >= goldenLow - proximity && price <= goldenHigh + proximity;

  const fullScore = config.scoring.fibConfirmation;
  let score = 0;
  let reason = "Not near any Fibonacci level";

  if (inGoldenZone) {
    score = fullScore;
    reason = `Price in Fibonacci golden zone (50-61.8%) at ${price.toFixed(2)}`;
  } else if (atLevel && nearestLevel) {
    if (nearestLevel.ratio === 0.382 || nearestLevel.ratio === 0.786) {
      score = Math.round(fullScore * 0.65);
    } else if (nearestLevel.ratio === 0.236) {
      score = Math.round(fullScore * 0.4);
    } else {
      score = Math.round(fullScore * 0.5);
    }
    reason = `Price near Fibonacci ${nearestLevel.label} at ${nearestLevel.price.toFixed(2)}`;
  }

  return {
    score,
    reasons: [reason],
    levels,
    swingHigh,
    swingLow,
    nearestLevel: atLevel ? nearestLevel : undefined,
    inGoldenZone,
  };
}
