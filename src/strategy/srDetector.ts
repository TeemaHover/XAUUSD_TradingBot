import { atr } from "../indicators/atr";
import { detectSwings } from "../market/swings";
import { AppConfig, Candle, DetectorResult, Zone } from "../types";

export interface SRResult extends DetectorResult {
  supportZones: Zone[];
  resistanceZones: Zone[];
  nearSupport: boolean;
  nearResistance: boolean;
  nearRoundNumber: boolean;
}

/**
 * Groups a list of prices into zones by clustering prices within `tolerance` of each other.
 * Returns zones sorted from weakest to strongest.
 */
function clusterIntoZones(prices: number[], tolerance: number): Zone[] {
  if (prices.length === 0) return [];
  const sorted = [...prices].sort((a, b) => a - b);
  const clusters: number[][] = [[sorted[0]]];

  for (let i = 1; i < sorted.length; i++) {
    const last = clusters[clusters.length - 1];
    const avg = last.reduce((sum, v) => sum + v, 0) / last.length;
    if (sorted[i] - avg <= tolerance) {
      last.push(sorted[i]);
    } else {
      clusters.push([sorted[i]]);
    }
  }

  return clusters.map((cluster) => {
    const avg = cluster.reduce((sum, v) => sum + v, 0) / cluster.length;
    return {
      low: avg - tolerance * 0.5,
      high: avg + tolerance * 0.5,
      // strength: 25 per touch, capped at 100 (4+ touches = maximum strength)
      strength: Math.min(100, cluster.length * 25)
    };
  });
}

function isNearZone(price: number, zones: Zone[], proximity: number): boolean {
  return zones.some((zone) => price >= zone.low - proximity && price <= zone.high + proximity);
}

function nearestRoundNumber(price: number, step: number): number {
  return Math.round(price / step) * step;
}

/**
 * Detects key support and resistance zones from clustered swing points.
 * Also checks for proximity to XAUUSD psychological round numbers ($50 increments).
 *
 * Scoring:
 * - Near a zone + near a round number (confluence) → full score
 * - Near a zone only → 60% of score
 * - Neither → 0
 */
export function detectSR(candles: Candle[], config: AppConfig): SRResult {
  const empty: SRResult = {
    score: 0,
    reasons: ["Not enough candles for S/R"],
    supportZones: [],
    resistanceZones: [],
    nearSupport: false,
    nearResistance: false,
    nearRoundNumber: false
  };

  const latest = candles.at(-1);
  if (!latest || candles.length < config.strategy.swingLookback * 2 + 1) return empty;

  const atrValues = atr(candles, config.strategy.atrLength);
  const atrValue = atrValues.at(-1) ?? 1;
  const tolerance = atrValue * config.strategy.srZoneToleranceAtr;
  const proximity = atrValue * config.strategy.srProximityAtr;

  const swings = detectSwings(candles, config.strategy.swingLookback);
  // Use last 20 swings so distant history doesn't pollute current zones
  const recentHighs = swings.filter((s) => s.type === "high").slice(-20).map((s) => s.price);
  const recentLows  = swings.filter((s) => s.type === "low").slice(-20).map((s) => s.price);

  const resistanceZones = clusterIntoZones(recentHighs, tolerance);
  const supportZones    = clusterIntoZones(recentLows,  tolerance);

  const price = latest.close;
  const nearSupport    = isNearZone(price, supportZones,    proximity);
  const nearResistance = isNearZone(price, resistanceZones, proximity);

  // XAUUSD psychological levels: $50 increments (2000, 2050, 2100 …)
  const roundStep   = 50;
  const nearest     = nearestRoundNumber(price, roundStep);
  const nearRoundNumber = Math.abs(price - nearest) <= proximity;

  const atLevel   = nearSupport || nearResistance || nearRoundNumber;
  const confluence = (nearSupport || nearResistance) && nearRoundNumber;

  const fullScore = config.scoring.srConfirmation;
  const score = confluence
    ? fullScore
    : atLevel
      ? Math.round(fullScore * 0.6)
      : 0;

  const reasons: string[] = [];
  if (nearSupport)           reasons.push("Price near support zone");
  if (nearResistance)        reasons.push("Price near resistance zone");
  if (nearRoundNumber)       reasons.push(`Price near round number ${nearest}`);
  if (confluence)            reasons.push("S/R + round number confluence");
  if (!atLevel)              reasons.push("No S/R level nearby");

  return { score, reasons, supportZones, resistanceZones, nearSupport, nearResistance, nearRoundNumber };
}
