import { atr } from "../indicators/atr";
import { ema } from "../indicators/ema";
import { detectSwings, lastSwing } from "../market/swings";
import { AppConfig, Candle, Direction, Zone } from "../types";
import { detectLiquidity } from "./liquidityDetector";
import { detectMarketStructure } from "./marketStructureDetector";
import { detectSR } from "./srDetector";
import { hasConfirmationCandle } from "./confirmationCandles";
import { FinalDecisionTrendDirection, SignalDecision } from "./signalEngine";

/**
 * Top-down multi-timeframe cascade:
 *
 *   1D  — directional bias (EMA + swing structure). No bias -> no trade.
 *   4H  — where: price must be AT a higher-timeframe S/R zone (1D + 4H zones).
 *   1H  — setup: rejection at the zone (liquidity sweep or structure break).
 *   5m  — trigger: BOS / confirmation candle in the bias direction.
 *
 * SL: behind the last 5m swing (buffered, ATR-capped).
 * TP: the NEXT opposing HTF zone — and if that target is closer than
 * `minRR` R-multiples, the trade is skipped entirely. No small-payoff trades.
 */

export interface MtfSettings {
  dailyEmaLength: number;
  zoneProximityAtr: number;
  stopBufferAtr: number;
  maxStopAtr: number;
  minRR: number;
  maxHoldBars: number;
  cooldownBars: number;
}

export function mtfSettings(config: AppConfig): MtfSettings {
  const c = config.mtf ?? {};
  return {
    dailyEmaLength: c.dailyEmaLength ?? 50,
    zoneProximityAtr: c.zoneProximityAtr ?? 1.0,
    stopBufferAtr: c.stopBufferAtr ?? 0.5,
    maxStopAtr: c.maxStopAtr ?? 3.0,
    minRR: c.minRR ?? 3,
    maxHoldBars: c.maxHoldBars ?? 2016,
    cooldownBars: c.cooldownBars ?? 288
  };
}

export type MtfBias = "bullish" | "bearish" | "sideways";

/** Daily bias: close vs EMA plus last-two-swings structure (like trendDetector, parametric length). */
export function mtfBias(daily: Candle[], emaLength: number, swingLookback: number): { bias: MtfBias; reason: string } {
  if (daily.length < emaLength + swingLookback * 2) {
    return { bias: "sideways", reason: `not enough daily candles (${daily.length} < ${emaLength + swingLookback * 2})` };
  }
  const closes = daily.map((c) => c.close);
  const emaValues = ema(closes, emaLength);
  const latest = daily.at(-1)!;
  const latestEma = emaValues.at(-1)!;
  const swings = detectSwings(daily, swingLookback);
  const highs = swings.filter((s) => s.type === "high").slice(-2);
  const lows = swings.filter((s) => s.type === "low").slice(-2);
  const higherHighs = highs.length === 2 && highs[1].price > highs[0].price;
  const higherLows = lows.length === 2 && lows[1].price > lows[0].price;
  const lowerHighs = highs.length === 2 && highs[1].price < highs[0].price;
  const lowerLows = lows.length === 2 && lows[1].price < lows[0].price;

  if (latest.close > latestEma && !(lowerHighs && lowerLows)) {
    return { bias: "bullish", reason: `close above EMA${emaLength}${higherHighs && higherLows ? " with HH/HL" : ""}` };
  }
  if (latest.close < latestEma && !(higherHighs && higherLows)) {
    return { bias: "bearish", reason: `close below EMA${emaLength}${lowerHighs && lowerLows ? " with LH/LL" : ""}` };
  }
  return { bias: "sideways", reason: "price and structure disagree" };
}

export interface HtfZones {
  supports: Zone[];
  resistances: Zone[];
}

/** Merge S/R zones from 1D and 4H. Daily zones get a strength bonus — they matter more. */
export function collectHtfZones(daily: Candle[], h4: Candle[], config: AppConfig): HtfZones {
  const zonesD = detectSR(daily, config);
  const zones4 = detectSR(h4, config);
  const boost = (z: Zone): Zone => ({ ...z, strength: Math.min(100, z.strength + 25) });
  return {
    supports: [...zonesD.supportZones.map(boost), ...zones4.supportZones],
    resistances: [...zonesD.resistanceZones.map(boost), ...zones4.resistanceZones]
  };
}

/** The zone price is currently sitting at (or slightly piercing), if any. */
export function activeZoneAt(price: number, zones: Zone[], proximity: number): Zone | undefined {
  const touching = zones.filter((z) => price >= z.low - proximity * 0.5 && price <= z.high + proximity);
  if (touching.length === 0) return undefined;
  return touching.sort((a, b) => b.strength - a.strength)[0];
}

/** Nearest opposing zone beyond the entry — the trade's target. */
export function nearestTargetBeyond(zones: Zone[], direction: Direction, entry: number): { price: number; zone: Zone } | undefined {
  if (direction === "long") {
    const above = zones.filter((z) => z.low > entry).sort((a, b) => a.low - b.low);
    return above.length > 0 ? { price: above[0].low, zone: above[0] } : undefined;
  }
  const below = zones.filter((z) => z.high < entry).sort((a, b) => b.high - a.high);
  return below.length > 0 ? { price: below[0].high, zone: below[0] } : undefined;
}

function reject(
  reasons: string[],
  score: number,
  direction: Direction | undefined,
  trendDirection: FinalDecisionTrendDirection,
  blocker: string
): SignalDecision {
  return {
    status: "rejected",
    score,
    reasons: [...reasons, blocker],
    finalDecision: {
      direction: direction ?? "none",
      trendDirection,
      setupType: direction ? "trend-following" : "no-context",
      score,
      requiredScore: 0,
      allowed: false,
      blockedBy: [],
      action: "reject"
    }
  };
}

export function calculateMtfSignal(
  entry5m: Candle[],
  h1: Candle[],
  h4: Candle[],
  daily: Candle[],
  spread: number,
  config: AppConfig
): SignalDecision {
  const settings = mtfSettings(config);
  const reasons: string[] = ["[MTF] top-down cascade (1D bias -> HTF zone -> 1H setup -> 5m trigger)"];
  const latest = entry5m.at(-1);
  if (!latest || entry5m.length < 50 || h1.length < 50 || h4.length < 30) {
    return reject(reasons, 0, undefined, "sideways", "Waiting: not enough candles on 5m/1h/4h yet");
  }

  // ── Layer 1: daily bias ─────────────────────────────────────────────
  const { bias, reason: biasReason } = mtfBias(daily, settings.dailyEmaLength, config.strategy.swingLookback);
  reasons.push(`1D bias: ${bias.toUpperCase()} (${biasReason})`);
  if (bias === "sideways") {
    return reject(reasons, 0, undefined, "sideways", "Waiting: no daily bias — stand aside");
  }
  const direction: Direction = bias === "bullish" ? "long" : "short";
  const trendDirection: FinalDecisionTrendDirection = bias;

  // ── Layer 2: price must be AT a HTF zone in our favor ──────────────
  const atr4hValue = atr(h4, config.strategy.atrLength).at(-1) ?? 0;
  const atr5mValue = atr(entry5m, config.strategy.atrLength).at(-1) ?? 0;
  if (atr4hValue <= 0 || atr5mValue <= 0) {
    return reject(reasons, 0, direction, trendDirection, "Waiting: ATR not ready");
  }
  const zones = collectHtfZones(daily, h4, config);
  const entryZones = direction === "long" ? zones.supports : zones.resistances;
  const proximity = atr4hValue * settings.zoneProximityAtr;
  const price = latest.close;
  const zone = activeZoneAt(price, entryZones, proximity);
  if (!zone) {
    return reject(reasons, 10, direction, trendDirection,
      `Waiting: price ${price.toFixed(2)} not at any HTF ${direction === "long" ? "support" : "resistance"} zone`);
  }
  reasons.push(`At HTF ${direction === "long" ? "support" : "resistance"} ${zone.low.toFixed(2)}-${zone.high.toFixed(2)} (strength ${zone.strength})`);

  // ── Layer 3: 1H rejection at the zone ──────────────────────────────
  const liq1h = detectLiquidity(h1, config);
  const struct1h = detectMarketStructure(h1, liq1h, config);
  const setup1h = direction === "long"
    ? liq1h.bullishSweep || struct1h.bos === "bullish" || struct1h.mss === "bullish"
    : liq1h.bearishSweep || struct1h.bos === "bearish" || struct1h.mss === "bearish";
  if (!setup1h) {
    return reject(reasons, 35, direction, trendDirection,
      "Waiting: at HTF zone but no 1h rejection yet (need sweep or structure break)");
  }
  reasons.push(`1H setup: ${[
    liq1h.bullishSweep || liq1h.bearishSweep ? "liquidity sweep" : "",
    struct1h.bos !== "none" ? `BOS ${struct1h.bos}` : "",
    struct1h.mss !== "none" ? `MSS ${struct1h.mss}` : ""
  ].filter(Boolean).join(", ")}`);

  // ── Layer 4: 5m trigger ────────────────────────────────────────────
  const liq5m = detectLiquidity(entry5m, config);
  const struct5m = detectMarketStructure(entry5m, liq5m, config);
  const structTrigger = direction === "long"
    ? struct5m.bos === "bullish" || struct5m.mss === "bullish"
    : struct5m.bos === "bearish" || struct5m.mss === "bearish";
  const candleTrigger = hasConfirmationCandle(entry5m, direction);
  if (!structTrigger && !candleTrigger) {
    return reject(reasons, 55, direction, trendDirection,
      "Waiting: 1h setup ready — need a 5m BOS or confirmation candle to trigger");
  }
  reasons.push(`5m trigger: ${structTrigger ? "structure break" : "confirmation candle"}`);

  // ── Entry / SL / TP ────────────────────────────────────────────────
  const entry = direction === "long" ? price + spread / 2 : price - spread / 2;
  const swings5m = detectSwings(entry5m, config.strategy.swingLookback);
  const anchor = direction === "long" ? lastSwing(swings5m, "low") : lastSwing(swings5m, "high");
  let stopLoss = direction === "long"
    ? (anchor ? anchor.price : zone.low) - atr5mValue * settings.stopBufferAtr
    : (anchor ? anchor.price : zone.high) + atr5mValue * settings.stopBufferAtr;
  const maxStop = atr5mValue * settings.maxStopAtr;
  if (Math.abs(entry - stopLoss) > maxStop) {
    stopLoss = direction === "long" ? entry - maxStop : entry + maxStop;
    reasons.push(`SL capped at ${settings.maxStopAtr}x 5m ATR`);
  }
  const risk = Math.abs(entry - stopLoss);
  if (risk < config.risk.minStopDistance) {
    return reject(reasons, 55, direction, trendDirection,
      `Stop distance ${risk.toFixed(2)} below minimum ${config.risk.minStopDistance}`);
  }

  const opposing = direction === "long" ? zones.resistances : zones.supports;
  const target = nearestTargetBeyond(opposing, direction, entry);
  if (!target) {
    return reject(reasons, 55, direction, trendDirection,
      "Waiting: no opposing HTF zone to target — nothing to aim at");
  }
  const rewardRisk = Math.abs(target.price - entry) / risk;
  if (rewardRisk < settings.minRR) {
    return reject(reasons, 55, direction, trendDirection,
      `Target zone only ${rewardRisk.toFixed(1)}R away (need ${settings.minRR}R) — skip small-payoff trade`);
  }

  const score = Math.min(100, Math.round(40 + zone.strength * 0.3 + Math.min(rewardRisk, 10) * 3));
  reasons.push(
    `Entry ${entry.toFixed(2)} | SL ${stopLoss.toFixed(2)} (risk ${risk.toFixed(2)})`,
    `TP ${target.price.toFixed(2)} at opposing zone (${rewardRisk.toFixed(1)}R)`
  );

  return {
    status: "trade",
    score,
    reasons,
    finalDecision: {
      direction,
      trendDirection,
      setupType: "trend-following",
      score,
      requiredScore: 0,
      allowed: true,
      blockedBy: [],
      action: "trade"
    },
    signal: {
      symbol: config.symbol,
      direction,
      entry,
      stopLoss,
      takeProfits: [target.price],
      tpMode: "price",
      entryType: "market",
      score,
      reasons,
      timestamp: latest.time
    }
  };
}
