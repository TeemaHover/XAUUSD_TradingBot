import { sessionFilter } from "../filters/sessionFilter";
import { newsFilter } from "../filters/newsFilter";
import { atr } from "../indicators/atr";
import { volatilityFilter } from "../filters/volatilityFilter";
import { lastSwing } from "../market/swings";
import { AppConfig, Candle, Direction, TradeSignal, WatchlistSignal } from "../types";
import { adaptiveRawScore, ScoreBreakdown } from "./adaptiveScoring";
import { hasConfirmationCandle } from "./confirmationCandles";
import { detectFvg } from "./fvgDetector";
import { detectLiquidity } from "./liquidityDetector";
import { detectMarketRegime } from "./marketRegimeDetector";
import { detectMarketStructure } from "./marketStructureDetector";
import { detectOrderBlock } from "./orderBlockDetector";
import { detectTrend } from "./trendDetector";
import { analyzeVolume } from "./volumeAnalyzer";
import { detectSR } from "./srDetector";
import { detectFib } from "./fibDetector";
import { detectDoublePattern } from "./doublePatternDetector";
import { detectRange, RangeResult } from "./rangeDetector";

export interface SignalDecision {
  status: "trade" | "watchlist" | "rejected";
  signal?: TradeSignal;
  watchlist?: WatchlistSignal;
  score: number;
  reasons: string[];
  finalDecision: FinalDecision;
}

export interface DirectionalContext {
  direction?: Direction;
  counterTrend: boolean;
  reason: string;
  /** "structure" = came from BOS/MSS/sweep; "range" = came from range boundary detection */
  source?: "structure" | "range";
}

export type FinalDecisionDirection = Direction | "none";
export type FinalDecisionTrendDirection = "bullish" | "bearish" | "sideways";
export type FinalDecisionSetupType = "trend-following" | "countertrend" | "range-trade" | "ai-trade" | "no-context";
export type FinalDecisionBlockedBy = "countertrend" | "lowVolume" | "lowScore" | "spread" | "news" | "riskLimit";
export type FinalDecisionAction = "trade" | "watchlist" | "reject";

export interface FinalDecision {
  direction: FinalDecisionDirection;
  trendDirection: FinalDecisionTrendDirection;
  setupType: FinalDecisionSetupType;
  score: number;
  requiredScore: number;
  allowed: boolean;
  blockedBy: FinalDecisionBlockedBy[];
  action: FinalDecisionAction;
}

function normalizeScore(score: number, maxScore: number): number {
  if (maxScore <= 0) return 0;
  return Math.min(100, Math.round((score / maxScore) * 100));
}

export function classifyDirectionalContext(
  currentTrend: ReturnType<typeof detectTrend>,
  higherTrend: ReturnType<typeof detectTrend>,
  liquidity: ReturnType<typeof detectLiquidity>,
  structure: ReturnType<typeof detectMarketStructure>
): DirectionalContext {
  const bullishBreak = liquidity.bullishSweep || structure.mss === "bullish" || structure.bos === "bullish";
  const bearishBreak = liquidity.bearishSweep || structure.mss === "bearish" || structure.bos === "bearish";

  if (bullishBreak && bearishBreak) {
    return { counterTrend: false, reason: "conflicting bullish and bearish lower-timeframe context", source: "structure" };
  }

  const lowerDirection: Direction | undefined = bullishBreak ? "long" : bearishBreak ? "short" : undefined;
  if (!lowerDirection) {
    return { counterTrend: false, reason: "no lower-timeframe BOS/MSS/sweep context" };
  }

  const anchorBias = higherTrend.bias !== "sideways" ? higherTrend.bias : currentTrend.bias;
  if (anchorBias === "sideways") {
    return { direction: lowerDirection, counterTrend: false, reason: `${lowerDirection} context with sideways trend anchor`, source: "structure" };
  }

  const aligned = (anchorBias === "bullish" && lowerDirection === "long")
    || (anchorBias === "bearish" && lowerDirection === "short");
  return {
    direction: lowerDirection,
    counterTrend: !aligned,
    source: "structure",
    reason: aligned
      ? `${lowerDirection} context aligned with ${anchorBias} trend`
      : `countertrend ${lowerDirection} context against ${anchorBias} trend`
  };
}

/**
 * Range trade context: when price is near a confirmed range boundary,
 * this provides directional context even without a structural BOS/MSS/sweep.
 * Long at support (bottom 15%), short at resistance (top 85%).
 */
export function classifyRangeContext(range: RangeResult, longThreshold = 15, shortThreshold = 85): DirectionalContext | undefined {
  if (!range.detected) return undefined;
  if (range.pricePosition <= longThreshold) {
    return {
      direction: "long",
      counterTrend: false,
      source: "range",
      reason: `Range trade [${range.timeframe}]: price at ${range.pricePosition.toFixed(0)}% -- buying support at ${range.low.toFixed(2)} (range ${range.low.toFixed(2)}-${range.high.toFixed(2)})`
    };
  }
  if (range.pricePosition >= shortThreshold) {
    return {
      direction: "short",
      counterTrend: false,
      source: "range",
      reason: `Range trade [${range.timeframe}]: price at ${range.pricePosition.toFixed(0)}% -- selling resistance at ${range.high.toFixed(2)} (range ${range.low.toFixed(2)}-${range.high.toFixed(2)})`
    };
  }
  return undefined;
}

export function counterTrendRejectionReason(
  context: DirectionalContext,
  score: number,
  config: AppConfig
): string | undefined {
  if (!context.counterTrend) return undefined;
  if (!config.strategy.allowCounterTrendTrades) return "Rejected: countertrend setup rejected";
  if (score < config.strategy.counterTrendMinScore) {
    return `Rejected: countertrend score ${score} below minimum ${config.strategy.counterTrendMinScore}`;
  }
  return undefined;
}

function trendDirection(
  currentTrend: ReturnType<typeof detectTrend> | undefined,
  higherTrend: ReturnType<typeof detectTrend> | undefined
): FinalDecisionTrendDirection {
  return higherTrend?.bias !== "sideways" ? higherTrend?.bias ?? "sideways" : currentTrend?.bias ?? "sideways";
}

export function buildFinalDecision(params: {
  direction?: Direction;
  currentTrend?: ReturnType<typeof detectTrend>;
  higherTrend?: ReturnType<typeof detectTrend>;
  counterTrend?: boolean;
  isRangeTrade?: boolean;
  score: number;
  config: AppConfig;
  action: FinalDecisionAction;
  blockedBy?: FinalDecisionBlockedBy[];
}): FinalDecision {
  const setupType: FinalDecisionSetupType = !params.direction
    ? "no-context"
    : params.isRangeTrade
      ? "range-trade"
      : params.counterTrend
        ? "countertrend"
        : "trend-following";
  const requiredScore = setupType === "countertrend"
    ? params.config.strategy.counterTrendMinScore
    : params.config.strategy.minScore;

  return {
    direction: params.direction ?? "none",
    trendDirection: trendDirection(params.currentTrend, params.higherTrend),
    setupType,
    score: params.score,
    requiredScore,
    allowed: params.action === "trade",
    blockedBy: params.blockedBy ?? [],
    action: params.action
  };
}

function stopLossForDirection(candles: Candle[], direction: Direction, entry: number, config: AppConfig): number {
  const liquidity = detectLiquidity(candles, config);
  const swing = lastSwing(liquidity.swings, direction === "long" ? "low" : "high");
  const atrValue = atr(candles, config.strategy.atrLength).at(-1) ?? 0;
  const buffer = Math.max(atrValue * config.risk.stopBufferAtr, config.risk.minStopDistance);

  let stop = swing?.price;

  if (stop === undefined) {
    const fallback = candles.slice(-Math.max(config.strategy.swingLookback, 3));
    stop = direction === "long"
      ? Math.min(...fallback.map((candle) => candle.low))
      : Math.max(...fallback.map((candle) => candle.high));
  }

  stop = direction === "long" ? stop - buffer : stop + buffer;

  if (direction === "long" && entry - stop < config.risk.minStopDistance) {
    stop = entry - config.risk.minStopDistance;
  }
  if (direction === "short" && stop - entry < config.risk.minStopDistance) {
    stop = entry + config.risk.minStopDistance;
  }

  return stop;
}

function takeProfits(entry: number, stopLoss: number, direction: Direction, config: AppConfig): number[] {
  const risk = Math.abs(entry - stopLoss);
  return config.tradeManagement.tpRMultiples.map((multiple) => (
    direction === "long" ? entry + risk * multiple : entry - risk * multiple
  ));
}

interface ZoneEntryPlan {
  entry: number;
  stopLoss: number;
  entryType: "market" | "limit";
  reasons: string[];
}

/**
 * Sniper/zone entry: instead of entering at market after a confirmation candle,
 * place the entry at the proximal edge of the nearest unmitigated order block or
 * unfilled FVG in the trade direction, with the stop behind the distal edge.
 * Entry closer to invalidation -> smaller stop -> more R for the same targets.
 * Returns undefined when no usable zone exists (zone mode takes no trade then).
 */
export function zoneEntryPlan(
  direction: Direction,
  latest: Candle,
  orderBlock: ReturnType<typeof detectOrderBlock>,
  fvg: ReturnType<typeof detectFvg>,
  atrValue: number,
  spread: number,
  config: AppConfig
): ZoneEntryPlan | undefined {
  const candidates: { low: number; high: number; label: string }[] = [];
  if (direction === "long") {
    if (orderBlock.bullish && !orderBlock.bullish.mitigated) {
      candidates.push({ low: orderBlock.bullish.low, high: orderBlock.bullish.high, label: "bullish OB" });
    }
    if (fvg.bullish && fvg.bullish.filledPercent < 50) {
      candidates.push({ low: fvg.bullish.low, high: fvg.bullish.high, label: "bullish FVG" });
    }
  } else {
    if (orderBlock.bearish && !orderBlock.bearish.mitigated) {
      candidates.push({ low: orderBlock.bearish.low, high: orderBlock.bearish.high, label: "bearish OB" });
    }
    if (fvg.bearish && fvg.bearish.filledPercent < 50) {
      candidates.push({ low: fvg.bearish.low, high: fvg.bearish.high, label: "bearish FVG" });
    }
  }

  // Zone must sit on the retracement side of price (below price for longs, above for shorts)
  const slack = atrValue * 0.1;
  const usable = candidates.filter((zone) => (
    direction === "long" ? zone.high <= latest.close + slack : zone.low >= latest.close - slack
  ));
  if (usable.length === 0) return undefined;

  // Prefer the zone whose proximal edge is closest to price (least retracement needed)
  const best = usable.sort((a, b) => (
    direction === "long" ? b.high - a.high : a.low - b.low
  ))[0];

  const proximal = direction === "long" ? best.high : best.low;
  const distal = direction === "long" ? best.low : best.high;
  const buffer = Math.max(atrValue * config.risk.stopBufferAtr, config.risk.minStopDistance * 0.5);

  const tolerance = (config.strategy.zoneTouchToleranceAtr ?? 0.15) * atrValue;
  const distance = direction === "long" ? latest.close - proximal : proximal - latest.close;
  const entryType: "market" | "limit" = distance > tolerance ? "limit" : "market";

  // If price is already at/inside the zone, enter at market price, not a worse level
  const rawEntry = entryType === "market"
    ? (direction === "long" ? Math.min(proximal, latest.close) : Math.max(proximal, latest.close))
    : proximal;
  const entry = direction === "long" ? rawEntry + spread / 2 : rawEntry - spread / 2;

  let stopLoss = direction === "long" ? distal - buffer : distal + buffer;
  if (direction === "long" && entry - stopLoss < config.risk.minStopDistance) {
    stopLoss = entry - config.risk.minStopDistance;
  }
  if (direction === "short" && stopLoss - entry < config.risk.minStopDistance) {
    stopLoss = entry + config.risk.minStopDistance;
  }

  return {
    entry,
    stopLoss,
    entryType,
    reasons: [
      `Zone entry (${best.label} ${best.low.toFixed(2)}-${best.high.toFixed(2)}): ${entryType} @ ${entry.toFixed(2)}, stop behind zone @ ${stopLoss.toFixed(2)}`
    ]
  };
}

/**
 * Structure-based final target: replace the last R-multiple TP with the nearest
 * opposing swing (front-run by 10% of risk) when that swing offers >= 1.5R.
 * Winners aim at real liquidity instead of an arbitrary 3R line.
 */
export function structureTakeProfits(
  entry: number,
  stopLoss: number,
  direction: Direction,
  swings: { price: number; type: "high" | "low" }[],
  config: AppConfig
): { tps: number[]; tpMode: "r" | "price"; reason?: string } {
  const base = takeProfits(entry, stopLoss, direction, config);
  if (!config.tradeManagement.structureTargets) return { tps: base, tpMode: "r" };

  const risk = Math.abs(entry - stopLoss);
  if (risk <= 0) return { tps: base, tpMode: "r" };

  const opposing = swings.filter((swing) => (
    direction === "long" ? swing.type === "high" && swing.price > entry : swing.type === "low" && swing.price < entry
  ));
  if (opposing.length === 0) return { tps: base, tpMode: "r" };

  const swingTarget = direction === "long"
    ? Math.min(...opposing.map((swing) => swing.price))
    : Math.max(...opposing.map((swing) => swing.price));
  // Front-run the liquidity slightly so the order fills before the crowd's
  const target = direction === "long" ? swingTarget - risk * 0.1 : swingTarget + risk * 0.1;
  const targetR = Math.abs(target - entry) / risk;
  if (targetR < 1.5) return { tps: base, tpMode: "r" };

  const tps = [...base];
  tps[tps.length - 1] = target;
  // Keep TP ordering monotonic if the structure target is closer than TP2
  const sorted = direction === "long" ? [...tps].sort((a, b) => a - b) : [...tps].sort((a, b) => b - a);
  return {
    tps: sorted,
    tpMode: "price",
    reason: `Structure TP: final target ${target.toFixed(2)} (${targetR.toFixed(1)}R) at opposing swing`
  };
}

/**
 * For range trades, TP targets are the range mid (TP1) and opposite boundary (TP2).
 * This replaces R-multiple TPs so the bot takes profit at the range ceiling/floor.
 */
function rangeTakeProfits(direction: Direction, range: RangeResult): number[] {
  return direction === "long"
    ? [range.mid, range.high]
    : [range.mid, range.low];
}

/**
 * Dumb mode: S/R only.
 *
 * Logic:
 *   - Near SUPPORT    => LONG
 *   - Near RESISTANCE => SHORT
 *
 * No BOS, no FVG, no trend, no session, no news.
 * Stop loss is placed just beyond the S/R zone boundary.
 * Take profits use the standard R-multiple system.
 */
function calculateDumbSignal(
  entryCandles: Candle[],
  spread: number,
  config: AppConfig
): SignalDecision {
  const latest = entryCandles.at(-1);
  if (!latest) {
    return {
      status: "rejected",
      score: 0,
      reasons: ["No candles provided"],
      finalDecision: buildFinalDecision({ score: 0, config, action: "reject", blockedBy: ["lowScore"] })
    };
  }

  const sr       = detectSR(entryCandles, config);
  const atrValue = atr(entryCandles, config.strategy.atrLength).at(-1) ?? 1;

  let direction: Direction | undefined;
  const reasons: string[] = ["[Dumb mode] S/R only -- no BOS, no FVG, no trend filter"];

  if (sr.nearSupport && sr.nearResistance) {
    return {
      status: "rejected",
      score: 0,
      reasons: [...reasons, "Price squeezed between support and resistance -- no edge at midpoint"],
      finalDecision: buildFinalDecision({ score: 0, config, action: "reject" })
    };
  }

  if (sr.nearSupport) {
    direction = "long";
    const zone = sr.supportZones[0];
    reasons.push(
      "Near support zone" + (zone ? " " + zone.low.toFixed(2) + " - " + zone.high.toFixed(2) + " (str:" + zone.strength + ")" : ""),
      "Buying at S/R floor"
    );
  } else if (sr.nearResistance) {
    direction = "short";
    const zone = sr.resistanceZones[0];
    reasons.push(
      "Near resistance zone" + (zone ? " " + zone.low.toFixed(2) + " - " + zone.high.toFixed(2) + " (str:" + zone.strength + ")" : ""),
      "Selling at S/R ceiling"
    );
  } else {
    return {
      status: "rejected",
      score: 0,
      reasons: [...reasons, "Waiting: not near any S/R level -- price in open space"],
      finalDecision: buildFinalDecision({ score: 0, config, action: "reject" })
    };
  }

  const entry = direction === "long" ? latest.close + spread / 2 : latest.close - spread / 2;

  // Stop loss: just beyond the S/R zone boundary
  const buffer = atrValue * config.risk.stopBufferAtr;
  const srZones = direction === "long" ? sr.supportZones : sr.resistanceZones;
  const srBoundary = srZones.length > 0
    ? (direction === "long"
        ? Math.min(...srZones.map((z) => z.low))
        : Math.max(...srZones.map((z) => z.high)))
    : (direction === "long" ? entry - atrValue : entry + atrValue);

  const stopLoss = direction === "long"
    ? srBoundary - buffer
    : srBoundary + buffer;

  const risk = Math.abs(entry - stopLoss);
  if (risk < config.risk.minStopDistance) {
    return {
      status: "rejected",
      score: 0,
      reasons: [...reasons, "Stop distance " + risk.toFixed(2) + " below minimum"],
      finalDecision: buildFinalDecision({ score: 0, config, action: "reject", blockedBy: ["riskLimit"] })
    };
  }

  const tps = takeProfits(entry, stopLoss, direction, config);

  return {
    status: "trade",
    score: 100,
    reasons,
    finalDecision: buildFinalDecision({ direction, score: 100, config, action: "trade" }),
    signal: {
      symbol: config.symbol,
      direction,
      entry,
      stopLoss,
      takeProfits: tps,
      score: 100,
      reasons,
      timestamp: latest.time
    }
  };
}


export function calculateSignal(
  entryCandles: Candle[],
  trendCandles: Candle[],
  higherTrendCandles: Candle[],
  spread: number,
  config: AppConfig
): SignalDecision {
  const latest = entryCandles.at(-1);
  if (!latest) {
    return {
      status: "rejected",
      score: 0,
      reasons: ["No candles provided"],
      finalDecision: buildFinalDecision({
        score: 0,
        config,
        action: "reject",
        blockedBy: ["lowScore"]
      })
    };
  }

  // Dumb mode bypasses all filters — only S/R + BOS + FVG
  if (config.strategy.dumbMode) {
    return calculateDumbSignal(entryCandles, spread, config);
  }

  const currentTrend = detectTrend(trendCandles, config);
  const higherTrend = detectTrend(higherTrendCandles, config);
  const liquidity = detectLiquidity(entryCandles, config);
  const structure = detectMarketStructure(entryCandles, liquidity, config);
  const orderBlock = detectOrderBlock(entryCandles, config);
  const fvg = detectFvg(entryCandles, config);
  const volume = analyzeVolume(entryCandles, config);
  const session = sessionFilter(latest.time, config);
  const volatility = volatilityFilter(entryCandles, spread, config);
  const regime = detectMarketRegime(entryCandles, config);
  const news = newsFilter(latest.time, config.symbol, config);
  const sr = detectSR(entryCandles, config);
  const fib = detectFib(entryCandles, config);
  const dp = detectDoublePattern(entryCandles, config);
  const range = detectRange(entryCandles, config, config.timeframes.entry);

  // Primary directional context from market structure
  const structureContext = classifyDirectionalContext(currentTrend, higherTrend, liquidity, structure);

  // Fall back to range context when no structural direction detected
  const rangeContext = !structureContext.direction
    ? classifyRangeContext(range, config.strategy.rangeLongThreshold, config.strategy.rangeShortThreshold)
    : undefined;
  const directionalContext = structureContext.direction ? structureContext : (rangeContext ?? structureContext);
  const direction = directionalContext.direction;
  const isRangeTrade = directionalContext.source === "range";

  const trendScore = Math.max(currentTrend.score, higherTrend.score);
  const breakdown: ScoreBreakdown = {
    trendAlignment: { score: trendScore, reasons: [] },
    liquiditySweep: liquidity,
    marketStructure: structure,
    orderBlock,
    fairValueGap: fvg,
    volumeConfirmation: volume,
    sessionAllowed: session,
    volatilityValid: volatility,
    srConfirmation: sr,
    fibConfirmation: fib,
    doublePattern: dp
  };
  const adaptive = adaptiveRawScore(breakdown, regime.regime, config);

  const score = normalizeScore(adaptive.rawScore, adaptive.maxScore);
  const reasons = [
    ...regime.reasons,
    ...higherTrend.reasons.map((reason) => `HTF: ${reason}`),
    ...currentTrend.reasons,
    ...liquidity.reasons,
    ...structure.reasons,
    ...orderBlock.reasons,
    ...fvg.reasons,
    ...volume.reasons,
    ...session.reasons,
    ...volatility.reasons,
    ...sr.reasons,
    ...fib.reasons,
    ...dp.reasons,
    ...(range.detected ? [`Range detected: ${range.summary}`] : []),
    ...news.reasons,
    directionalContext.reason,
    ...adaptive.reasons
  ];

  if (news.blocked) {
    return {
      status: "rejected",
      score,
      reasons: [...reasons, "Rejected: news blackout window"],
      finalDecision: buildFinalDecision({
        direction,
        currentTrend,
        higherTrend,
        counterTrend: directionalContext.counterTrend,
        isRangeTrade,
        score,
        config,
        action: "reject",
        blockedBy: ["news"]
      })
    };
  }

  if (!direction) {
    return {
      status: "rejected",
      score,
      reasons: [...reasons, "Rejected: no directional context"],
      finalDecision: buildFinalDecision({
        currentTrend,
        higherTrend,
        counterTrend: directionalContext.counterTrend,
        score,
        config,
        action: "reject"
      })
    };
  }

  const counterTrendRejection = counterTrendRejectionReason(directionalContext, score, config);
  if (counterTrendRejection) {
    return {
      status: "rejected",
      score,
      reasons: [...reasons, counterTrendRejection],
      finalDecision: buildFinalDecision({
        direction,
        currentTrend,
        higherTrend,
        counterTrend: directionalContext.counterTrend,
        isRangeTrade,
        score,
        config,
        action: "reject",
        blockedBy: ["countertrend"]
      })
    };
  }

  const zoneMode = config.strategy.entryMode === "zone" && !isRangeTrade;

  // Zone mode: the limit price at the zone edge IS the trigger — no confirmation candle needed
  if (!zoneMode && config.strategy.requireConfirmationCandle && !hasConfirmationCandle(entryCandles, direction)) {
    return {
      status: "rejected",
      score,
      reasons: [...reasons, "Rejected: no confirmation candle"],
      finalDecision: buildFinalDecision({
        direction,
        currentTrend,
        higherTrend,
        counterTrend: directionalContext.counterTrend,
        isRangeTrade,
        score,
        config,
        action: "reject"
      })
    };
  }

  if (score < config.strategy.watchlistScore) {
    return {
      status: "rejected",
      score,
      reasons: [...reasons, `Rejected: score ${score} below watchlist threshold`],
      finalDecision: buildFinalDecision({
        direction,
        currentTrend,
        higherTrend,
        counterTrend: directionalContext.counterTrend,
        isRangeTrade,
        score,
        config,
        action: "reject",
        blockedBy: ["lowScore"]
      })
    };
  }

  if (score < config.strategy.minScore) {
    return {
      status: "watchlist",
      score,
      reasons,
      finalDecision: buildFinalDecision({
        direction,
        currentTrend,
        higherTrend,
        counterTrend: directionalContext.counterTrend,
        isRangeTrade,
        score,
        config,
        action: "watchlist",
        blockedBy: ["lowScore"]
      }),
      watchlist: {
        symbol: config.symbol,
        direction,
        score,
        reasons,
        timestamp: latest.time,
        regime: regime.regime
      }
    };
  }

  let entry: number;
  let stopLoss: number;
  let entryType: "market" | "limit" = "market";

  if (zoneMode) {
    const atrValue = atr(entryCandles, config.strategy.atrLength).at(-1) ?? 1;
    const plan = zoneEntryPlan(direction, latest, orderBlock, fvg, atrValue, spread, config);
    if (!plan) {
      return {
        status: "rejected",
        score,
        reasons: [...reasons, "Rejected: zone mode — no unmitigated OB / unfilled FVG to anchor a limit entry"],
        finalDecision: buildFinalDecision({
          direction,
          currentTrend,
          higherTrend,
          counterTrend: directionalContext.counterTrend,
          isRangeTrade,
          score,
          config,
          action: "reject"
        })
      };
    }
    entry = plan.entry;
    stopLoss = plan.stopLoss;
    entryType = plan.entryType;
    reasons.push(...plan.reasons);
  } else {
    entry = direction === "long" ? latest.close + spread / 2 : latest.close - spread / 2;
    stopLoss = stopLossForDirection(entryCandles, direction, entry, config);
  }

  const risk = Math.abs(entry - stopLoss);
  if (risk < config.risk.minStopDistance) {
    return {
      status: "rejected",
      score,
      reasons: [...reasons, `Rejected: stop distance ${risk.toFixed(2)} below minimum`],
      finalDecision: buildFinalDecision({
        direction,
        currentTrend,
        higherTrend,
        counterTrend: directionalContext.counterTrend,
        isRangeTrade,
        score,
        config,
        action: "reject",
        blockedBy: ["riskLimit"]
      })
    };
  }

  // Range trades target the opposite side of the range; trend trades use
  // R-multiples, optionally upgrading the final TP to a structure target.
  let tps: number[];
  let tpMode: "r" | "price" = "r";
  if (isRangeTrade && range.detected) {
    tps = rangeTakeProfits(direction, range);
    tpMode = "price";
  } else {
    const structured = structureTakeProfits(entry, stopLoss, direction, liquidity.swings, config);
    tps = structured.tps;
    tpMode = structured.tpMode;
    if (structured.reason) reasons.push(structured.reason);
  }

  return {
    status: "trade",
    score,
    reasons,
    finalDecision: buildFinalDecision({
      direction,
      currentTrend,
      higherTrend,
      counterTrend: directionalContext.counterTrend,
      isRangeTrade,
      score,
      config,
      action: "trade"
    }),
    signal: {
      symbol: config.symbol,
      direction,
      entry,
      stopLoss,
      takeProfits: tps,
      score,
      reasons,
      timestamp: latest.time,
      regime: regime.regime,
      entryType,
      tpMode
    }
  };
}
