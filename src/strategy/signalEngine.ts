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
}

export type FinalDecisionDirection = Direction | "none";
export type FinalDecisionTrendDirection = "bullish" | "bearish" | "sideways";
export type FinalDecisionSetupType = "trend-following" | "countertrend" | "no-context";
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
    return { counterTrend: false, reason: "conflicting bullish and bearish lower-timeframe context" };
  }

  const lowerDirection: Direction | undefined = bullishBreak ? "long" : bearishBreak ? "short" : undefined;
  if (!lowerDirection) {
    return { counterTrend: false, reason: "no lower-timeframe BOS/MSS/sweep context" };
  }

  const anchorBias = higherTrend.bias !== "sideways" ? higherTrend.bias : currentTrend.bias;
  if (anchorBias === "sideways") {
    return { direction: lowerDirection, counterTrend: false, reason: `${lowerDirection} context with sideways trend anchor` };
  }

  const aligned = (anchorBias === "bullish" && lowerDirection === "long")
    || (anchorBias === "bearish" && lowerDirection === "short");
  return {
    direction: lowerDirection,
    counterTrend: !aligned,
    reason: aligned
      ? `${lowerDirection} context aligned with ${anchorBias} trend`
      : `countertrend ${lowerDirection} context against ${anchorBias} trend`
  };
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
  score: number;
  config: AppConfig;
  action: FinalDecisionAction;
  blockedBy?: FinalDecisionBlockedBy[];
}): FinalDecision {
  const setupType: FinalDecisionSetupType = !params.direction
    ? "no-context"
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
  const directionalContext = classifyDirectionalContext(currentTrend, higherTrend, liquidity, structure);
  const direction = directionalContext.direction;

  const trendScore = Math.max(currentTrend.score, higherTrend.score);
  const breakdown: ScoreBreakdown = {
    trendAlignment: { score: trendScore, reasons: [] },
    liquiditySweep: liquidity,
    marketStructure: structure,
    orderBlock,
    fairValueGap: fvg,
    volumeConfirmation: volume,
    sessionAllowed: session,
    volatilityValid: volatility
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
        score,
        config,
        action: "reject",
        blockedBy: ["countertrend"]
      })
    };
  }

  if (!hasConfirmationCandle(entryCandles, direction)) {
    return {
      status: "rejected",
      score,
      reasons: [...reasons, "Rejected: no confirmation candle"],
      finalDecision: buildFinalDecision({
        direction,
        currentTrend,
        higherTrend,
        counterTrend: directionalContext.counterTrend,
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

  const entry = direction === "long" ? latest.close + spread / 2 : latest.close - spread / 2;
  const stopLoss = stopLossForDirection(entryCandles, direction, entry, config);
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
        score,
        config,
        action: "reject",
        blockedBy: ["riskLimit"]
      })
    };
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
      score,
      config,
      action: "trade"
    }),
    signal: {
      symbol: config.symbol,
      direction,
      entry,
      stopLoss,
      takeProfits: takeProfits(entry, stopLoss, direction, config),
      score,
      reasons,
      timestamp: latest.time,
      regime: regime.regime
    }
  };
}
