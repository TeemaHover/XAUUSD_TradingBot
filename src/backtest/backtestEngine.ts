import fs from "node:fs";
import { AppConfig, Candle, Direction, TradeSignal } from "../types";
import { calculateSignal } from "../strategy/signalEngine";
import { logger } from "../logger/logger";
import { calculatePositionSize, RiskGuard } from "../risk/positionSizing";
import { TradeGuard } from "../execution/tradeGuards";

export interface BacktestResult {
  totalTrades: number;
  winRate: number;
  profitFactor: number;
  maxDrawdown: number;
  expectancy: number;
  totalReturn: number;
  averageR: number;
  endingBalance: number;
  performanceSummary: {
    grossWinR: number;
    grossLossR: number;
    rejectedByRisk: number;
    rejectedByTradeGuard: number;
    canceledLimitOrders: number;
  };
  trades: BacktestTrade[];
}

export interface BacktestOptions {
  startIndex?: number;
  endIndex?: number;
  maxHoldBars?: number;
  signalProvider?: (
    entryWindow: Candle[],
    trendWindow: Candle[],
    higherTrendWindow: Candle[],
    spread: number,
    config: AppConfig,
    index: number
  ) => ReturnType<typeof calculateSignal>;
}

interface BacktestTrade {
  entryTime: number;
  exitTime: number;
  direction: Direction;
  entry: number;
  stopLoss: number;
  takeProfits: number[];
  resultR: number;
  score: number;
  exitReason: string;
  commissionR: number;
  slippageR: number;
}

export function resolveTrade(
  signal: TradeSignal,
  futureCandles: Candle[],
  config: AppConfig,
  volume: number,
  balance: number,
  maxBars = 96
): { exitTime: number; resultR: number; exitReason: string; commissionR: number; slippageR: number } {
  const initialRisk = Math.abs(signal.entry - signal.stopLoss);
  const tpFractions = signal.takeProfits.map(() => 1 / signal.takeProfits.length);
  const hitTargets = new Set<number>();
  let stopLoss = signal.stopLoss;
  let realizedR = 0;
  const slippageR = initialRisk > 0 ? (config.mockBroker.slippage * 2) / initialRisk : 0;
  const riskAmount = balance * config.risk.riskPerTrade;
  const commissionR = riskAmount > 0 ? (config.mockBroker.commissionPerLot * volume * 2) / riskAmount : 0;
  const halfSpread = config.mockBroker.spread / 2;

  for (const candle of futureCandles.slice(0, maxBars)) {
    if (signal.direction === "long") {
      const stopTouched = candle.low <= stopLoss + halfSpread;
      const touchedTargets = signal.takeProfits
        .map((target, index) => ({ target, index }))
        .filter(({ target, index }) => !hitTargets.has(index) && candle.high >= target + halfSpread);

      if (stopTouched && touchedTargets.length > 0) {
        const remainingFraction = 1 - [...hitTargets].reduce((sum, index) => sum + tpFractions[index], 0);
        const stopR = stopLoss >= signal.entry ? 0 : -1;
        return {
          exitTime: candle.time,
          resultR: realizedR + remainingFraction * stopR - commissionR - slippageR,
          exitReason: "ambiguous_stop_first",
          commissionR,
          slippageR
        };
      }

      if (stopTouched) {
        const remainingFraction = 1 - [...hitTargets].reduce((sum, index) => sum + tpFractions[index], 0);
        const stopR = stopLoss >= signal.entry ? 0 : -1;
        return {
          exitTime: candle.time,
          resultR: realizedR + remainingFraction * stopR - commissionR - slippageR,
          exitReason: stopR === 0 ? "breakeven" : "stop_loss",
          commissionR,
          slippageR
        };
      }
      for (const { index: targetIndex } of touchedTargets) {
        hitTargets.add(targetIndex);
        realizedR += tpFractions[targetIndex] * (initialRisk > 0
          ? Math.abs(signal.takeProfits[targetIndex] - signal.entry) / initialRisk
          : 0);
        if (targetIndex === 0 && config.tradeManagement.moveToBreakEvenAfterTp1) {
          stopLoss = signal.entry;
        }
      }
    } else {
      const stopTouched = candle.high >= stopLoss - halfSpread;
      const touchedTargets = signal.takeProfits
        .map((target, index) => ({ target, index }))
        .filter(({ target, index }) => !hitTargets.has(index) && candle.low <= target - halfSpread);

      if (stopTouched && touchedTargets.length > 0) {
        const remainingFraction = 1 - [...hitTargets].reduce((sum, index) => sum + tpFractions[index], 0);
        const stopR = stopLoss <= signal.entry ? 0 : -1;
        return {
          exitTime: candle.time,
          resultR: realizedR + remainingFraction * stopR - commissionR - slippageR,
          exitReason: "ambiguous_stop_first",
          commissionR,
          slippageR
        };
      }

      if (stopTouched) {
        const remainingFraction = 1 - [...hitTargets].reduce((sum, index) => sum + tpFractions[index], 0);
        const stopR = stopLoss <= signal.entry ? 0 : -1;
        return {
          exitTime: candle.time,
          resultR: realizedR + remainingFraction * stopR - commissionR - slippageR,
          exitReason: stopR === 0 ? "breakeven" : "stop_loss",
          commissionR,
          slippageR
        };
      }
      for (const { index: targetIndex } of touchedTargets) {
        hitTargets.add(targetIndex);
        realizedR += tpFractions[targetIndex] * (initialRisk > 0
          ? Math.abs(signal.takeProfits[targetIndex] - signal.entry) / initialRisk
          : 0);
        if (targetIndex === 0 && config.tradeManagement.moveToBreakEvenAfterTp1) {
          stopLoss = signal.entry;
        }
      }
    }

    if (hitTargets.size === signal.takeProfits.length) {
      return {
        exitTime: candle.time,
        resultR: realizedR - commissionR - slippageR,
        exitReason: "all_targets",
        commissionR,
        slippageR
      };
    }
  }

  const last = futureCandles[Math.min(maxBars, futureCandles.length) - 1];
  return {
    exitTime: last?.time ?? signal.timestamp,
    resultR: realizedR - commissionR - slippageR,
    exitReason: "timeout",
    commissionR,
    slippageR
  };
}

function maxDrawdown(equityCurve: number[]): number {
  let peak = equityCurve[0] ?? 0;
  let drawdown = 0;
  for (const value of equityCurve) {
    peak = Math.max(peak, value);
    drawdown = Math.max(drawdown, peak - value);
  }
  return drawdown;
}

export function applyNextOpenFill(signal: TradeSignal, nextCandle: Candle, config: AppConfig): TradeSignal | undefined {
  const entry = signal.direction === "long"
    ? nextCandle.open + config.mockBroker.spread / 2 + config.mockBroker.slippage
    : nextCandle.open - config.mockBroker.spread / 2 - config.mockBroker.slippage;
  const risk = Math.abs(entry - signal.stopLoss);
  if (risk < config.risk.minStopDistance) return undefined;
  // A gap between signal and fill can invert the trade geometry: fill beyond
  // the stop (instant fake-"breakeven" exit) or beyond a price-mode target
  // (instant fake win). A real broker would reject these stops — skip them.
  const stopInverted = signal.direction === "long" ? entry <= signal.stopLoss : entry >= signal.stopLoss;
  if (stopInverted) return undefined;
  if (signal.tpMode === "price") {
    const tpInverted = signal.takeProfits.some((tp) => (
      signal.direction === "long" ? tp <= entry : tp >= entry
    ));
    if (tpInverted) return undefined;
  }

  // Price-level TPs (range boundaries, structure targets) stay put when the
  // fill price shifts; R-multiple TPs are recomputed from the actual fill.
  const takeProfits = signal.tpMode === "price"
    ? signal.takeProfits
    : config.tradeManagement.tpRMultiples.map((multiple) => (
      signal.direction === "long" ? entry + risk * multiple : entry - risk * multiple
    ));

  return {
    ...signal,
    entry,
    timestamp: nextCandle.time,
    takeProfits
  };
}

/**
 * Simulate a pending limit order: returns the index of the bar that fills it
 * (bid/ask touches the limit price) or -1 if it expires untouched.
 */
export function findLimitFillIndex(
  signal: TradeSignal,
  candles: Candle[],
  fromIndex: number,
  expiryBars: number,
  endIndex: number,
  halfSpread: number
): number {
  const limitEnd = Math.min(fromIndex + expiryBars, endIndex);
  for (let j = fromIndex; j < limitEnd; j += 1) {
    const candle = candles[j];
    const touched = signal.direction === "long"
      ? candle.low <= signal.entry - halfSpread
      : candle.high >= signal.entry + halfSpread;
    if (touched) return j;
  }
  return -1;
}

export function runBacktest(
  candles: Candle[],
  config: AppConfig,
  outputPath = "backtest-results.json",
  options: BacktestOptions = {}
): BacktestResult {
  const trades: BacktestTrade[] = [];
  let balance = config.mockBroker.balance;
  const equityCurve: number[] = [balance];
  const warmup = Math.max(config.strategy.emaLength + 10, 220);
  const riskGuard = new RiskGuard(config.risk.maxDailyLoss, config.risk.maxConsecutiveLosses, config.mockBroker.balance);
  const tradeGuard = new TradeGuard(config);
  let rejectedByRisk = 0;
  let rejectedByTradeGuard = 0;
  let canceledLimitOrders = 0;

  const startIndex = Math.max(warmup, options.startIndex ?? warmup);
  const maxHoldBars = options.maxHoldBars ?? 96;
  const endIndex = Math.min(candles.length, options.endIndex ?? candles.length);

  let currentDay = -1;
  const totalCandles = Math.max(1, endIndex - 1 - startIndex);
  for (let i = startIndex; i < endIndex - 1; i += 1) {
    // Heartbeat so quiet log levels still show the backtest is alive.
    const processed = i - startIndex;
    if (processed > 0 && processed % 20000 === 0) {
      logger.warn(`Backtest progress: ${Math.round((processed / totalCandles) * 100)}% (candle ${processed.toLocaleString()} of ${totalCandles.toLocaleString()})`);
    }
    // New UTC day: reset daily loss + consecutive-loss streak (mirrors live,
    // where the risk guard is rehydrated from the current day's history).
    const candleDay = Math.floor(candles[i].time / 86_400_000);
    if (candleDay !== currentDay) {
      currentDay = candleDay;
      riskGuard.resetDaily();
    }

    const entryWindow = candles.slice(0, i + 1);
    const trendWindow = candles.slice(Math.max(0, i - 260), i + 1);
    const higherTrendWindow = candles.slice(Math.max(0, i - 260), i + 1);
    const decision = options.signalProvider
      ? options.signalProvider(entryWindow, trendWindow, higherTrendWindow, config.mockBroker.spread, config, i)
      : calculateSignal(entryWindow, trendWindow, higherTrendWindow, config.mockBroker.spread, config);

    logger.info("Backtest signal", {
      index: i,
      status: decision.status,
      score: decision.score,
      finalDecision: decision.finalDecision
    });
    if (!decision.signal) continue;
    if (!riskGuard.canTrade()) {
      rejectedByRisk += 1;
      logger.warn("Backtest trade rejected by risk guard", riskGuard.status());
      continue;
    }
    const guard = tradeGuard.canOpen(decision.signal.timestamp);
    if (!guard.allowed) {
      rejectedByTradeGuard += 1;
      logger.warn("Backtest trade rejected by trade guard", { reasons: guard.reasons });
      continue;
    }

    let filledSignal: TradeSignal | undefined;
    let resolutionStart = i + 1;

    if (decision.signal.entryType === "limit") {
      const expiryBars = config.tradeManagement.limitExpiryBars ?? 12;
      const fillIndex = findLimitFillIndex(
        decision.signal,
        candles,
        i + 1,
        expiryBars,
        endIndex,
        config.mockBroker.spread / 2
      );
      if (fillIndex < 0) {
        canceledLimitOrders += 1;
        logger.info("Backtest limit order expired untouched", { index: i, entry: decision.signal.entry });
        i += expiryBars; // one pending order at a time — no new signals while waiting
        continue;
      }
      filledSignal = { ...decision.signal, timestamp: candles[fillIndex].time };
      // Include the fill bar: it can run through the zone and hit the stop same-bar
      resolutionStart = fillIndex;
      i = fillIndex;
    } else {
      filledSignal = applyNextOpenFill(decision.signal, candles[i + 1], config);
    }

    if (!filledSignal) {
      logger.warn("Backtest trade rejected: next-open fill invalidated stop distance", {
        index: i,
        symbol: decision.signal.symbol
      });
      continue;
    }

    const volume = calculatePositionSize(
      balance,
      config.risk.riskPerTrade,
      filledSignal.entry,
      filledSignal.stopLoss,
      config.risk
    );
    const resolutionCandles = candles.slice(resolutionStart, endIndex);
    const resolved = resolveTrade(filledSignal, resolutionCandles, config, volume, balance, maxHoldBars);
    const profit = resolved.resultR * balance * config.risk.riskPerTrade;
    balance += profit;
    riskGuard.recordResult(profit);
    tradeGuard.recordOpen(filledSignal.timestamp);
    tradeGuard.recordClose(resolved.exitTime, profit);
    const trade: BacktestTrade = {
      entryTime: filledSignal.timestamp,
      exitTime: resolved.exitTime,
      direction: filledSignal.direction,
      entry: filledSignal.entry,
      stopLoss: filledSignal.stopLoss,
      takeProfits: filledSignal.takeProfits,
      resultR: resolved.resultR,
      score: filledSignal.score,
      exitReason: resolved.exitReason,
      commissionR: resolved.commissionR,
      slippageR: resolved.slippageR
    };
    trades.push(trade);
    equityCurve.push(balance);
    i += 12;
  }

  const wins = trades.filter((trade) => trade.resultR > 0);
  const losses = trades.filter((trade) => trade.resultR < 0);
  const grossWin = wins.reduce((sum, trade) => sum + trade.resultR, 0);
  const grossLoss = Math.abs(losses.reduce((sum, trade) => sum + trade.resultR, 0));
  const totalR = trades.reduce((sum, trade) => sum + trade.resultR, 0);
  const result: BacktestResult = {
    totalTrades: trades.length,
    winRate: trades.length ? wins.length / trades.length : 0,
    profitFactor: grossLoss > 0 ? grossWin / grossLoss : grossWin > 0 ? Infinity : 0,
    maxDrawdown: maxDrawdown(equityCurve) / config.mockBroker.balance,
    expectancy: trades.length ? totalR / trades.length : 0,
    totalReturn: (balance - config.mockBroker.balance) / config.mockBroker.balance,
    averageR: trades.length ? totalR / trades.length : 0,
    endingBalance: balance,
    performanceSummary: {
      grossWinR: grossWin,
      grossLossR: grossLoss,
      rejectedByRisk,
      rejectedByTradeGuard,
      canceledLimitOrders
    },
    trades
  };

  fs.writeFileSync(outputPath, `${JSON.stringify(result, null, 2)}\n`);
  logger.info("Backtest results", result);
  return result;
}
