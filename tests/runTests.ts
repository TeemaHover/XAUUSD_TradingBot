import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { atr } from "../src/indicators/atr";
import { ema } from "../src/indicators/ema";
import { loadConfig } from "../src/config/loadConfig";
import { sampleCandles } from "../src/data/sampleCandles";
import { sessionFilter } from "../src/filters/sessionFilter";
import { detectLiquidity } from "../src/strategy/liquidityDetector";
import { calculatePositionSize, normalizeVolume, RiskGuard } from "../src/risk/positionSizing";
import { buildFinalDecision, calculateSignal, classifyDirectionalContext, counterTrendRejectionReason } from "../src/strategy/signalEngine";
import { resolveTrade, runBacktest } from "../src/backtest/backtestEngine";
import { runWalkForward } from "../src/backtest/walkForward";
import { detectMarketRegime } from "../src/strategy/marketRegimeDetector";
import { newsFilter } from "../src/filters/newsFilter";
import { adaptiveRawScore } from "../src/strategy/adaptiveScoring";
import { TradeGuard } from "../src/execution/tradeGuards";
import { Candle, LiquidityResult, MarketStructureResult, TradeSignal, TrendResult } from "../src/types";
import { formatOpenPositionAlert } from "../src/execution/executionEngine";
import { formatBotStartedAlert, formatBotStoppedAlert } from "../src/main";
import { SqliteJournal } from "../src/journal/sqliteJournal";

function testEma(): void {
  const values = ema([1, 2, 3, 4], 3);
  assert.equal(values.length, 4);
  assert.equal(values[0], 1);
  assert.ok(values.at(-1)! > values[0]);
}

function testAtr(): void {
  const candles = sampleCandles(20);
  const values = atr(candles, 14);
  assert.equal(values.length, candles.length);
  assert.ok(values.at(-1)! > 0);
}

function testRiskSizing(): void {
  const config = loadConfig();
  const volume = calculatePositionSize(10000, 0.005, 2300, 2295, config.risk);
  assert.equal(volume, 0.1);
}

function testBotLoopEnvOverrides(): void {
  const previousLoop = process.env.BOT_LOOP_ENABLED;
  const previousInterval = process.env.BOT_INTERVAL_SECONDS;
  process.env.BOT_LOOP_ENABLED = "false";
  process.env.BOT_INTERVAL_SECONDS = "15";
  try {
    const config = loadConfig();
    assert.equal(config.bot.loopEnabled, false);
    assert.equal(config.bot.intervalSeconds, 15);
  } finally {
    if (previousLoop === undefined) {
      delete process.env.BOT_LOOP_ENABLED;
    } else {
      process.env.BOT_LOOP_ENABLED = previousLoop;
    }
    if (previousInterval === undefined) {
      delete process.env.BOT_INTERVAL_SECONDS;
    } else {
      process.env.BOT_INTERVAL_SECONDS = previousInterval;
    }
  }
}

function testTelegramEnvOverrides(): void {
  const previousEnabled = process.env.TELEGRAM_ENABLED;
  const previousToken = process.env.TELEGRAM_BOT_TOKEN;
  const previousChat = process.env.TELEGRAM_CHAT_ID;
  process.env.TELEGRAM_ENABLED = "true";
  process.env.TELEGRAM_BOT_TOKEN = "test-token";
  process.env.TELEGRAM_CHAT_ID = "test-chat";
  try {
    const config = loadConfig();
    assert.equal(config.alerts.telegram.enabled, true);
    assert.equal(config.alerts.telegram.botToken, "test-token");
    assert.equal(config.alerts.telegram.chatId, "test-chat");
  } finally {
    if (previousEnabled === undefined) {
      delete process.env.TELEGRAM_ENABLED;
    } else {
      process.env.TELEGRAM_ENABLED = previousEnabled;
    }
    if (previousToken === undefined) {
      delete process.env.TELEGRAM_BOT_TOKEN;
    } else {
      process.env.TELEGRAM_BOT_TOKEN = previousToken;
    }
    if (previousChat === undefined) {
      delete process.env.TELEGRAM_CHAT_ID;
    } else {
      process.env.TELEGRAM_CHAT_ID = previousChat;
    }
  }
}

function testOpenPositionAlertMessage(): void {
  const config = loadConfig();
  const signal: TradeSignal = {
    symbol: config.symbol,
    direction: "long",
    entry: 2300,
    stopLoss: 2295,
    takeProfits: [2305, 2310],
    score: 88,
    reasons: [],
    timestamp: Date.UTC(2026, 0, 1)
  };
  const message = formatOpenPositionAlert({
    ...signal,
    id: "position-1",
    volume: 0.1,
    openedAt: signal.timestamp,
    remainingVolume: 0.1
  }, signal, config);

  assert.match(message, /Trade opened/);
  assert.match(message, /Symbol: GOLD/);
  assert.match(message, /Direction: LONG/);
  assert.match(message, /Volume: 0.1/);
  assert.match(message, /Score: 88/);
}

function testSqliteJournalWrites(): void {
  const config = loadConfig();
  const dbPath = path.join(os.tmpdir(), `trading-journal-${Date.now()}.sqlite`);
  config.journal.enabled = true;
  config.journal.path = dbPath;

  const journal = new SqliteJournal(config);
  journal.init();
  journal.recordBotEvent("started", { test: true });
  journal.recordSignal(calculateSignal(sampleCandles(320), sampleCandles(320), sampleCandles(320), config.mockBroker.spread, config));
  journal.close();

  assert.equal(fs.existsSync(dbPath), true);

  const { DatabaseSync } = require("node:sqlite") as {
    DatabaseSync: new (filename: string) => {
      prepare(sql: string): { get(): { count: number } };
      close(): void;
    };
  };
  const db = new DatabaseSync(dbPath);
  try {
    assert.equal(db.prepare("SELECT COUNT(*) AS count FROM bot_events").get().count, 1);
    assert.equal(db.prepare("SELECT COUNT(*) AS count FROM signals").get().count, 1);
  } finally {
    db.close();
    fs.unlinkSync(dbPath);
  }
}

function testBotLifecycleAlertMessages(): void {
  const config = loadConfig();
  const started = formatBotStartedAlert(config, 10000);
  assert.match(started, /Bot started/);
  assert.match(started, /Symbol: GOLD/);
  assert.match(started, /Broker: MT5/);
  assert.match(started, /Loop: ON every \d+s/);
  assert.match(started, /Balance: 10000/);

  const stopped = formatBotStoppedAlert(config);
  assert.match(stopped, /Bot stopped/);
  assert.match(stopped, /Symbol: GOLD/);
  assert.match(stopped, /Broker: MT5/);
  assert.match(stopped, /Stopped at:/);
}

function testRiskGuard(): void {
  const guard = new RiskGuard(0.03, 3, 10000);
  assert.equal(guard.canTrade(), true);
  guard.recordResult(-100);
  guard.recordResult(-100);
  guard.recordResult(-100);
  assert.equal(guard.canTrade(), false);
}

function testRiskGuardHydratesFromHistory(): void {
  const guard = new RiskGuard(0.03, 3, 10000);
  guard.hydrateFromClosedProfits([-50, -75, -25]);
  assert.equal(guard.canTrade(), false);
  assert.equal(guard.status().dailyLoss, 150);
  assert.equal(guard.status().consecutiveLosses, 3);
}

function testVolumeNormalization(): void {
  const volume = normalizeVolume(0.237, {
    tickSize: 0.01,
    tickValue: 1,
    volumeStep: 0.05,
    minVolume: 0.1,
    maxVolume: 2
  });
  assert.equal(volume, 0.2);
}

function testSessionOffset(): void {
  const config = loadConfig();
  config.sessions.utcOffsetMinutes = 120;
  config.sessions.allowed = ["London"];
  const timestamp = Date.UTC(2026, 0, 1, 5, 0, 0);
  const result = sessionFilter(timestamp, config);
  assert.equal(result.score, config.scoring.sessionAllowed);
}

function testLiquidityShape(): void {
  const config = loadConfig();
  const result = detectLiquidity(sampleCandles(260), config);
  assert.ok(Array.isArray(result.swings));
  assert.equal(typeof result.bullishSweep, "boolean");
}

function testSignalDecisionShape(): void {
  const config = loadConfig();
  const candles = sampleCandles(320);
  const decision = calculateSignal(candles, candles, candles, config.mockBroker.spread, config);
  assert.ok(["trade", "watchlist", "rejected"].includes(decision.status));
  assert.equal(typeof decision.score, "number");
  assert.ok(Array.isArray(decision.reasons));
  assert.ok(["long", "short", "none"].includes(decision.finalDecision.direction));
  assert.ok(["bullish", "bearish", "sideways"].includes(decision.finalDecision.trendDirection));
  assert.ok(["trend-following", "countertrend", "no-context"].includes(decision.finalDecision.setupType));
  assert.equal(typeof decision.finalDecision.score, "number");
  assert.equal(typeof decision.finalDecision.requiredScore, "number");
  assert.equal(typeof decision.finalDecision.allowed, "boolean");
  assert.ok(["trade", "watchlist", "reject"].includes(decision.finalDecision.action));
  assert.ok(Array.isArray(decision.finalDecision.blockedBy));
}

function trend(bias: TrendResult["bias"]): TrendResult {
  return { bias, confidence: 80, score: 20, reasons: [`${bias} test trend`] };
}

function liquidity(flags: Partial<Pick<LiquidityResult, "bullishSweep" | "bearishSweep">> = {}): LiquidityResult {
  return {
    bullishSweep: flags.bullishSweep ?? false,
    bearishSweep: flags.bearishSweep ?? false,
    equalHighs: [],
    equalLows: [],
    swings: [],
    score: 0,
    reasons: []
  };
}

function structure(bos: MarketStructureResult["bos"], mss: MarketStructureResult["mss"] = "none"): MarketStructureResult {
  return { bos, mss, choch: mss, score: 0, reasons: [] };
}

function testDirectionalContextCombinations(): void {
  const cases = [
    {
      name: "bearish HTF plus bullish BOS is countertrend long",
      htf: "bearish" as const,
      bos: "bullish" as const,
      expectedDirection: "long",
      counterTrend: true
    },
    {
      name: "bearish HTF plus bearish BOS is aligned short",
      htf: "bearish" as const,
      bos: "bearish" as const,
      expectedDirection: "short",
      counterTrend: false
    },
    {
      name: "bearish HTF plus bearish MSS is aligned short",
      htf: "bearish" as const,
      bos: "none" as const,
      mss: "bearish" as const,
      expectedDirection: "short",
      counterTrend: false
    },
    {
      name: "bullish HTF plus bearish BOS is countertrend short",
      htf: "bullish" as const,
      bos: "bearish" as const,
      expectedDirection: "short",
      counterTrend: true
    },
    {
      name: "bullish HTF plus bullish BOS is aligned long",
      htf: "bullish" as const,
      bos: "bullish" as const,
      expectedDirection: "long",
      counterTrend: false
    }
  ];

  for (const item of cases) {
    const result = classifyDirectionalContext(
      trend(item.htf),
      trend(item.htf),
      liquidity(),
      structure(item.bos, item.mss ?? "none")
    );
    assert.equal(result.direction, item.expectedDirection, item.name);
    assert.equal(result.counterTrend, item.counterTrend, item.name);
  }
}

function testCountertrendConfigGate(): void {
  const config = loadConfig();
  const current = trend("bearish");
  const higher = trend("bearish");
  const context = classifyDirectionalContext(current, higher, liquidity(), structure("bullish"));
  assert.equal(context.direction, "long");
  assert.equal(context.counterTrend, true);
  assert.equal(config.strategy.allowCounterTrendTrades, false);
  assert.equal(counterTrendRejectionReason(context, 100, config), "Rejected: countertrend setup rejected");

  config.strategy.allowCounterTrendTrades = true;
  config.strategy.counterTrendMinScore = 85;
  assert.equal(counterTrendRejectionReason(context, 84, config), "Rejected: countertrend score 84 below minimum 85");
  assert.equal(counterTrendRejectionReason(context, 85, config), undefined);

  const aligned = classifyDirectionalContext(trend("bearish"), trend("bearish"), liquidity(), structure("bearish"));
  assert.equal(counterTrendRejectionReason(aligned, 50, config), undefined);
}

function testFinalDecisionContext(): void {
  const config = loadConfig();
  config.strategy.minScore = 70;
  config.strategy.counterTrendMinScore = 90;

  const noContext = buildFinalDecision({
    currentTrend: trend("bullish"),
    higherTrend: trend("bullish"),
    score: 42,
    config,
    action: "reject"
  });
  assert.equal(noContext.direction, "none");
  assert.equal(noContext.trendDirection, "bullish");
  assert.equal(noContext.setupType, "no-context");
  assert.equal(noContext.requiredScore, 70);
  assert.equal(noContext.allowed, false);
  assert.equal(noContext.action, "reject");

  const countertrend = buildFinalDecision({
    direction: "long",
    currentTrend: trend("bearish"),
    higherTrend: trend("bearish"),
    counterTrend: true,
    score: 91,
    config,
    action: "reject",
    blockedBy: ["countertrend"]
  });
  assert.equal(countertrend.direction, "long");
  assert.equal(countertrend.trendDirection, "bearish");
  assert.equal(countertrend.setupType, "countertrend");
  assert.equal(countertrend.requiredScore, 90);
  assert.deepEqual(countertrend.blockedBy, ["countertrend"]);

  const trade = buildFinalDecision({
    direction: "short",
    currentTrend: trend("bearish"),
    higherTrend: trend("bearish"),
    counterTrend: false,
    score: 80,
    config,
    action: "trade"
  });
  assert.equal(trade.direction, "short");
  assert.equal(trade.setupType, "trend-following");
  assert.equal(trade.allowed, true);
  assert.equal(trade.action, "trade");
}

function testBacktestRuns(): void {
  const config = loadConfig();
  const outputPath = path.join(os.tmpdir(), "xauusd-v1-backtest-test.json");
  const result = runBacktest(sampleCandles(360), config, outputPath);
  assert.equal(typeof result.totalTrades, "number");
  assert.equal(typeof result.maxDrawdown, "number");
  assert.ok(result.endingBalance > 0);
}

function testMarketRegimeDetector(): void {
  const config = loadConfig();
  const result = detectMarketRegime(sampleCandles(320), config);
  assert.ok(["trending", "ranging", "highVolatility", "lowVolatility"].includes(result.regime));
  assert.equal(typeof result.adx, "number");
}

function testNewsBlackout(): void {
  const config = loadConfig();
  const now = Date.UTC(2026, 0, 1, 12, 0, 0);
  config.news.events = [{ time: now, title: "NFP", impact: "high", symbols: [config.symbol] }];
  const result = newsFilter(now, config.symbol, config);
  assert.equal(result.blocked, true);
}

function testAdaptiveScoring(): void {
  const config = loadConfig();
  const result = adaptiveRawScore({
    trendAlignment: { score: 10, reasons: [] },
    liquiditySweep: { score: 10, reasons: [] },
    marketStructure: { score: 10, reasons: [] },
    orderBlock: { score: 10, reasons: [] },
    fairValueGap: { score: 10, reasons: [] },
    volumeConfirmation: { score: 10, reasons: [] },
    sessionAllowed: { score: 10, reasons: [] },
    volatilityValid: { score: 10, reasons: [] }
  }, "trending", config);
  assert.ok(result.rawScore > 0);
  assert.ok(result.maxScore > 0);
}

function testTradeGuard(): void {
  const config = loadConfig();
  config.tradeGuards.maxTradesPerSession = 1;
  const guard = new TradeGuard(config);
  const time = Date.UTC(2026, 0, 1, 8, 0, 0);
  assert.equal(guard.canOpen(time).allowed, true);
  guard.recordOpen(time);
  assert.equal(guard.canOpen(time).allowed, false);
  guard.recordClose(time, -100);
  assert.equal(guard.canOpen(time + 1).allowed, false);
}

function testWalkForwardRuns(): void {
  const config = loadConfig();
  config.walkForward.trainWindow = 260;
  config.walkForward.testWindow = 120;
  config.walkForward.stepSize = 120;
  const outputPath = path.join(os.tmpdir(), "xauusd-v2-walk-forward-test.json");
  const result = runWalkForward(sampleCandles(520), config, outputPath);
  assert.ok(result.summary.segments >= 1);
}

function testBacktestDoesNotUseCandlesBeyondEndIndex(): void {
  const config = loadConfig();
  config.tradeManagement.tpRMultiples = [1];
  config.mockBroker.spread = 0;
  config.mockBroker.slippage = 0;
  config.mockBroker.commissionPerLot = 0;
  config.risk.minStopDistance = 0.1;
  const candles: Candle[] = Array.from({ length: 230 }, (_, index) => ({
    time: Date.UTC(2026, 0, 1, 0, index, 0),
    open: 100,
    high: index === 222 ? 200 : 100.2,
    low: 99.8,
    close: 100,
    volume: 100
  }));

  const forcedSignal: TradeSignal = {
    symbol: config.symbol,
    direction: "long",
    entry: 100,
    stopLoss: 99,
    takeProfits: [101],
    score: 100,
    reasons: ["forced test signal"],
    timestamp: candles[220].time
  };

  const result = runBacktest(candles, config, path.join(os.tmpdir(), "xauusd-no-lookahead-test.json"), {
    startIndex: 220,
    endIndex: 222,
    signalProvider: (_entry, _trend, _higher, _spread, _config, index) => (
      index === 220
        ? {
          status: "trade",
          signal: forcedSignal,
          score: 100,
          reasons: [],
          finalDecision: buildFinalDecision({
            direction: "long",
            currentTrend: trend("bullish"),
            higherTrend: trend("bullish"),
            score: 100,
            config,
            action: "trade"
          })
        }
        : {
          status: "rejected",
          score: 0,
          reasons: [],
          finalDecision: buildFinalDecision({
            score: 0,
            config,
            action: "reject",
            blockedBy: ["lowScore"]
          })
        }
    )
  });

  assert.equal(result.totalTrades, 1);
  assert.equal(result.trades[0].exitReason, "timeout");
  assert.equal(result.trades[0].resultR, 0);
}

function testAmbiguousCandleIsStopFirst(): void {
  const config = loadConfig();
  config.mockBroker.spread = 0;
  config.mockBroker.slippage = 0;
  config.mockBroker.commissionPerLot = 0;
  const signal: TradeSignal = {
    symbol: config.symbol,
    direction: "long",
    entry: 100,
    stopLoss: 99,
    takeProfits: [101],
    score: 100,
    reasons: [],
    timestamp: Date.UTC(2026, 0, 1)
  };
  const result = resolveTrade(signal, [{
    time: signal.timestamp + 60_000,
    open: 100,
    high: 102,
    low: 98,
    close: 100,
    volume: 100
  }], config, 1, 10000);

  assert.equal(result.exitReason, "ambiguous_stop_first");
  assert.ok(result.resultR < 0);
}

testEma();
testAtr();
testRiskSizing();
testBotLoopEnvOverrides();
testTelegramEnvOverrides();
testOpenPositionAlertMessage();
testSqliteJournalWrites();
testBotLifecycleAlertMessages();
testRiskGuard();
testRiskGuardHydratesFromHistory();
testVolumeNormalization();
testSessionOffset();
testLiquidityShape();
testSignalDecisionShape();
testDirectionalContextCombinations();
testCountertrendConfigGate();
testFinalDecisionContext();
testBacktestRuns();
testMarketRegimeDetector();
testNewsBlackout();
testAdaptiveScoring();
testTradeGuard();
testWalkForwardRuns();
testBacktestDoesNotUseCandlesBeyondEndIndex();
testAmbiguousCandleIsStopFirst();

console.log("All tests passed");
