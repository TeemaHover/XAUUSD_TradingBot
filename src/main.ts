import { MockBroker } from "./broker/MockBroker";
import { Mt5Broker } from "./broker/Mt5Broker";
import { MetaApiBroker } from "./broker/MetaApiBroker";
import { Broker } from "./broker/Broker";
import { sampleCandles } from "./data/sampleCandles";
import { loadConfig } from "./config/loadConfig";
import { DashboardServer, DashboardState } from "./dashboard/dashboardServer";
import { ExecutionEngine } from "./execution/executionEngine";
import { logger } from "./logger/logger";
import { RiskGuard } from "./risk/positionSizing";
import { calculateSignal } from "./strategy/signalEngine";
import { detectRange } from "./strategy/rangeDetector";
import { detectSR } from "./strategy/srDetector";
import { AppConfig, Candle, TradingMode } from "./types";
import { TelegramAlerts } from "./alerts/telegram";
import { SqliteJournal } from "./journal/sqliteJournal";
import { applyTradingMode, TRADING_MODES } from "./modes/tradingModes";
import { aiPredict, buildAiSignal } from "./strategy/aiSignalEngine";
import { detectMarketRegime } from "./strategy/marketRegimeDetector";
import { hmmPredict } from "./strategy/hmmRegime";

interface MarketSnapshot {
  entryCandles: Candle[];
  trendCandles: Candle[];
  higherTrendCandles: Candle[];
}

async function fetchMarketSnapshot(broker: Broker, config: AppConfig): Promise<MarketSnapshot> {
  const entryCandles = await broker.getCandles(config.symbol, config.timeframes.entry, config.mt5.bars.entry);
  const trendCandles = await broker.getCandles(config.symbol, config.timeframes.trend, config.mt5.bars.trend);
  const higherTrendCandles = await broker.getCandles(config.symbol, config.timeframes.higherTrend, config.mt5.bars.higherTrend);
  return { entryCandles, trendCandles, higherTrendCandles };
}

async function createBroker(config: AppConfig): Promise<Broker> {
  // "auto": same command works on both machines — Windows talks to the local
  // MT5 terminal, macOS/Linux goes through MetaApi cloud.
  if (config.broker.mode === "auto") {
    const resolved = process.platform === "win32" ? "mt5" : "metaapi";
    logger.info("Broker mode AUTO resolved by platform", {
      platform: process.platform,
      broker: resolved
    });
    config.broker.mode = resolved;
  }

  if (config.broker.mode === "mt5") {
    const broker = new Mt5Broker(config);
    await broker.connect();
    return broker;
  }

  if (config.broker.mode === "metaapi") {
    const broker = new MetaApiBroker(config);
    await broker.connect();
    return broker;
  }

  // Mock mode: try seeding real candles through the MT5 bridge; if the
  // bridge is unavailable (e.g. macOS), fall back to generated samples.
  let entryCandles: Candle[];
  let trendCandles: Candle[];
  let higherTrendCandles: Candle[];
  try {
    const mt5 = new Mt5Broker(config);
    await mt5.connect();
    [entryCandles, trendCandles, higherTrendCandles] = await Promise.all([
      mt5.getCandles(config.symbol, config.timeframes.entry, config.mt5.bars.entry),
      mt5.getCandles(config.symbol, config.timeframes.trend, config.mt5.bars.trend),
      mt5.getCandles(config.symbol, config.timeframes.higherTrend, config.mt5.bars.higherTrend)
    ]);
    await mt5.disconnect();
  } catch (error) {
    logger.warn("MT5 bridge unavailable, mock broker will use generated sample candles", {
      error: error instanceof Error ? error.message : String(error)
    });
    entryCandles = sampleCandles(config.mt5.bars.entry);
    trendCandles = sampleCandles(config.mt5.bars.trend);
    higherTrendCandles = sampleCandles(config.mt5.bars.higherTrend);
  }

  const broker = new MockBroker(config.mockBroker.balance, config.mockBroker.spread, {
    [config.timeframes.entry]: entryCandles,
    [config.timeframes.trend]: trendCandles,
    [config.timeframes.higherTrend]: higherTrendCandles
  });
  await broker.connect();
  return broker;
}

function signalAction(decision: ReturnType<typeof calculateSignal>): "BUY" | "SELL" | "WAIT" {
  if (decision.signal?.direction === "long") return "BUY";
  if (decision.signal?.direction === "short") return "SELL";
  return "WAIT";
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => { setTimeout(resolve, ms); });
}

/**
 * Parse --mode <beginner|advanced|expert|dumb> from process.argv.
 * Returns undefined if not specified.
 */
function parseModeArg(): TradingMode | undefined {
  const idx = process.argv.indexOf("--mode");
  if (idx === -1) return undefined;
  const val = process.argv[idx + 1];
  const valid: TradingMode[] = ["beginner", "advanced", "expert", "dumb", "ai"];
  if (valid.includes(val as TradingMode)) return val as TradingMode;
  logger.warn("Unknown --mode value, ignoring", { val, valid });
  return undefined;
}

/** Returns a one-line summary of the nearest support/resistance zones around price. */
function srSummary(sr: ReturnType<typeof detectSR>, price: number): string {
  // Nearest support = highest support zone whose high is below current price
  const supports = sr.supportZones
    .filter((z) => z.high < price)
    .sort((a, b) => b.high - a.high);

  // Nearest resistance = lowest resistance zone whose low is above current price
  const resistances = sr.resistanceZones
    .filter((z) => z.low > price)
    .sort((a, b) => a.low - b.low);

  const supportStr = supports.length > 0
    ? supports[0].low.toFixed(2) + " - " + supports[0].high.toFixed(2) + " (str:" + supports[0].strength + ")"
    : "none";

  const resistanceStr = resistances.length > 0
    ? resistances[0].low.toFixed(2) + " - " + resistances[0].high.toFixed(2) + " (str:" + resistances[0].strength + ")"
    : "none";

  return "S " + supportStr + "  |  R " + resistanceStr;
}

export function formatBotStartedAlert(config: AppConfig, balance: number, mode: TradingMode): string {
  return [
    "Bot started",
    "Symbol: " + config.symbol,
    "Broker: " + config.broker.mode.toUpperCase(),
    "Mode: " + mode.toUpperCase() + " -- " + TRADING_MODES[mode].description,
    "Loop: " + (config.bot.loopEnabled ? "ON every " + config.bot.intervalSeconds + "s" : "OFF"),
    "Timeframes: entry=" + config.timeframes.entry + ", trend=" + config.timeframes.trend + ", higherTrend=" + config.timeframes.higherTrend,
    "Balance: " + balance,
    "Execution: " + (config.mt5.dryRun ? "DRY_RUN" : "DEMO/LIVE ORDERING ENABLED")
  ].join("\n");
}

export function formatBotStoppedAlert(config: AppConfig): string {
  return [
    "Bot stopped",
    "Symbol: " + config.symbol,
    "Broker: " + config.broker.mode.toUpperCase(),
    "Stopped at: " + new Date().toISOString()
  ].join("\n");
}

interface LoopState {
  lastCandleTime?: number;
  skipLogged?: boolean;
}

async function scanOnce(
  broker: Broker,
  config: AppConfig,
  execution: ExecutionEngine,
  dashboardState: DashboardState,
  journal: SqliteJournal,
  mode: TradingMode,
  loopState: LoopState
): Promise<void> {
  const { entryCandles, trendCandles, higherTrendCandles } = await fetchMarketSnapshot(broker, config);

  // One decision per entry candle: with a short poll interval the same candle
  // would otherwise be evaluated many times, firing duplicate market orders
  // and long/short flip-flops inside a single bar. The bridge returns closed
  // bars only, so a new last-candle time means a candle just closed.
  const latestCandleTime = entryCandles.at(-1)?.time;
  if ((config.bot.evaluateOncePerCandle ?? true) && latestCandleTime !== undefined) {
    if (loopState.lastCandleTime === latestCandleTime) {
      if (!loopState.skipLogged) {
        logger.info(
          `Candle ${new Date(latestCandleTime).toISOString()} already evaluated — waiting for next ${config.timeframes.entry} close`
        );
        loopState.skipLogged = true;
      }
      return;
    }
    loopState.lastCandleTime = latestCandleTime;
    loopState.skipLogged = false;
  }

  // Regime snapshot for the dashboard (once per candle, display-only).
  const rulesRegime = detectMarketRegime(entryCandles, config);
  const hmmRegime = await hmmPredict(
    entryCandles,
    config.regime.hmmModelPath ?? "models/hmm_model.npz",
    config.mt5.pythonPath
  );
  dashboardState.regime = {
    hmmState: hmmRegime.state,
    hmmLabel: hmmRegime.label,
    hmmConfidence: hmmRegime.confidence,
    rules: rulesRegime.regime,
    updatedAt: Date.now()
  };

  const spread = await broker.getSpread(config.symbol);
  // --- AI mode: bypass rule engine, use neural network ---
  let decision: ReturnType<typeof calculateSignal>;
  if (config.strategy.aiMode) {
    const prediction = await aiPredict(
      {
        [config.timeframes.entry]: entryCandles,
        [config.timeframes.trend]: trendCandles,
        [config.timeframes.higherTrend]: higherTrendCandles
      },
      config.strategy.aiModelPath,
      config.mt5.pythonPath
    );
    const aiResult = buildAiSignal(prediction, entryCandles, spread, config, higherTrendCandles, trendCandles);
    decision = {
      status: aiResult.status,
      score: aiResult.score,
      reasons: aiResult.reasons,
      finalDecision: {
        direction: aiResult.direction ?? "none",
        trendDirection: "sideways",
        setupType: "ai-trade",
        score: aiResult.score,
        requiredScore: Math.round((config.strategy.aiConfidenceThreshold ?? 0.55) * 100),
        allowed: aiResult.status === "trade",
        blockedBy: [],
        action: aiResult.status === "trade" ? "trade" : "reject"
      },
      signal: aiResult.status === "trade" && aiResult.direction ? {
        symbol: config.symbol,
        direction: aiResult.direction,
        entry: aiResult.entry!,
        stopLoss: aiResult.stopLoss!,
        takeProfits: aiResult.takeProfits!,
        entryType: aiResult.entryType,
        score: aiResult.score,
        reasons: aiResult.reasons,
        timestamp: entryCandles.at(-1)?.time ?? Date.now()
      } : undefined
    };
  } else {
    decision = calculateSignal(entryCandles, trendCandles, higherTrendCandles, spread, config);
  }
  const action = signalAction(decision);
  dashboardState.lastDecision = decision;
  journal.recordSignal(decision);

  const recentHigh = Math.max(...entryCandles.slice(-20).map((c) => c.high));
  const recentLow  = Math.min(...entryCandles.slice(-20).map((c) => c.low));
  const currentPrice = entryCandles.at(-1)?.close ?? 0;
  const range = detectRange(entryCandles, config, config.timeframes.entry);
  const sr = detectSR(entryCandles, config);

  if (decision.status === "rejected" && !decision.signal) {
    // Routine WAIT cycle: one line instead of the full block. The AI line in
    // reasons carries direction+confidence; last reason is the blocker.
    const aiLine = decision.reasons.find((r) => r.includes("Confidence")) ?? "";
    const blocker = decision.reasons[decision.reasons.length - 1] ?? "";
    logger.info(
      `WAIT ${currentPrice.toFixed(2)} | ${aiLine.replace("- ", "")}${blocker && blocker !== aiLine ? " | " + blocker : ""}`
    );
  } else {
    logger.info("Signal calculation", {
      mode,
      action,
      status: decision.status,
      score: decision.score,
      price: currentPrice.toFixed(2),
      recentHigh: recentHigh.toFixed(2),
      recentLow: recentLow.toFixed(2),
      nearestSR: srSummary(sr, currentPrice),
      range: range.summary,
      finalDecision: decision.finalDecision,
      reasons: decision.reasons
    });
  }

  if (decision.signal) {
    dashboardState.lastSignal = decision.signal;
    await execution.execute(decision.signal);
  }
}

async function main(): Promise<void> {
  const baseConfig = loadConfig();

  // Initialise journal early so we can read saved mode
  const journal = new SqliteJournal(baseConfig);
  journal.init();

  // Resolve trading mode: CLI arg > saved in SQLite > default "expert"
  const argMode = parseModeArg();
  const savedMode = journal.loadMode();
  const mode: TradingMode = argMode ?? savedMode ?? "expert";

  // If a new mode was passed via CLI, persist it
  if (argMode && argMode !== savedMode) {
    journal.saveMode(argMode);
  }

  // Apply mode overrides to config
  const config = applyTradingMode(baseConfig, mode);

  const broker = await createBroker(config);
  const startingBalance = await broker.getBalance();
  const riskGuard = new RiskGuard(config.risk.maxDailyLoss, config.risk.maxConsecutiveLosses, startingBalance);
  const execution = new ExecutionEngine(broker, config, riskGuard, journal);
  const dashboardState: DashboardState = { dailyPnl: 0 };
  const dashboard = new DashboardServer(config, broker, dashboardState);
  const alerts = new TelegramAlerts(config);
  let stopping = false;

  logger.info("Trading mode active", {
    mode,
    description: TRADING_MODES[mode].description,
    minScore: config.strategy.minScore,
    watchlistScore: config.strategy.watchlistScore,
    allowCounterTrend: config.strategy.allowCounterTrendTrades,
    requireConfirmation: config.strategy.requireConfirmationCandle,
    dumbMode: config.strategy.dumbMode,
    rangeLong: config.strategy.rangeLongThreshold + "%",
    rangeShort: config.strategy.rangeShortThreshold + "%"
  });

  dashboard.start();
  process.once("SIGINT", () => { stopping = true; });
  process.once("SIGTERM", () => { stopping = true; });
  journal.recordBotEvent("started", {
    broker: config.broker.mode,
    mode,
    loopEnabled: config.bot.loopEnabled,
    intervalSeconds: config.bot.intervalSeconds,
    balance: startingBalance
  });
  await alerts.send(formatBotStartedAlert(config, startingBalance, mode));

  const loopState: LoopState = {};

  try {
    await scanOnce(broker, config, execution, dashboardState, journal, mode, loopState);

    while (config.bot.loopEnabled && !stopping) {
      await sleep(config.bot.intervalSeconds * 1000);
      if (stopping) break;
      try {
        await scanOnce(broker, config, execution, dashboardState, journal, mode, loopState);
      } catch (error) {
        logger.error("Signal loop failed", { message: error instanceof Error ? error.message : String(error) });
        journal.recordBotEvent("error", { message: error instanceof Error ? error.message : String(error) });
      }
    }

    if (config.dashboard.enabled && !config.bot.loopEnabled) {
      logger.info("Dashboard is running. Press Ctrl+C to stop.");
      await new Promise<void>((resolve) => {
        process.once("SIGINT", resolve);
        process.once("SIGTERM", resolve);
      });
    }
  } finally {
    journal.recordBotEvent("stopped");
    await alerts.send(formatBotStoppedAlert(config));
    journal.close();
    dashboard.stop();
    await broker.disconnect();
  }
}

if (require.main === module) {
  main().catch((error) => {
    logger.error("Bot failed", { message: error instanceof Error ? error.message : String(error) });
    process.exitCode = 1;
  });
}
