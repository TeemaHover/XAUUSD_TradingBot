import { MockBroker } from "./broker/MockBroker";
import { Mt5Broker } from "./broker/Mt5Broker";
import { Broker } from "./broker/Broker";
import { loadConfig } from "./config/loadConfig";
import { sampleCandles } from "./data/sampleCandles";
import { DashboardServer, DashboardState } from "./dashboard/dashboardServer";
import { ExecutionEngine } from "./execution/executionEngine";
import { logger } from "./logger/logger";
import { RiskGuard } from "./risk/positionSizing";
import { calculateSignal } from "./strategy/signalEngine";
import { AppConfig, Candle } from "./types";
import { TelegramAlerts } from "./alerts/telegram";
import { SqliteJournal } from "./journal/sqliteJournal";

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
  if (config.broker.mode === "mt5") {
    const broker = new Mt5Broker(config);
    await broker.connect();
    return broker;
  }

  const entryCandles = sampleCandles(320);
  const trendCandles = sampleCandles(260);
  const higherTrendCandles = sampleCandles(260);
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
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

export function formatBotStartedAlert(config: AppConfig, balance: number): string {
  return [
    "Bot started",
    `Symbol: ${config.symbol}`,
    `Broker: ${config.broker.mode.toUpperCase()}`,
    `Loop: ${config.bot.loopEnabled ? `ON every ${config.bot.intervalSeconds}s` : "OFF"}`,
    `Timeframes: entry=${config.timeframes.entry}, trend=${config.timeframes.trend}, higherTrend=${config.timeframes.higherTrend}`,
    `Balance: ${balance}`,
    `Mode: ${config.mt5.dryRun ? "DRY_RUN" : "DEMO/LIVE ORDERING ENABLED"}`
  ].join("\n");
}

export function formatBotStoppedAlert(config: AppConfig): string {
  return [
    "Bot stopped",
    `Symbol: ${config.symbol}`,
    `Broker: ${config.broker.mode.toUpperCase()}`,
    `Stopped at: ${new Date().toISOString()}`
  ].join("\n");
}

async function scanOnce(
  broker: Broker,
  config: AppConfig,
  execution: ExecutionEngine,
  dashboardState: DashboardState,
  journal: SqliteJournal
): Promise<void> {
  const { entryCandles, trendCandles, higherTrendCandles } = await fetchMarketSnapshot(broker, config);
  const spread = await broker.getSpread(config.symbol);
  const decision = calculateSignal(entryCandles, trendCandles, higherTrendCandles, spread, config);
  const action = signalAction(decision);
  dashboardState.lastDecision = decision;
  journal.recordSignal(decision);

  logger.info("Signal calculation", {
    action,
    status: decision.status,
    score: decision.score,
    finalDecision: decision.finalDecision,
    reasons: decision.reasons
  });

  if (decision.signal) {
    dashboardState.lastSignal = decision.signal;
    await execution.execute(decision.signal);
  }
}

async function main(): Promise<void> {
  const config = loadConfig();
  const broker = await createBroker(config);
  const startingBalance = await broker.getBalance();
  const riskGuard = new RiskGuard(config.risk.maxDailyLoss, config.risk.maxConsecutiveLosses, startingBalance);
  const journal = new SqliteJournal(config);
  journal.init();
  const execution = new ExecutionEngine(broker, config, riskGuard, journal);
  const dashboardState: DashboardState = { dailyPnl: 0 };
  const dashboard = new DashboardServer(config, broker, dashboardState);
  const alerts = new TelegramAlerts(config);
  let stopping = false;

  dashboard.start();
  process.once("SIGINT", () => { stopping = true; });
  process.once("SIGTERM", () => { stopping = true; });
  journal.recordBotEvent("started", {
    broker: config.broker.mode,
    loopEnabled: config.bot.loopEnabled,
    intervalSeconds: config.bot.intervalSeconds,
    balance: startingBalance
  });
  await alerts.send(formatBotStartedAlert(config, startingBalance));

  try {
    await scanOnce(broker, config, execution, dashboardState, journal);

    while (config.bot.loopEnabled && !stopping) {
      await sleep(config.bot.intervalSeconds * 1000);
      if (stopping) break;
      try {
        await scanOnce(broker, config, execution, dashboardState, journal);
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
