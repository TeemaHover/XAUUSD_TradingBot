import fs from "node:fs";
import path from "node:path";
import { AppConfig } from "../types";

/** Minimal .env loader (no dependency): KEY=VALUE lines, existing env wins. */
function loadDotEnv(envPath = ".env"): void {
  const resolved = path.resolve(envPath);
  if (!fs.existsSync(resolved)) return;
  for (const line of fs.readFileSync(resolved, "utf8").split(/\r?\n/)) {
    const match = /^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$/.exec(line);
    if (!match || line.trim().startsWith("#")) continue;
    const [, key, rawValue] = match;
    if (process.env[key] !== undefined) continue;
    process.env[key] = rawValue.replace(/^["']|["']$/g, "");
  }
}

export function loadConfig(configPath = "config/default.json"): AppConfig {
  loadDotEnv();
  const resolved = path.resolve(configPath);
  const raw = fs.readFileSync(resolved, "utf8");
  const config = JSON.parse(raw) as AppConfig;
  applyEnvOverrides(config);
  validateConfig(config);
  return config;
}

function assertPositive(value: number, name: string): void {
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`Invalid config: ${name} must be greater than zero`);
  }
}

function assertNonNegative(value: number, name: string): void {
  if (!Number.isFinite(value) || value < 0) {
    throw new Error(`Invalid config: ${name} must be zero or greater (0 = unlimited)`);
  }
}

function envBoolean(name: string): boolean | undefined {
  const value = process.env[name];
  if (value === undefined) return undefined;
  const normalized = value.trim().toLowerCase();
  if (["true", "1", "yes", "y"].includes(normalized)) return true;
  if (["false", "0", "no", "n"].includes(normalized)) return false;
  throw new Error(`Invalid env: ${name} must be true or false`);
}

function envNumber(name: string): number | undefined {
  const value = process.env[name];
  if (value === undefined) return undefined;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    throw new Error(`Invalid env: ${name} must be a number`);
  }
  return parsed;
}

function applyEnvOverrides(config: AppConfig): void {
  config.bot ??= { loopEnabled: false, intervalSeconds: 60 };

  const loopEnabled = envBoolean("BOT_LOOP_ENABLED");
  if (loopEnabled !== undefined) config.bot.loopEnabled = loopEnabled;

  const intervalSeconds = envNumber("BOT_INTERVAL_SECONDS");
  if (intervalSeconds !== undefined) config.bot.intervalSeconds = intervalSeconds;

  const oncePerCandle = envBoolean("BOT_EVALUATE_ONCE_PER_CANDLE");
  if (oncePerCandle !== undefined) config.bot.evaluateOncePerCandle = oncePerCandle;

  if (process.env.BROKER_MODE !== undefined) {
    config.broker.mode = process.env.BROKER_MODE.trim() as typeof config.broker.mode;
  }

  if (process.env.DASHBOARD_HOST !== undefined) {
    config.dashboard.host = process.env.DASHBOARD_HOST.trim();
  }
  if (process.env.DASHBOARD_PASSWORD !== undefined) {
    config.dashboard.password = process.env.DASHBOARD_PASSWORD;
  }

  const telegramEnabled = envBoolean("TELEGRAM_ENABLED");
  if (telegramEnabled !== undefined) config.alerts.telegram.enabled = telegramEnabled;
  if (process.env.TELEGRAM_BOT_TOKEN !== undefined) {
    config.alerts.telegram.botToken = process.env.TELEGRAM_BOT_TOKEN;
  }
  if (process.env.TELEGRAM_CHAT_ID !== undefined) {
    config.alerts.telegram.chatId = process.env.TELEGRAM_CHAT_ID;
  }
}

function validateConfig(config: AppConfig): void {
  if (!config.symbol) throw new Error("Invalid config: symbol is required");
  if (!config.bot) throw new Error("Invalid config: bot is required");
  assertPositive(config.bot.intervalSeconds, "bot.intervalSeconds");
  if (!config.journal) throw new Error("Invalid config: journal is required");
  if (!config.journal.path) throw new Error("Invalid config: journal.path is required");
  if (!["mock", "mt5", "metaapi", "auto"].includes(config.broker.mode)) {
    throw new Error("Invalid config: broker.mode must be mock, mt5, metaapi, or auto");
  }
  assertPositive(config.risk.riskPerTrade, "risk.riskPerTrade");
  assertNonNegative(config.risk.maxDailyLoss, "risk.maxDailyLoss");
  assertNonNegative(config.risk.maxConsecutiveLosses, "risk.maxConsecutiveLosses");
  assertPositive(config.risk.minStopDistance, "risk.minStopDistance");
  assertPositive(config.risk.tickSize, "risk.tickSize");
  assertPositive(config.risk.tickValue, "risk.tickValue");
  assertPositive(config.risk.volumeStep, "risk.volumeStep");
  assertPositive(config.risk.minVolume, "risk.minVolume");
  assertPositive(config.risk.maxVolume, "risk.maxVolume");
  if (config.risk.minVolume > config.risk.maxVolume) {
    throw new Error("Invalid config: risk.minVolume cannot exceed risk.maxVolume");
  }
  if (config.strategy.watchlistScore > config.strategy.minScore) {
    throw new Error("Invalid config: watchlistScore cannot exceed minScore");
  }
  if (config.strategy.counterTrendMinScore < config.strategy.minScore) {
    throw new Error("Invalid config: counterTrendMinScore cannot be below minScore");
  }
  if (config.tradeManagement.tpRMultiples.length === 0) {
    throw new Error("Invalid config: at least one TP multiple is required");
  }
  assertPositive(config.regime.adxLength, "regime.adxLength");
  assertPositive(config.regime.lookback, "regime.lookback");
  assertPositive(config.news.blackoutMinutesBefore, "news.blackoutMinutesBefore");
  assertPositive(config.news.blackoutMinutesAfter, "news.blackoutMinutesAfter");
  assertNonNegative(config.tradeGuards.cooldownAfterLossMinutes, "tradeGuards.cooldownAfterLossMinutes");
  assertNonNegative(config.tradeGuards.maxTradesPerSession, "tradeGuards.maxTradesPerSession");
  if (!Number.isFinite(config.sessions.utcOffsetMinutes)) {
    throw new Error("Invalid config: sessions.utcOffsetMinutes must be finite");
  }
  assertPositive(config.dashboard.port, "dashboard.port");
  assertPositive(config.walkForward.trainWindow, "walkForward.trainWindow");
  assertPositive(config.walkForward.testWindow, "walkForward.testWindow");
  assertPositive(config.walkForward.stepSize, "walkForward.stepSize");
  assertPositive(config.mt5.deviation, "mt5.deviation");
  assertPositive(config.mt5.magic, "mt5.magic");
  assertPositive(config.mt5.bars.entry, "mt5.bars.entry");
  assertPositive(config.mt5.bars.trend, "mt5.bars.trend");
  assertPositive(config.mt5.bars.higherTrend, "mt5.bars.higherTrend");
}
