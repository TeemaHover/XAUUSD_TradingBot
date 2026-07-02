import { BacktestResult, runBacktest } from "./backtestEngine";
import { runWalkForward } from "./walkForward";
import { loadCandlesFromCsv } from "./csv";
import { loadConfig } from "../config/loadConfig";
import { logger } from "../logger/logger";
import { AppConfig } from "../types";

const csvPath = process.argv[2] ?? "data/xauusd.csv";
const outputPath = process.argv[3] ?? "backtest-results.json";
const config = loadConfig();
const candles = loadCandlesFromCsv(csvPath);

function withEntryMode(base: AppConfig, entryMode: "market" | "zone"): AppConfig {
  const clone = JSON.parse(JSON.stringify(base)) as AppConfig;
  clone.strategy.entryMode = entryMode;
  return clone;
}

function row(label: string, result: BacktestResult): string {
  return [
    label.padEnd(22),
    String(result.totalTrades).padStart(7),
    (result.winRate * 100).toFixed(1).padStart(8) + "%",
    result.profitFactor.toFixed(2).padStart(8),
    result.expectancy.toFixed(3).padStart(9) + "R",
    (result.maxDrawdown * 100).toFixed(1).padStart(8) + "%",
    (result.totalReturn * 100).toFixed(1).padStart(9) + "%",
    String(result.performanceSummary.canceledLimitOrders).padStart(9)
  ].join("  ");
}

function compareEntries(): void {
  logger.info("Running entry-mode comparison: market vs zone");
  const market = runBacktest(candles, withEntryMode(config, "market"), "backtest-market.json");
  const zone = runBacktest(candles, withEntryMode(config, "zone"), "backtest-zone.json");

  const header = [
    "".padEnd(22),
    "trades".padStart(7),
    "winRate".padStart(9),
    "PF".padStart(8),
    "expect".padStart(10),
    "maxDD".padStart(9),
    "return".padStart(10),
    "canceled".padStart(9)
  ].join("  ");

  /* eslint-disable no-console */
  console.log("\n=== ENTRY MODE COMPARISON ===");
  console.log(header);
  console.log(row("market (confirmation)", market));
  console.log(row("zone (sniper limit)", zone));
  console.log("\nCompare EXPECTANCY (avg R/trade) and maxDD, not win rate alone.");
  console.log("Details: backtest-market.json / backtest-zone.json");
  /* eslint-enable no-console */
}

if (candles.length === 0) {
  logger.error("No candles loaded", { csvPath });
  process.exitCode = 1;
} else if (process.argv.includes("--walk-forward")) {
  runWalkForward(candles, config, outputPath);
} else if (process.argv.includes("--compare-entries")) {
  compareEntries();
} else {
  runBacktest(candles, config, outputPath);
}
