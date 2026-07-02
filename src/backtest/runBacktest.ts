import { BacktestResult, runBacktest } from "./backtestEngine";
import { firstPredictionIndex, loadAiPredictions, makeAiSignalProvider } from "./aiBacktest";
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

const tableHeader = [
  "".padEnd(22),
  "trades".padStart(7),
  "winRate".padStart(9),
  "PF".padStart(8),
  "expect".padStart(10),
  "maxDD".padStart(9),
  "return".padStart(10),
  "canceled".padStart(9)
].join("  ");

function compareEntries(): void {
  logger.info("Running entry-mode comparison: market vs zone");
  const market = runBacktest(candles, withEntryMode(config, "market"), "backtest-market.json");
  const zone = runBacktest(candles, withEntryMode(config, "zone"), "backtest-zone.json");

  /* eslint-disable no-console */
  console.log("\n=== ENTRY MODE COMPARISON ===");
  console.log(tableHeader);
  console.log(row("market (confirmation)", market));
  console.log(row("zone (sniper limit)", zone));
  console.log("\nCompare EXPECTANCY (avg R/trade) and maxDD, not win rate alone.");
  console.log("Details: backtest-market.json / backtest-zone.json");
  /* eslint-enable no-console */
}

/**
 * AI mode: backtest precomputed AI predictions (scripts/ai_backtest_predict.py)
 * against the rule engine over the SAME holdout window, same costs.
 */
function compareAi(): void {
  const flagIndex = process.argv.indexOf("--ai");
  const next = process.argv[flagIndex + 1];
  const predictionsPath = next && !next.startsWith("--") ? next : "data/ai_predictions.csv";

  logger.info("Loading AI predictions", { predictionsPath });
  const predictions = loadAiPredictions(predictionsPath);
  const startIndex = firstPredictionIndex(candles, predictions);
  if (startIndex < 0) {
    logger.error("No candle in the CSV matches any prediction timestamp", { predictionsPath });
    process.exitCode = 1;
    return;
  }
  logger.info("AI backtest window", {
    startIndex,
    candlesInWindow: candles.length - startIndex,
    predictions: predictions.size
  });

  const ai = runBacktest(candles, config, "backtest-ai.json", {
    startIndex,
    signalProvider: makeAiSignalProvider(predictions)
  });
  const rules = runBacktest(candles, config, "backtest-rules-holdout.json", { startIndex });

  /* eslint-disable no-console */
  console.log("\n=== AI vs RULES (same holdout window, same costs) ===");
  console.log(tableHeader);
  console.log(row("AI (CNN predictions)", ai));
  console.log(row("Rules (signal engine)", rules));
  console.log("\nJudge by EXPECTANCY (avg R/trade) and maxDD. If AI expectancy <= 0");
  console.log("or <= rules, the model is not adding edge yet — retrain with more");
  console.log("data / longer forward horizon before trusting it live.");
  console.log("Details: backtest-ai.json / backtest-rules-holdout.json");
  /* eslint-enable no-console */
}

if (candles.length === 0) {
  logger.error("No candles loaded", { csvPath });
  process.exitCode = 1;
} else if (process.argv.includes("--ai")) {
  compareAi();
} else if (process.argv.includes("--walk-forward")) {
  runWalkForward(candles, config, outputPath);
} else if (process.argv.includes("--compare-entries")) {
  compareEntries();
} else {
  runBacktest(candles, config, outputPath);
}
