import { runBacktest } from "./backtestEngine";
import { runWalkForward } from "./walkForward";
import { loadCandlesFromCsv } from "./csv";
import { loadConfig } from "../config/loadConfig";
import { logger } from "../logger/logger";

const csvPath = process.argv[2] ?? "data/xauusd.csv";
const outputPath = process.argv[3] ?? "backtest-results.json";
const config = loadConfig();
const candles = loadCandlesFromCsv(csvPath);

if (candles.length === 0) {
  logger.error("No candles loaded", { csvPath });
  process.exitCode = 1;
} else {
  if (process.argv.includes("--walk-forward")) {
    runWalkForward(candles, config, outputPath);
  } else {
    runBacktest(candles, config, outputPath);
  }
}
