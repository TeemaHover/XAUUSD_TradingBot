import { BacktestResult, runBacktest } from "./backtestEngine";
import { firstPredictionIndex, loadAiPredictions, makeAiSignalProvider } from "./aiBacktest";
import { runWalkForward } from "./walkForward";
import { loadCandlesFromCsv } from "./csv";
import { closedUpTo, resampleCandles, TIMEFRAME_MS } from "./resample";
import { loadConfig } from "../config/loadConfig";
import { logger } from "../logger/logger";
import { calculateMtfSignal, mtfSettings } from "../strategy/mtfCascade";
import { AppConfig, Candle } from "../types";

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

  if (process.argv.includes("--sweep")) {
    sweepConfidence(predictions, startIndex);
    return;
  }

  const ai = runBacktest(candles, config, "backtest-ai.json", {
    startIndex,
    signalProvider: makeAiSignalProvider(predictions)
  });

  if (process.argv.includes("--ai-only")) {
    /* eslint-disable no-console */
    console.log("\n=== AI BACKTEST (rules comparison skipped: --ai-only) ===");
    console.log(tableHeader);
    console.log(row("AI (CNN predictions)", ai));
    console.log("Details: backtest-ai.json");
    /* eslint-enable no-console */
    return;
  }

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

/**
 * Confidence threshold sweep: rerun the AI backtest at a range of
 * aiConfidenceThreshold values (same predictions, same window, same costs).
 * Shows the trade count / expectancy trade-off so the threshold is chosen
 * from evidence instead of a guess. Run: ... --ai [predictions.csv] --sweep
 */
function sweepConfidence(
  predictions: ReturnType<typeof loadAiPredictions>,
  startIndex: number
): void {
  const thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70];

  /* eslint-disable no-console */
  console.log("\n=== AI CONFIDENCE THRESHOLD SWEEP (same holdout, same costs) ===");
  console.log(tableHeader);
  for (const threshold of thresholds) {
    const swept = JSON.parse(JSON.stringify(config)) as AppConfig;
    swept.strategy.aiConfidenceThreshold = threshold;
    const result = runBacktest(candles, swept, "backtest-ai-sweep.json", {
      startIndex,
      signalProvider: makeAiSignalProvider(predictions)
    });
    console.log(row(`threshold ${threshold.toFixed(2)}`, result));
  }
  console.log("\nPick by EXPECTANCY with enough trades to matter (>50 ideally).");
  console.log("A threshold that only leaves a handful of trades is curve-fitting.");
  /* eslint-enable no-console */
}

/**
 * MTF cascade backtest: resample the 5m series to 1h/4h/1d and evaluate the
 * top-down cascade at every bar, feeding each layer only candles that were
 * CLOSED at that moment (no higher-timeframe lookahead).
 */
function runMtf(): void {
  const settings = mtfSettings(config);
  const h1 = resampleCandles(candles, "1h");
  const h4 = resampleCandles(candles, "4h");
  const d1 = resampleCandles(candles, "1d");
  logger.warn(`MTF backtest: ${candles.length.toLocaleString()} 5m candles -> ${h1.length} h1, ${h4.length} h4, ${d1.length} d1`);

  // Warmup: the daily bias needs dailyEmaLength + swing lookback closed days
  const neededDays = settings.dailyEmaLength + config.strategy.swingLookback * 2;
  if (d1.length <= neededDays) {
    logger.error(`Not enough daily candles for the cascade (${d1.length} <= ${neededDays})`);
    process.exitCode = 1;
    return;
  }
  const warmupCutoff = d1[neededDays].time + TIMEFRAME_MS["1d"];
  let startIndex = candles.findIndex((c) => c.time + TIMEFRAME_MS["5m"] >= warmupCutoff);
  if (startIndex < 0) startIndex = candles.length;
  logger.warn(`MTF warmup: ${neededDays} daily candles -> starting at 5m index ${startIndex.toLocaleString()}`);

  // One idea = one trade: after a signal, suppress re-entries for cooldownBars
  // (the cascade's conditions persist for hours — without this, every 5m bar
  // at the zone would fire another overlapping trade).
  let lastSignalIndex = -Infinity;
  const provider = (
    entryWindow: Candle[],
    _trend: Candle[],
    _higher: Candle[],
    spread: number,
    cfg: AppConfig,
    index: number
  ) => {
    if (index - lastSignalIndex < settings.cooldownBars) {
      return calculateMtfSignal([], [], [], [], spread, cfg); // cheap "not enough candles" reject
    }
    const cutoff = entryWindow.at(-1)!.time + TIMEFRAME_MS["5m"];
    const decision = calculateMtfSignal(
      entryWindow.slice(-400),
      closedUpTo(h1, "1h", cutoff, 400),
      closedUpTo(h4, "4h", cutoff, 400),
      closedUpTo(d1, "1d", cutoff, 400),
      spread,
      cfg
    );
    if (decision.signal) lastSignalIndex = index;
    return decision;
  };

  const result = runBacktest(candles, config, outputPath, {
    startIndex,
    maxHoldBars: settings.maxHoldBars,
    signalProvider: provider
  });

  /* eslint-disable no-console */
  console.log("\n=== MTF CASCADE BACKTEST ===");
  console.log(tableHeader);
  console.log(row("MTF cascade", result));
  console.log(`\nSettings: minRR=${settings.minRR}, dailyEMA=${settings.dailyEmaLength}, maxHold=${settings.maxHoldBars} bars`);
  console.log(`Details: ${outputPath}`);
  /* eslint-enable no-console */
}

if (candles.length === 0) {
  logger.error("No candles loaded", { csvPath });
  process.exitCode = 1;
} else if (process.argv.includes("--mtf")) {
  runMtf();
} else if (process.argv.includes("--ai")) {
  compareAi();
} else if (process.argv.includes("--walk-forward")) {
  runWalkForward(candles, config, outputPath);
} else if (process.argv.includes("--compare-entries")) {
  compareEntries();
} else {
  runBacktest(candles, config, outputPath);
}
