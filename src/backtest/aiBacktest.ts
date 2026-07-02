import * as fs from "fs";
import { AppConfig, Candle, TradeSignal } from "../types";
import { AiPrediction, buildAiSignal } from "../strategy/aiSignalEngine";
import { buildFinalDecision, SignalDecision } from "../strategy/signalEngine";

/**
 * Offline AI backtesting: consumes predictions precomputed by
 * scripts/ai_backtest_predict.py (data/ai_predictions.csv) so the backtest
 * needs no MT5 connection and no per-candle Python calls.
 */

export function loadAiPredictions(path: string): Map<number, AiPrediction> {
  const map = new Map<number, AiPrediction>();
  const lines = fs.readFileSync(path, "utf-8").trim().split(/\r?\n/);
  for (const line of lines.slice(1)) {
    const [time, direction, confidence] = line.split(",");
    map.set(Number(time), {
      direction: direction as AiPrediction["direction"],
      confidence: Number(confidence)
    });
  }
  return map;
}

function rejected(score: number, config: AppConfig, reasons: string[]): SignalDecision {
  return {
    status: "rejected",
    score,
    reasons,
    finalDecision: buildFinalDecision({ score, config, action: "reject" })
  };
}

/**
 * Signal provider for runBacktest(): looks up the AI prediction for the
 * current candle and builds a trade using the same SL/TP logic as live AI
 * mode (buildAiSignal). Candles without a prediction are skipped, which is
 * how the holdout window is enforced.
 */
export function makeAiSignalProvider(predictions: Map<number, AiPrediction>) {
  return (
    entryWindow: Candle[],
    _trendWindow: Candle[],
    _higherTrendWindow: Candle[],
    spread: number,
    config: AppConfig
  ): SignalDecision => {
    const latest = entryWindow.at(-1);
    if (!latest) return rejected(0, config, ["No candles"]);

    const prediction = predictions.get(latest.time);
    if (!prediction) return rejected(0, config, ["No AI prediction for this candle"]);

    const result = buildAiSignal(prediction, entryWindow, spread, config);
    if (result.status !== "trade" || !result.direction) {
      return rejected(result.score, config, result.reasons);
    }

    const signal: TradeSignal = {
      symbol: config.symbol,
      direction: result.direction,
      entry: result.entry!,
      stopLoss: result.stopLoss!,
      takeProfits: result.takeProfits!,
      score: result.score,
      reasons: result.reasons,
      timestamp: latest.time,
      entryType: "market",
      tpMode: "r"
    };

    return {
      status: "trade",
      score: result.score,
      reasons: result.reasons,
      finalDecision: buildFinalDecision({
        direction: result.direction,
        score: result.score,
        config,
        action: "trade"
      }),
      signal
    };
  };
}

/** First candle index that has a prediction — start both backtests there. */
export function firstPredictionIndex(candles: Candle[], predictions: Map<number, AiPrediction>): number {
  for (let i = 0; i < candles.length; i += 1) {
    if (predictions.has(candles[i].time)) return i;
  }
  return -1;
}
