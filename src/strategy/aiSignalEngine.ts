import { AppConfig, Candle } from "../types";
import { Mt5Broker } from "../broker/Mt5Broker";
import { detectSR } from "./srDetector";
import { logger } from "../logger/logger";

export interface AiPrediction {
  direction: "long" | "short" | "hold";
  confidence: number;
  reason?: string;
}

export interface AiSignalResult {
  status: "trade" | "rejected";
  direction?: "long" | "short";
  entry?: number;
  stopLoss?: number;
  takeProfits?: number[];
  score: number;
  confidence: number;
  reasons: string[];
}

/**
 * Call the Python AI bridge to get a prediction for the current candles.
 * Falls back to "hold" if the model file doesn't exist yet.
 */
export async function aiPredict(
  broker: Mt5Broker,
  candles: Candle[],
  modelPath: string
): Promise<AiPrediction> {
  try {
    // The bridge detects model type (single-TF vs multi-TF) from the .npz file
    // and fetches the required candles from MT5 directly for multi-TF models.
    // We still pass candles as a fallback for single-TF models.
    const symbol = (broker as any).config?.symbol ?? "GOLD";
    const result = await (broker as any).call("ai_predict", {
      symbol,
      modelPath,
      candles: candles.map((c) => ({
        time: c.time,
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
        volume: c.volume ?? 1
      }))
    });
    return result as AiPrediction;
  } catch (err) {
    logger.warn("AI predict failed, returning hold", {
      message: err instanceof Error ? err.message : String(err)
    });
    return { direction: "hold", confidence: 0, reason: "bridge error" };
  }
}

/**
 * Build a TradeSignal from an AI prediction.
 * Uses S/R zone boundary for SL, ATR-based TPs.
 */
export function buildAiSignal(
  prediction: AiPrediction,
  candles: Candle[],
  spread: number,
  config: AppConfig
): AiSignalResult {
  const threshold = config.strategy.aiConfidenceThreshold ?? 0.5;
  const reasons: string[] = [
    "[AI mode] Neural network prediction",
    "Direction: " + prediction.direction.toUpperCase() + " | Confidence: " + (prediction.confidence * 100).toFixed(1) + "%",
    "Threshold: " + (threshold * 100).toFixed(0) + "%"
  ];

  if (prediction.direction === "hold") {
    return {
      status: "rejected",
      score: 0,
      confidence: prediction.confidence,
      reasons: [...reasons, "AI says HOLD — no trade"]
    };
  }

  if (prediction.confidence < threshold) {
    return {
      status: "rejected",
      score: 0,
      confidence: prediction.confidence,
      reasons: [...reasons, "Confidence " + (prediction.confidence * 100).toFixed(1) + "% below threshold " + (threshold * 100).toFixed(0) + "%"]
    };
  }

  const latest = candles.at(-1)!;
  const direction = prediction.direction;
  const entry = direction === "long"
    ? latest.close + spread / 2
    : latest.close - spread / 2;

  // ATR for SL distance
  const atrValues: number[] = [];
  for (let i = 1; i < Math.min(candles.length, 15); i++) {
    const c = candles[candles.length - i];
    const p = candles[candles.length - i - 1];
    atrValues.push(Math.max(
      c.high - c.low,
      Math.abs(c.high - p.close),
      Math.abs(c.low - p.close)
    ));
  }
  const atrVal = atrValues.length > 0
    ? atrValues.reduce((a, b) => a + b, 0) / atrValues.length
    : 1;

  // SL: try S/R zone boundary first, fall back to 1.5x ATR
  const sr = detectSR(candles, config);
  let stopLoss: number;

  if (direction === "long" && sr.supportZones.length > 0) {
    const nearest = sr.supportZones
      .filter((z) => z.high < entry)
      .sort((a, b) => b.high - a.high)[0];
    stopLoss = nearest
      ? nearest.low - atrVal * config.risk.stopBufferAtr
      : entry - atrVal * 1.5;
  } else if (direction === "short" && sr.resistanceZones.length > 0) {
    const nearest = sr.resistanceZones
      .filter((z) => z.low > entry)
      .sort((a, b) => a.low - b.low)[0];
    stopLoss = nearest
      ? nearest.high + atrVal * config.risk.stopBufferAtr
      : entry + atrVal * 1.5;
  } else {
    stopLoss = direction === "long"
      ? entry - atrVal * 1.5
      : entry + atrVal * 1.5;
  }

  const risk = Math.abs(entry - stopLoss);
  if (risk < config.risk.minStopDistance) {
    return {
      status: "rejected",
      score: 0,
      confidence: prediction.confidence,
      reasons: [...reasons, "Stop distance " + risk.toFixed(2) + " below minimum"]
    };
  }

  // TPs: 1R, 2R, 3R
  const tps = [1, 2, 3].map((mult) =>
    direction === "long" ? entry + risk * mult : entry - risk * mult
  );

  const score = Math.round(prediction.confidence * 100);
  reasons.push(
    "Entry: " + entry.toFixed(2),
    "SL: " + stopLoss.toFixed(2) + " (risk: " + risk.toFixed(2) + ")",
    "TPs: " + tps.map((t) => t.toFixed(2)).join(" / ")
  );

  return {
    status: "trade",
    direction,
    entry,
    stopLoss,
    takeProfits: tps,
    score,
    confidence: prediction.confidence,
    reasons
  };
}
