import { spawn } from "node:child_process";
import path from "node:path";
import { AppConfig, Candle } from "../types";
import { detectSR } from "./srDetector";
import { detectFvg } from "./fvgDetector";
import { detectTrend } from "./trendDetector";
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
  entryType?: "market" | "limit";
  score: number;
  confidence: number;
  reasons: string[];
}

function toPlainCandles(candles: Candle[]): Array<Record<string, number>> {
  return candles.map((c) => ({
    time: c.time,
    open: c.open,
    high: c.high,
    low: c.low,
    close: c.close,
    volume: c.volume ?? 1
  }));
}

function resolvePython(configuredPath?: string): string {
  if (configuredPath && configuredPath !== "python") return configuredPath;
  return process.platform === "win32" ? "python" : "python3";
}

/**
 * Get an AI prediction for the current candles by running the local Python
 * predictor (scripts/ai_predict_live.py). Broker-agnostic: candles come from
 * whichever broker is active (mock, mt5, or metaapi). Only needs numpy.
 * Falls back to "hold" on any failure.
 */
export async function aiPredict(
  candleSets: Record<string, Candle[]>,
  modelPath: string,
  pythonPath?: string
): Promise<AiPrediction> {
  const scriptPath = path.resolve("scripts/ai_predict_live.py");
  const payload = JSON.stringify({
    modelPath,
    candles: Object.fromEntries(
      Object.entries(candleSets).map(([tf, candles]) => [tf, toPlainCandles(candles)])
    )
  });

  try {
    const stdout = await new Promise<string>((resolve, reject) => {
      const child = spawn(resolvePython(pythonPath), [scriptPath], { cwd: process.cwd() });
      let out = "";
      let err = "";
      const timer = setTimeout(() => {
        child.kill();
        reject(new Error("AI predictor timed out after 30s"));
      }, 30000);
      child.stdout.on("data", (chunk) => { out += chunk; });
      child.stderr.on("data", (chunk) => { err += chunk; });
      child.on("error", (error) => { clearTimeout(timer); reject(error); });
      child.on("close", (code) => {
        clearTimeout(timer);
        if (code === 0 && out.trim()) resolve(out);
        else reject(new Error(err.trim() || `predictor exited with code ${code}`));
      });
      child.stdin.write(payload);
      child.stdin.end();
    });

    const result = JSON.parse(stdout.trim()) as AiPrediction;
    if (result.reason) {
      logger.warn("AI predictor returned hold", { reason: result.reason });
    }
    return result;
  } catch (err) {
    logger.warn("AI predict failed, returning hold", {
      message: err instanceof Error ? err.message : String(err)
    });
    return { direction: "hold", confidence: 0, reason: "predictor error" };
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
  config: AppConfig,
  higherTrendCandles?: Candle[],
  trendCandles?: Candle[]
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

  // Higher-timeframe trend gate: the model has a long bias, so refuse trades
  // that fight the 4h trend (longs in a downtrend, shorts in an uptrend).
  if ((config.strategy.aiTrendFilter ?? true) && higherTrendCandles && higherTrendCandles.length > 0) {
    const trend = detectTrend(higherTrendCandles, config);
    const fightsTrend =
      (prediction.direction === "long" && trend.bias === "bearish") ||
      (prediction.direction === "short" && trend.bias === "bullish");
    if (fightsTrend) {
      return {
        status: "rejected",
        score: 0,
        confidence: prediction.confidence,
        reasons: [...reasons, `Trend filter: ${prediction.direction.toUpperCase()} against ${trend.bias} higher-TF trend — no trade`]
      };
    }
    reasons.push(`Trend filter: higher-TF trend is ${trend.bias} — OK`);
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
  let entry = direction === "long"
    ? latest.close + spread / 2
    : latest.close - spread / 2;
  let entryType: "market" | "limit" = "market";

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

  // FVG pullback entry: instead of chasing at market, park a limit order in
  // an unfilled fair value gap so the pullback comes to us (better price,
  // tighter stop). Gaps are searched on all available timeframes, preferring
  // higher ones (stronger zones) as long as the entry is realistically
  // reachable (within maxLimitDistanceAtr of current price).
  // Falls back to market entry when no usable gap exists.
  let fvgZone: { low: number; high: number } | undefined;
  if ((config.strategy.aiEntryMode ?? "market") === "fvg") {
    const maxReach = atrVal * 3; // limit orders farther than this rarely fill before expiry
    const timeframeSets: Array<{ label: string; set: Candle[] | undefined }> = [
      { label: "4h", set: higherTrendCandles },
      { label: "1h", set: trendCandles },
      { label: "5m", set: candles }
    ];

    for (const { label, set } of timeframeSets) {
      if (!set || set.length < 3) continue;
      const fvg = detectFvg(set, config);
      const zone = direction === "long" ? fvg.bullish : fvg.bearish;
      if (!zone || zone.filledPercent >= 80) continue;

      const zoneEntry = (zone.high + zone.low) / 2; // midpoint of the gap
      const improves = direction === "long" ? zoneEntry < entry : zoneEntry > entry;
      const reachable = Math.abs(zoneEntry - latest.close) <= maxReach;
      if (improves && reachable) {
        entry = zoneEntry;
        entryType = "limit";
        fvgZone = zone;
        reasons.push(`FVG pullback entry (${label}): limit at gap midpoint (${zone.low.toFixed(2)}-${zone.high.toFixed(2)}, ${zone.filledPercent.toFixed(0)}% filled)`);
        break; // highest usable timeframe wins
      }
    }
    if (!fvgZone) {
      reasons.push("No reachable unfilled FVG on 4h/1h/5m — market entry");
    }
  }

  // SL: FVG boundary if entering on a gap, else S/R zone boundary, else 1.5x ATR
  const sr = detectSR(candles, config);
  let stopLoss: number;

  if (fvgZone) {
    stopLoss = direction === "long"
      ? fvgZone.low - atrVal * config.risk.stopBufferAtr
      : fvgZone.high + atrVal * config.risk.stopBufferAtr;
  } else if (direction === "long" && sr.supportZones.length > 0) {
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

  // Cap the stop: a far-away S/R zone must not create an oversized SL,
  // because TPs are R-multiples and would drift out of reach with it.
  const maxStop = atrVal * (config.risk.maxStopAtr ?? 2.5);
  if (Math.abs(entry - stopLoss) > maxStop) {
    stopLoss = direction === "long" ? entry - maxStop : entry + maxStop;
    reasons.push(`SL capped at ${(config.risk.maxStopAtr ?? 2.5)}x ATR`);
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
    entryType,
    score,
    confidence: prediction.confidence,
    reasons
  };
}
