import { spawn } from "node:child_process";
import path from "node:path";
import { Candle } from "../types";
import { logger } from "../logger/logger";

export interface HmmRegimeResult {
  state: number;
  label: string;
  confidence: number;
  probs: number[];
  reason?: string;
}

const UNKNOWN: HmmRegimeResult = { state: -1, label: "unknown", confidence: 0, probs: [] };

function resolvePython(configuredPath?: string): string {
  if (configuredPath && configuredPath !== "python") return configuredPath;
  return process.platform === "win32" ? "python" : "python3";
}

/**
 * Classify the current market regime with the trained HMM by running
 * scripts/hmm_predict_live.py. Display-only: fails soft to "unknown" so a
 * missing model or python problem can never break the trading loop.
 */
export async function hmmPredict(
  candles: Candle[],
  modelPath: string,
  pythonPath?: string
): Promise<HmmRegimeResult> {
  const scriptPath = path.resolve("scripts/hmm_predict_live.py");
  const payload = JSON.stringify({
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

  try {
    const stdout = await new Promise<string>((resolve, reject) => {
      const child = spawn(resolvePython(pythonPath), [scriptPath], { cwd: process.cwd() });
      let out = "";
      let err = "";
      const timer = setTimeout(() => {
        child.kill();
        reject(new Error("HMM predictor timed out after 15s"));
      }, 15000);
      child.stdout.on("data", (chunk) => { out += chunk; });
      child.stderr.on("data", (chunk) => { err += chunk; });
      child.on("error", (error) => { clearTimeout(timer); reject(error); });
      child.on("close", (code) => {
        clearTimeout(timer);
        if (code === 0 && out.trim()) resolve(out);
        else reject(new Error(err.trim() || `hmm predictor exited with code ${code}`));
      });
      child.stdin.write(payload);
      child.stdin.end();
    });

    const result = JSON.parse(stdout.trim()) as HmmRegimeResult;
    if (result.reason) {
      logger.warn("HMM regime unavailable", { reason: result.reason });
    }
    return result;
  } catch (err) {
    logger.warn("HMM regime predict failed", {
      message: err instanceof Error ? err.message : String(err)
    });
    return UNKNOWN;
  }
}
