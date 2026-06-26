import { AppConfig, Candle, VolumeResult } from "../types";

export function analyzeVolume(candles: Candle[], config: AppConfig, lookback = 20): VolumeResult {
  const latest = candles.at(-1);
  if (!latest || candles.length < lookback) {
    return { averageVolume: 0, relativeVolume: 0, spike: false, score: 0, reasons: ["Not enough candles for volume"] };
  }

  const previous = candles.slice(-lookback - 1, -1);
  const averageVolume = previous.reduce((sum, candle) => sum + candle.volume, 0) / previous.length;
  const relativeVolume = averageVolume > 0 ? latest.volume / averageVolume : 0;
  const spike = relativeVolume >= config.strategy.volumeSpikeMultiplier;

  return {
    averageVolume,
    relativeVolume,
    spike,
    score: spike ? config.scoring.volumeConfirmation : Math.round(config.scoring.volumeConfirmation * Math.min(relativeVolume, 1)),
    reasons: [spike ? `Volume spike: ${relativeVolume.toFixed(2)}x` : `Relative volume: ${relativeVolume.toFixed(2)}x`]
  };
}
