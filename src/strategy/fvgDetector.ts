import { AppConfig, Candle, FvgResult, Zone } from "../types";

function filledPercent(candles: Candle[], zone: Zone, direction: "bullish" | "bearish"): number {
  const size = zone.high - zone.low;
  if (size <= 0) return 100;
  const after = candles.filter((candle) => (zone.time ? candle.time > zone.time : true));
  if (after.length === 0) return 0;

  if (direction === "bullish") {
    const deepest = Math.min(...after.map((candle) => candle.low));
    return Math.max(0, Math.min(100, ((zone.high - deepest) / size) * 100));
  }

  const highest = Math.max(...after.map((candle) => candle.high));
  return Math.max(0, Math.min(100, ((highest - zone.low) / size) * 100));
}

export function detectFvg(candles: Candle[], config: AppConfig): FvgResult {
  if (candles.length < 3) return { score: 0, reasons: ["Not enough candles for FVG"] };

  let bullish: (Zone & { filledPercent: number }) | undefined;
  let bearish: (Zone & { filledPercent: number }) | undefined;
  const recent = candles.slice(-20);

  for (let i = 2; i < recent.length; i += 1) {
    const first = recent[i - 2];
    const third = recent[i];
    if (first.high < third.low) {
      const zone = { low: first.high, high: third.low, strength: 60, time: third.time };
      bullish = { ...zone, filledPercent: filledPercent(candles, zone, "bullish") };
    }
    if (first.low > third.high) {
      const zone = { low: third.high, high: first.low, strength: 60, time: third.time };
      bearish = { ...zone, filledPercent: filledPercent(candles, zone, "bearish") };
    }
  }

  const bestFill = Math.min(bullish?.filledPercent ?? 100, bearish?.filledPercent ?? 100);
  const found = Boolean(bullish || bearish);
  const usable = found && bestFill < 80;
  return {
    bullish,
    bearish,
    score: usable ? config.scoring.fairValueGap : found ? Math.round(config.scoring.fairValueGap * 0.3) : 0,
    reasons: [
      bullish ? `Bullish FVG filled ${bullish.filledPercent.toFixed(0)}%` : "",
      bearish ? `Bearish FVG filled ${bearish.filledPercent.toFixed(0)}%` : ""
    ].filter(Boolean)
  };
}
