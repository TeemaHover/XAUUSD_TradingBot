import { atr } from "../indicators/atr";
import { detectSwings } from "../market/swings";
import { AppConfig, Candle, LiquidityResult, Zone } from "../types";

function equalZones(prices: number[], tolerance: number): Zone[] {
  const zones: Zone[] = [];
  const sorted = [...prices].sort((a, b) => a - b);

  for (let i = 0; i < sorted.length - 1; i += 1) {
    if (Math.abs(sorted[i + 1] - sorted[i]) <= tolerance) {
      const low = Math.min(sorted[i], sorted[i + 1]);
      const high = Math.max(sorted[i], sorted[i + 1]);
      zones.push({ low, high, strength: 60 });
    }
  }

  return zones;
}

export function detectLiquidity(candles: Candle[], config: AppConfig): LiquidityResult {
  const swings = detectSwings(candles, config.strategy.swingLookback);
  const latest = candles.at(-1);
  if (!latest || swings.length < 2) {
    return { score: 0, reasons: ["Not enough swings for liquidity"], bullishSweep: false, bearishSweep: false, equalHighs: [], equalLows: [], swings };
  }

  const atrValues = atr(candles, config.strategy.atrLength);
  const tolerance = (atrValues.at(-1) ?? 0) * config.strategy.equalHighLowToleranceAtr;
  const swingHighs = swings.filter((swing) => swing.type === "high");
  const swingLows = swings.filter((swing) => swing.type === "low");
  const previousHigh = swingHighs.at(-1);
  const previousLow = swingLows.at(-1);
  const bullishSweep = Boolean(previousLow && latest.low < previousLow.price && latest.close > previousLow.price);
  const bearishSweep = Boolean(previousHigh && latest.high > previousHigh.price && latest.close < previousHigh.price);
  const equalHighs = equalZones(swingHighs.slice(-8).map((swing) => swing.price), tolerance);
  const equalLows = equalZones(swingLows.slice(-8).map((swing) => swing.price), tolerance);
  const hasSweep = bullishSweep || bearishSweep;

  return {
    score: hasSweep ? config.scoring.liquiditySweep : 0,
    reasons: [
      bullishSweep ? "Bullish liquidity sweep detected" : "",
      bearishSweep ? "Bearish liquidity sweep detected" : "",
      equalHighs.length > 0 ? `Equal highs detected: ${equalHighs.length}` : "",
      equalLows.length > 0 ? `Equal lows detected: ${equalLows.length}` : ""
    ].filter(Boolean),
    bullishSweep,
    bearishSweep,
    equalHighs,
    equalLows,
    swings
  };
}
