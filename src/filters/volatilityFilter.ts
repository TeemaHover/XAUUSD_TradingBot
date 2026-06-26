import { atr } from "../indicators/atr";
import { AppConfig, Candle, DetectorResult } from "../types";

export function volatilityFilter(candles: Candle[], spread: number, config: AppConfig): DetectorResult {
  const atrValues = atr(candles, config.strategy.atrLength);
  const latestAtr = atrValues.at(-1) ?? 0;
  const atrValid = latestAtr >= config.strategy.minAtr;
  const spreadValid = spread <= config.strategy.maxSpread;
  const valid = atrValid && spreadValid;

  return {
    score: valid ? config.scoring.volatilityValid : 0,
    reasons: [
      valid
        ? `Volatility valid: ATR=${latestAtr.toFixed(2)}, spread=${spread.toFixed(2)}`
        : `Volatility rejected: ATR=${latestAtr.toFixed(2)}, spread=${spread.toFixed(2)}`
    ]
  };
}
