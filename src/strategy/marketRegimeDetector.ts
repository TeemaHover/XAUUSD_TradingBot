import { atr } from "../indicators/atr";
import { adx } from "../indicators/adx";
import { AppConfig, Candle, MarketRegimeResult } from "../types";

export function detectMarketRegime(candles: Candle[], config: AppConfig): MarketRegimeResult {
  if (candles.length < Math.max(config.regime.lookback, config.regime.adxLength + 5)) {
    return {
      regime: "ranging",
      atr: 0,
      averageAtr: 0,
      adx: 0,
      score: 0,
      reasons: ["Not enough candles for market regime"]
    };
  }

  const atrValues = atr(candles, config.strategy.atrLength);
  const latestAtr = atrValues.at(-1) ?? 0;
  const atrWindow = atrValues.slice(-config.regime.lookback);
  const averageAtr = atrWindow.reduce((sum, value) => sum + value, 0) / atrWindow.length;
  const adxValue = adx(candles, config.regime.adxLength).at(-1) ?? 0;

  let regime: MarketRegimeResult["regime"] = "ranging";
  if (averageAtr > 0 && latestAtr >= averageAtr * config.regime.highVolatilityAtrMultiplier) {
    regime = "highVolatility";
  } else if (averageAtr > 0 && latestAtr <= averageAtr * config.regime.lowVolatilityAtrMultiplier) {
    regime = "lowVolatility";
  } else if (adxValue >= config.regime.trendAdxThreshold) {
    regime = "trending";
  } else if (adxValue <= config.regime.rangeAdxThreshold) {
    regime = "ranging";
  }

  return {
    regime,
    atr: latestAtr,
    averageAtr,
    adx: adxValue,
    score: 0,
    reasons: [`Market regime: ${regime} ADX=${adxValue.toFixed(1)} ATR=${latestAtr.toFixed(2)}`]
  };
}
