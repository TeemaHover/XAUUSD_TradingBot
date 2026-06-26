import { ema } from "../indicators/ema";
import { detectSwings } from "../market/swings";
import { AppConfig, Candle, TrendResult } from "../types";

export function detectTrend(candles: Candle[], config: AppConfig): TrendResult {
  if (candles.length < config.strategy.emaLength) {
    return { bias: "sideways", confidence: 0, score: 0, reasons: ["Not enough candles for EMA trend"] };
  }

  const closes = candles.map((candle) => candle.close);
  const emaValues = ema(closes, config.strategy.emaLength);
  const latest = candles.at(-1)!;
  const latestEma = emaValues.at(-1)!;
  const swings = detectSwings(candles, config.strategy.swingLookback);
  const highs = swings.filter((swing) => swing.type === "high").slice(-2);
  const lows = swings.filter((swing) => swing.type === "low").slice(-2);

  const higherHighs = highs.length === 2 && highs[1].price > highs[0].price;
  const higherLows = lows.length === 2 && lows[1].price > lows[0].price;
  const lowerHighs = highs.length === 2 && highs[1].price < highs[0].price;
  const lowerLows = lows.length === 2 && lows[1].price < lows[0].price;

  let bias: TrendResult["bias"] = "sideways";
  let confidence = 35;
  const reasons: string[] = [];

  if (latest.close > latestEma && higherHighs && higherLows) {
    bias = "bullish";
    confidence = 90;
    reasons.push("Bullish trend: close above EMA200 with HH/HL");
  } else if (latest.close < latestEma && lowerHighs && lowerLows) {
    bias = "bearish";
    confidence = 90;
    reasons.push("Bearish trend: close below EMA200 with LH/LL");
  } else if (latest.close > latestEma) {
    bias = "bullish";
    confidence = 60;
    reasons.push("Bullish lean: close above EMA200");
  } else if (latest.close < latestEma) {
    bias = "bearish";
    confidence = 60;
    reasons.push("Bearish lean: close below EMA200");
  } else {
    reasons.push("Sideways trend: price near EMA200");
  }

  return {
    bias,
    confidence,
    score: Math.round((confidence / 100) * config.scoring.trendAlignment),
    reasons
  };
}
