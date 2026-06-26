import { AppConfig, Candle, OrderBlockResult, Zone } from "../types";

function isMitigated(candles: Candle[], zone: Zone, direction: "bullish" | "bearish"): boolean {
  return candles.some((candle) => {
    if (direction === "bullish") return candle.low <= zone.high && candle.high >= zone.low;
    return candle.high >= zone.low && candle.low <= zone.high;
  });
}

export function detectOrderBlock(candles: Candle[], config: AppConfig): OrderBlockResult {
  const lookback = Math.max(config.strategy.swingLookback * 4, 12);
  const recent = candles.slice(-lookback);
  const latest = candles.at(-1);
  if (!latest || recent.length < 4) return { score: 0, reasons: ["Not enough candles for order block"] };

  const bodies = recent.map((candle) => Math.abs(candle.close - candle.open));
  const averageBody = bodies.reduce((sum, body) => sum + body, 0) / bodies.length;
  const impulsiveMove = Math.abs(latest.close - latest.open) > averageBody * 1.5;
  const bullishImpulse = impulsiveMove && latest.close > latest.open;
  const bearishImpulse = impulsiveMove && latest.close < latest.open;
  let bullish: Zone | undefined;
  let bearish: Zone | undefined;

  if (bullishImpulse) {
    const candle = [...recent].reverse().find((item) => item.close < item.open);
    if (candle) {
      const afterOrderBlock = candles.slice(candles.indexOf(candle) + 1, -1);
      bullish = {
        low: candle.low,
        high: candle.high,
        strength: 70,
        time: candle.time,
        mitigated: isMitigated(afterOrderBlock, { low: candle.low, high: candle.high, strength: 70 }, "bullish")
      };
    }
  }

  if (bearishImpulse) {
    const candle = [...recent].reverse().find((item) => item.close > item.open);
    if (candle) {
      const afterOrderBlock = candles.slice(candles.indexOf(candle) + 1, -1);
      bearish = {
        low: candle.low,
        high: candle.high,
        strength: 70,
        time: candle.time,
        mitigated: isMitigated(afterOrderBlock, { low: candle.low, high: candle.high, strength: 70 }, "bearish")
      };
    }
  }

  const found = Boolean(bullish || bearish);
  const hasTradableBlock = Boolean((bullish && !bullish.mitigated) || (bearish && !bearish.mitigated));
  return {
    bullish,
    bearish,
    score: hasTradableBlock ? config.scoring.orderBlock : found ? Math.round(config.scoring.orderBlock * 0.35) : 0,
    reasons: [
      bullish ? `Bullish OB ${bullish.mitigated ? "mitigated" : "unmitigated"}` : "",
      bearish ? `Bearish OB ${bearish.mitigated ? "mitigated" : "unmitigated"}` : ""
    ].filter(Boolean)
  };
}
