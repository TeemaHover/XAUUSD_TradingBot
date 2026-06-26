import { lastSwing } from "../market/swings";
import { AppConfig, Candle, LiquidityResult, MarketStructureResult } from "../types";

export function detectMarketStructure(
  candles: Candle[],
  liquidity: LiquidityResult,
  config: AppConfig
): MarketStructureResult {
  const latest = candles.at(-1);
  const lastHigh = lastSwing(liquidity.swings, "high");
  const lastLow = lastSwing(liquidity.swings, "low");
  if (!latest || !lastHigh || !lastLow) {
    return { bos: "none", mss: "none", choch: "none", score: 0, reasons: ["Not enough swings for market structure"] };
  }

  const bullishBreak = latest.close > lastHigh.price;
  const bearishBreak = latest.close < lastLow.price;
  const bullishMss = liquidity.bullishSweep && bullishBreak;
  const bearishMss = liquidity.bearishSweep && bearishBreak;
  const bos = bullishBreak ? "bullish" : bearishBreak ? "bearish" : "none";
  const mss = bullishMss ? "bullish" : bearishMss ? "bearish" : "none";
  const choch = mss;
  const valid = mss !== "none" || bos !== "none";

  return {
    bos,
    mss,
    choch,
    score: valid ? config.scoring.marketStructure : 0,
    reasons: [
      bos !== "none" ? `BOS detected: ${bos}` : "",
      mss !== "none" ? `MSS/CHOCH detected: ${mss}` : ""
    ].filter(Boolean)
  };
}
