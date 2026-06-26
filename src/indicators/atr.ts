import { Candle } from "../types";

export function trueRanges(candles: Candle[]): number[] {
  return candles.map((candle, index) => {
    if (index === 0) return candle.high - candle.low;
    const previousClose = candles[index - 1].close;
    return Math.max(
      candle.high - candle.low,
      Math.abs(candle.high - previousClose),
      Math.abs(candle.low - previousClose)
    );
  });
}

export function atr(candles: Candle[], length: number): number[] {
  const ranges = trueRanges(candles);
  if (ranges.length === 0) return [];

  const output: number[] = [];
  for (let i = 0; i < ranges.length; i += 1) {
    const start = Math.max(0, i - length + 1);
    const slice = ranges.slice(start, i + 1);
    output.push(slice.reduce((sum, value) => sum + value, 0) / slice.length);
  }

  return output;
}
