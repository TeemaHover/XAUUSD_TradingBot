import { Candle } from "../types";
import { trueRanges } from "./atr";

function smooth(values: number[], length: number): number[] {
  return values.map((_, index) => {
    const start = Math.max(0, index - length + 1);
    const slice = values.slice(start, index + 1);
    return slice.reduce((sum, value) => sum + value, 0) / slice.length;
  });
}

export function adx(candles: Candle[], length: number): number[] {
  if (candles.length < 2) return [];

  const plusDm: number[] = [0];
  const minusDm: number[] = [0];
  for (let i = 1; i < candles.length; i += 1) {
    const upMove = candles[i].high - candles[i - 1].high;
    const downMove = candles[i - 1].low - candles[i].low;
    plusDm.push(upMove > downMove && upMove > 0 ? upMove : 0);
    minusDm.push(downMove > upMove && downMove > 0 ? downMove : 0);
  }

  const tr = trueRanges(candles);
  const smoothedTr = smooth(tr, length);
  const smoothedPlus = smooth(plusDm, length);
  const smoothedMinus = smooth(minusDm, length);
  const dx = candles.map((_, index) => {
    const trValue = smoothedTr[index];
    if (trValue <= 0) return 0;
    const plusDi = (smoothedPlus[index] / trValue) * 100;
    const minusDi = (smoothedMinus[index] / trValue) * 100;
    const total = plusDi + minusDi;
    return total <= 0 ? 0 : (Math.abs(plusDi - minusDi) / total) * 100;
  });

  return smooth(dx, length);
}
