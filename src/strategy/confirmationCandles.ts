import { Candle, Direction } from "../types";

export function hasConfirmationCandle(candles: Candle[], direction: Direction): boolean {
  if (candles.length < 2) return false;
  const current = candles.at(-1)!;
  const previous = candles.at(-2)!;
  const body = Math.abs(current.close - current.open);
  const range = current.high - current.low;
  const upperWick = current.high - Math.max(current.open, current.close);
  const lowerWick = Math.min(current.open, current.close) - current.low;
  const strongBody = range > 0 && body / range >= 0.55;

  const bullishEngulfing = current.close > current.open && previous.close < previous.open && current.close > previous.open && current.open < previous.close;
  const bearishEngulfing = current.close < current.open && previous.close > previous.open && current.close < previous.open && current.open > previous.close;
  const bullishPin = current.close > current.open && lowerWick > body * 1.5 && upperWick < body;
  const bearishPin = current.close < current.open && upperWick > body * 1.5 && lowerWick < body;

  if (direction === "long") return bullishEngulfing || bullishPin || (current.close > current.open && strongBody);
  return bearishEngulfing || bearishPin || (current.close < current.open && strongBody);
}
