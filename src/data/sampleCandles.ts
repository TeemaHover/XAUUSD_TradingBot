import { Candle } from "../types";

/**
 * Deterministic gold-like sample candle generator for tests and mock runs.
 * Uses a seeded PRNG so results are reproducible across runs.
 */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a += 0x6d2b79f5;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function sampleCandles(count: number, seed = 42, startPrice = 3300): Candle[] {
  const rand = mulberry32(seed);
  const candles: Candle[] = [];
  const startTime = 1_750_000_000; // fixed epoch seconds
  const stepSeconds = 300; // M5 bars

  let price = startPrice;
  let trend = 0.15; // gentle drift, flips periodically to create swings

  for (let i = 0; i < count; i++) {
    if (i % 40 === 0 && i > 0) {
      trend = -trend * (0.8 + rand() * 0.6); // flip/scale drift -> swing structure
    }
    const volatility = 1.5 + rand() * 2.5; // dollars per bar
    const open = price;
    const drift = trend + (rand() - 0.5) * volatility;
    const close = Math.max(1, open + drift);
    const high = Math.max(open, close) + rand() * volatility * 0.6;
    const low = Math.min(open, close) - rand() * volatility * 0.6;
    const volume = Math.round(500 + rand() * 1500);

    candles.push({
      time: startTime + i * stepSeconds,
      open: round2(open),
      high: round2(high),
      low: round2(low),
      close: round2(close),
      volume,
    });
    price = close;
  }
  return candles;
}

function round2(value: number): number {
  return Math.round(value * 100) / 100;
}
