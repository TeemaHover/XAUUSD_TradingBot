import fs from "node:fs";
import { Candle } from "../types";

export function loadCandlesFromCsv(path: string): Candle[] {
  const raw = fs.readFileSync(path, "utf8").trim();
  if (!raw) return [];

  const [headerLine, ...lines] = raw.split(/\r?\n/);
  const headers = headerLine.split(",").map((value) => value.trim());

  return lines.map((line) => {
    const values = line.split(",").map((value) => value.trim());
    const row = Object.fromEntries(headers.map((header, index) => [header, values[index]]));
    return {
      time: Number(row.time),
      open: Number(row.open),
      high: Number(row.high),
      low: Number(row.low),
      close: Number(row.close),
      volume: Number(row.volume)
    };
  }).filter((candle) => Number.isFinite(candle.time) && Number.isFinite(candle.close));
}
