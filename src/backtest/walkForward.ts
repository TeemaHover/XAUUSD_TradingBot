import fs from "node:fs";
import { AppConfig, Candle } from "../types";
import { runBacktest, BacktestResult } from "./backtestEngine";
import { logger } from "../logger/logger";

export interface WalkForwardSegment {
  index: number;
  trainStart: number;
  trainEnd: number;
  testStart: number;
  testEnd: number;
  result: BacktestResult;
}

export interface WalkForwardResult {
  segments: WalkForwardSegment[];
  summary: {
    segments: number;
    totalTrades: number;
    averageReturn: number;
    averageMaxDrawdown: number;
    averageExpectancy: number;
  };
}

export function runWalkForward(candles: Candle[], config: AppConfig, outputPath = config.walkForward.outputPath): WalkForwardResult {
  const segments: WalkForwardSegment[] = [];
  const { trainWindow, testWindow, stepSize } = config.walkForward;

  for (let start = 0, index = 0; start + trainWindow + testWindow <= candles.length; start += stepSize, index += 1) {
    const trainStart = start;
    const trainEnd = start + trainWindow;
    const testStart = trainEnd;
    const testEnd = testStart + testWindow;
    const contextStart = Math.max(0, testStart - trainWindow);
    const warmupAndTest = candles.slice(contextStart, testEnd);
    const segmentOutput = outputPath.replace(/\.json$/i, `-${index}.json`);
    const result = runBacktest(warmupAndTest, config, segmentOutput, {
      startIndex: testStart - contextStart,
      endIndex: testEnd - contextStart
    });

    segments.push({
      index,
      trainStart: candles[trainStart].time,
      trainEnd: candles[trainEnd - 1].time,
      testStart: candles[testStart].time,
      testEnd: candles[testEnd - 1].time,
      result
    });
  }

  const summary = {
    segments: segments.length,
    totalTrades: segments.reduce((sum, segment) => sum + segment.result.totalTrades, 0),
    averageReturn: average(segments.map((segment) => segment.result.totalReturn)),
    averageMaxDrawdown: average(segments.map((segment) => segment.result.maxDrawdown)),
    averageExpectancy: average(segments.map((segment) => segment.result.expectancy))
  };
  const output: WalkForwardResult = { segments, summary };
  fs.writeFileSync(outputPath, `${JSON.stringify(output, null, 2)}\n`);
  logger.info("Walk-forward results", summary);
  return output;
}

function average(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}
