import { AppConfig, DetectorResult, MarketRegime, ScoreComponent } from "../types";

export type ScoreBreakdown = Record<ScoreComponent, DetectorResult>;

export function adaptiveRawScore(
  breakdown: ScoreBreakdown,
  regime: MarketRegime,
  config: AppConfig
): { rawScore: number; maxScore: number; reasons: string[] } {
  const reasons: string[] = [];
  const multipliers = config.adaptiveScoring.regimeScoreMultipliers[regime];
  let rawScore = 0;
  let maxScore = 0;

  for (const [component, result] of Object.entries(breakdown) as [ScoreComponent, DetectorResult][]) {
    const multiplier = config.adaptiveScoring.enabled ? multipliers[component] : 1;
    rawScore += result.score * multiplier;
    maxScore += config.scoring[component] * multiplier;
    if (multiplier !== 1) reasons.push(`Adaptive ${component} x${multiplier.toFixed(2)} for ${regime}`);
  }

  return { rawScore, maxScore, reasons };
}
