import { AppConfig, TradingMode } from "../types";

export interface ModeOverride {
  strategy: Partial<AppConfig["strategy"]>;
  tradeGuards?: Partial<AppConfig["tradeGuards"]>;
  description: string;
}

export const TRADING_MODES: Record<TradingMode, ModeOverride> = {
  /**
   * Dumb: pure price action — S/R + BOS + FVG only.
   * Ignores trend, volume, session, and all other filters.
   * Enters after BOS forms near S/R, fills into the nearest FVG.
   * Good for learning the core ICT entry model without noise.
   */
  dumb: {
    description: "S/R only — simple price action entries. No trade limit.",
    strategy: {
      dumbMode: true,
      minScore: 1,
      watchlistScore: 1,
      allowCounterTrendTrades: false,
      counterTrendMinScore: 99,
      requireConfirmationCandle: false,
      rangeLongThreshold: 30,
      rangeShortThreshold: 70,
      rangeMinAlternations: 1,
    },
    tradeGuards: {
      maxTradesPerSession: 0  // 0 = unlimited
    }
  },

  /**
   * Beginner: loose rules, easier to get signals.
   * Good for learning how the bot works and seeing more activity.
   * Counter-trend trades allowed, lower score thresholds, wider range zones.
   */
  beginner: {
    description: "Loose rules — more signals, higher risk. Good for learning.",
    strategy: {
      minScore: 45,
      watchlistScore: 30,
      allowCounterTrendTrades: true,
      counterTrendMinScore: 50,
      requireConfirmationCandle: false,
      rangeLongThreshold: 25,
      rangeShortThreshold: 75,
      rangeMinAlternations: 1,
    }
  },

  /**
   * Advanced: balanced rules.
   * Confirmation candle required, counter-trend trades off, tighter range zones.
   */
  advanced: {
    description: "Balanced rules — moderate signals, moderate risk.",
    strategy: {
      minScore: 60,
      watchlistScore: 45,
      allowCounterTrendTrades: false,
      counterTrendMinScore: 70,
      requireConfirmationCandle: true,
      rangeLongThreshold: 20,
      rangeShortThreshold: 80,
      rangeMinAlternations: 2,
    }
  },

  /**
   * Expert: strict rules — current default behavior.
   * All confirmations required, tight range zones, high score threshold.
   */
  expert: {
    description: "Strict rules — fewer but higher quality signals.",
    strategy: {
      minScore: 75,
      watchlistScore: 60,
      allowCounterTrendTrades: false,
      counterTrendMinScore: 85,
      requireConfirmationCandle: true,
      rangeLongThreshold: 15,
      rangeShortThreshold: 85,
      rangeMinAlternations: 2,
    }
  },
  /**
   * AI: neural network predictions replace the rule-based scoring engine.
   * Requires a trained model at models/ai_model.npz.
   * Train with: python scripts/ai_collect.py && python scripts/ai_train.py data/gold_5m.csv
   */
  ai: {
    description: "Neural network predictions — train first with ai_train.py.",
    strategy: {
      aiMode: true,
      aiConfidenceThreshold: 0.55,
      aiModelPath: "models/ai_model.npz",
      dumbMode: false,
      minScore: 1,
      watchlistScore: 1,
      allowCounterTrendTrades: true,
      counterTrendMinScore: 1,
      requireConfirmationCandle: false,
      rangeLongThreshold: 15,
      rangeShortThreshold: 85,
      rangeMinAlternations: 2,
    }
  }

};

/**
 * Applies mode overrides on top of the loaded config.
 * Returns a new config object — does not mutate the original.
 */
export function applyTradingMode(config: AppConfig, mode: TradingMode): AppConfig {
  const override = TRADING_MODES[mode];
  return {
    ...config,
    strategy: {
      ...config.strategy,
      ...override.strategy
    },
    ...(override.tradeGuards ? {
      tradeGuards: {
        ...config.tradeGuards,
        ...override.tradeGuards
      }
    } : {})
  };
}
