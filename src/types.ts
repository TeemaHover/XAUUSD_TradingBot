export type Direction = "long" | "short";
export type Bias = "bullish" | "bearish" | "sideways";
export type Timeframe = "5m" | "15m" | "1h" | "4h" | "1d";
export type SessionName = "Asian" | "London" | "NewYork" | "OffSession";
export type MarketRegime = "trending" | "ranging" | "highVolatility" | "lowVolatility";
export type TradingMode = "beginner" | "advanced" | "expert" | "dumb" | "ai";
export type ScoreComponent =
  | "trendAlignment"
  | "liquiditySweep"
  | "marketStructure"
  | "orderBlock"
  | "fairValueGap"
  | "volumeConfirmation"
  | "sessionAllowed"
  | "volatilityValid"
  | "srConfirmation"
  | "fibConfirmation"
  | "doublePattern";

export interface Candle {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface TradeSignal {
  symbol: string;
  direction: Direction;
  entry: number;
  stopLoss: number;
  takeProfits: number[];
  score: number;
  reasons: string[];
  timestamp: number;
  regime?: MarketRegime;
}

export interface WatchlistSignal {
  symbol: string;
  direction?: Direction;
  score: number;
  reasons: string[];
  timestamp: number;
  regime?: MarketRegime;
}

export interface SwingPoint {
  index: number;
  time: number;
  price: number;
  type: "high" | "low";
}

export interface Zone {
  low: number;
  high: number;
  strength: number;
  mitigated?: boolean;
  time?: number;
}

export interface DetectorResult {
  score: number;
  reasons: string[];
}

export interface TrendResult extends DetectorResult {
  bias: Bias;
  confidence: number;
}

export interface LiquidityResult extends DetectorResult {
  bullishSweep: boolean;
  bearishSweep: boolean;
  equalHighs: Zone[];
  equalLows: Zone[];
  swings: SwingPoint[];
}

export interface MarketStructureResult extends DetectorResult {
  bos: Bias | "none";
  mss: Bias | "none";
  choch: Bias | "none";
}

export interface OrderBlockResult extends DetectorResult {
  bullish?: Zone;
  bearish?: Zone;
}

export interface FvgResult extends DetectorResult {
  bullish?: Zone & { filledPercent: number };
  bearish?: Zone & { filledPercent: number };
}

export interface VolumeResult extends DetectorResult {
  averageVolume: number;
  relativeVolume: number;
  spike: boolean;
}

export interface SRResult extends DetectorResult {
  supportZones: Zone[];
  resistanceZones: Zone[];
  nearSupport: boolean;
  nearResistance: boolean;
  nearRoundNumber: boolean;
}

export interface MarketRegimeResult extends DetectorResult {
  regime: MarketRegime;
  atr: number;
  averageAtr: number;
  adx: number;
}

export interface NewsEvent {
  time: number;
  title: string;
  impact: "low" | "medium" | "high";
  symbols?: string[];
}

export interface AppConfig {
  symbol: string;
  bot: {
    loopEnabled: boolean;
    intervalSeconds: number;
  };
  broker: {
    mode: "mock" | "mt5";
  };
  timeframes: {
    entry: Timeframe;
    trend: Timeframe;
    higherTrend: Timeframe;
  };
  risk: {
    riskPerTrade: number;
    maxDailyLoss: number;
    maxConsecutiveLosses: number;
    minStopDistance: number;
    stopBufferAtr: number;
    contractSize: number;
    tickSize: number;
    tickValue: number;
    volumeStep: number;
    minVolume: number;
    maxVolume: number;
  };
  strategy: {
    emaLength: number;
    atrLength: number;
    minScore: number;
    watchlistScore: number;
    swingLookback: number;
    equalHighLowToleranceAtr: number;
    volumeSpikeMultiplier: number;
    minAtr: number;
    maxSpread: number;
    allowCounterTrendTrades: boolean;
    counterTrendMinScore: number;
    /** require a confirmation candle before entry */
    requireConfirmationCandle: boolean;
    /** range trade: long when price position <= this % */
    rangeLongThreshold: number;
    /** range trade: short when price position >= this % */
    rangeShortThreshold: number;
    /** minimum alternations required to confirm a range */
    rangeMinAlternations: number;
    /** dumb mode: only trade S/R + BOS + FVG — ignores all other filters */
    dumbMode: boolean;
    /** ai mode: use neural network model for trading decisions */
    aiMode: boolean;
    /** minimum model confidence to enter a trade (0-1, default 0.55) */
    aiConfidenceThreshold: number;
    /** path to trained model weights (default models/ai_model.npz) */
    aiModelPath: string;
    srProximityAtr: number;
    srZoneToleranceAtr: number;
  };
  regime: {
    adxLength: number;
    trendAdxThreshold: number;
    rangeAdxThreshold: number;
    highVolatilityAtrMultiplier: number;
    lowVolatilityAtrMultiplier: number;
    lookback: number;
  };
  adaptiveScoring: {
    enabled: boolean;
    regimeScoreMultipliers: Record<MarketRegime, Record<ScoreComponent, number>>;
  };
  news: {
    enabled: boolean;
    blackoutMinutesBefore: number;
    blackoutMinutesAfter: number;
    events: NewsEvent[];
  };
  tradeGuards: {
    cooldownAfterLossMinutes: number;
    maxTradesPerSession: number;
  };
  alerts: {
    telegram: {
      enabled: boolean;
      botToken: string;
      chatId: string;
    };
  };
  dashboard: {
    enabled: boolean;
    host: string;
    port: number;
  };
  journal: {
    enabled: boolean;
    path: string;
  };
  walkForward: {
    enabled: boolean;
    trainWindow: number;
    testWindow: number;
    stepSize: number;
    outputPath: string;
  };
  scoring: {
    trendAlignment: number;
    liquiditySweep: number;
    marketStructure: number;
    orderBlock: number;
    fairValueGap: number;
    volumeConfirmation: number;
    sessionAllowed: number;
    volatilityValid: number;
    srConfirmation: number;
    fibConfirmation: number;
    doublePattern: number;
  };
  sessions: {
    enabled: boolean;
    allowed: SessionName[];
    utcOffsetMinutes: number;
  };
  tradeManagement: {
    tpRMultiples: number[];
    moveToBreakEvenAfterTp1: boolean;
    trailingStop: boolean;
  };
  mockBroker: {
    balance: number;
    spread: number;
    slippage: number;
    commissionPerLot: number;
  };
  mt5: {
    pythonPath: string;
    bridgePath: string;
    dryRun: boolean;
    deviation: number;
    magic: number;
    bars: {
      entry: number;
      trend: number;
      higherTrend: number;
    };
  };
}

export interface SymbolSpec {
  symbol: string;
  point: number;
  digits: number;
  tickSize: number;
  tickValue: number;
  contractSize: number;
  volumeStep: number;
  minVolume: number;
  maxVolume: number;
  minStopDistance: number;
}

export interface OrderRequest {
  symbol: string;
  direction: Direction;
  volume: number;
  entry: number;
  stopLoss: number;
  takeProfits: number[];
  comment?: string;
}

export interface Position extends OrderRequest {
  id: string;
  openedAt: number;
  remainingVolume: number;
  closedAt?: number;
  realizedR?: number;
}

export interface TradeHistoryItem extends Position {
  closedAt: number;
  realizedR: number;
  profit?: number;
}
