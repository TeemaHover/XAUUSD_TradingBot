import { Broker } from "../broker/Broker";
import { newsFilter } from "../filters/newsFilter";
import { sessionFilter } from "../filters/sessionFilter";
import { logger } from "../logger/logger";
import { calculatePositionSize, PositionSizingSpec, RiskGuard } from "../risk/positionSizing";
import { AppConfig, Position, TradeSignal } from "../types";
import { TradeGuard } from "./tradeGuards";
import { TelegramAlerts } from "../alerts/telegram";
import { SqliteJournal } from "../journal/sqliteJournal";

export function formatOpenPositionAlert(position: Position, signal: TradeSignal, config: AppConfig): string {
  return [
    "Trade opened",
    `Symbol: ${position.symbol}`,
    `Direction: ${position.direction.toUpperCase()}`,
    `Volume: ${position.volume}`,
    `Entry: ${position.entry}`,
    `Stop loss: ${position.stopLoss}`,
    `Take profits: ${position.takeProfits.join(", ")}`,
    `Score: ${signal.score}`,
    `Position ID: ${position.id}`,
    `Mode: ${config.mt5.dryRun ? "DRY_RUN" : "DEMO/LIVE ORDERING ENABLED"}`
  ].join("\n");
}

export class ExecutionEngine {
  private executing = false;

  constructor(
    private readonly broker: Broker,
    private readonly config: AppConfig,
    private readonly riskGuard: RiskGuard,
    private readonly journal?: SqliteJournal,
    private readonly tradeGuard = new TradeGuard(config),
    private readonly alerts = new TelegramAlerts(config)
  ) {}

  async execute(signal: TradeSignal): Promise<void> {
    if (this.executing) {
      logger.warn("Trade rejected: execution already in progress", { symbol: signal.symbol });
      return;
    }
    this.executing = true;
    try {
      await this.executeLocked(signal);
    } finally {
      this.executing = false;
    }
  }

  private async executeLocked(signal: TradeSignal): Promise<void> {
    const history = await this.broker.getTradeHistory();
    this.riskGuard.hydrateFromClosedProfits(history.map((trade) => trade.profit ?? 0));

    if (!this.riskGuard.canTrade()) {
      logger.warn("Trade rejected by risk guard", { symbol: signal.symbol, score: signal.score, status: this.riskGuard.status() });
      return;
    }

    const session = sessionFilter(signal.timestamp, this.config);
    if (this.config.sessions.enabled && session.score <= 0) {
      logger.info("Trade rejected by session filter at execution", { symbol: signal.symbol, reasons: session.reasons });
      return;
    }

    const news = newsFilter(signal.timestamp, signal.symbol, this.config);
    if (news.blocked) {
      logger.info("Trade rejected by news blackout at execution", { symbol: signal.symbol, reasons: news.reasons });
      return;
    }

    const guard = this.tradeGuard.canOpen(signal.timestamp);
    if (!guard.allowed) {
      logger.info("Trade rejected by trade guard", { symbol: signal.symbol, reasons: guard.reasons });
      return;
    }

    const openPositions = await this.broker.getOpenPositions();
    if (openPositions.some((position) => position.symbol === signal.symbol)) {
      logger.info("Trade rejected: position already open", { symbol: signal.symbol });
      return;
    }

    const balance = await this.broker.getBalance();
    const symbolSpec = await this.broker.getSymbolSpec(signal.symbol);
    const sizingSpec: PositionSizingSpec = {
      tickSize: symbolSpec.tickSize || this.config.risk.tickSize,
      tickValue: symbolSpec.tickValue || this.config.risk.tickValue,
      volumeStep: symbolSpec.volumeStep || this.config.risk.volumeStep,
      minVolume: symbolSpec.minVolume || this.config.risk.minVolume,
      maxVolume: symbolSpec.maxVolume || this.config.risk.maxVolume
    };
    const stopDistance = Math.abs(signal.entry - signal.stopLoss);
    if (stopDistance < Math.max(this.config.risk.minStopDistance, symbolSpec.minStopDistance)) {
      logger.warn("Trade rejected: stop distance below configured/broker minimum", {
        symbol: signal.symbol,
        stopDistance,
        configMin: this.config.risk.minStopDistance,
        brokerMin: symbolSpec.minStopDistance
      });
      return;
    }
    const volume = calculatePositionSize(balance, this.config.risk.riskPerTrade, signal.entry, signal.stopLoss, sizingSpec);
    if (volume < sizingSpec.minVolume || volume > sizingSpec.maxVolume) {
      logger.warn("Trade rejected: calculated volume outside broker limits", { symbol: signal.symbol, volume, sizingSpec });
      return;
    }

    const syncedPositions = await this.broker.getOpenPositions();
    if (syncedPositions.some((position) => position.symbol === signal.symbol)) {
      logger.info("Trade rejected after position sync: position already open", { symbol: signal.symbol });
      return;
    }

    const timeframeSeconds: Record<string, number> = { "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400 };
    const expiryBars = this.config.tradeManagement.limitExpiryBars ?? 12;
    const barSeconds = timeframeSeconds[this.config.timeframes.entry] ?? 300;

    const position = await this.broker.placeOrder({
      symbol: signal.symbol,
      direction: signal.direction,
      volume,
      entry: signal.entry,
      stopLoss: signal.stopLoss,
      takeProfits: signal.takeProfits,
      comment: `score=${signal.score}`,
      entryType: signal.entryType ?? "market",
      expirySeconds: signal.entryType === "limit" ? expiryBars * barSeconds : undefined
    });

    logger.info(signal.entryType === "limit" ? "Placed pending limit order" : "Executed trade", position);
    this.tradeGuard.recordOpen(signal.timestamp);
    this.journal?.recordTradeOpened(position, signal, signal.reasons.find((reason) => reason.includes("context")) ?? "unknown");
    await this.alerts.send(formatOpenPositionAlert(position, signal, this.config));
  }
}
