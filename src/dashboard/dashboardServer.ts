import http from "node:http";
import { Broker } from "../broker/Broker";
import { logger } from "../logger/logger";
import { AppConfig, TradeSignal } from "../types";
import { SignalDecision } from "../strategy/signalEngine";

export interface DashboardState {
  lastDecision?: SignalDecision;
  lastSignal?: TradeSignal;
  dailyPnl: number;
}

export class DashboardServer {
  private server?: http.Server;

  constructor(
    private readonly config: AppConfig,
    private readonly broker: Broker,
    private readonly state: DashboardState
  ) {}

  start(): void {
    if (!this.config.dashboard.enabled || this.server) return;

    this.server = http.createServer(async (_req, res) => {
      try {
        const openTrades = await this.broker.getOpenPositions();
        const payload = {
          symbol: this.config.symbol,
          currentSignalScore: this.state.lastDecision?.score ?? null,
          currentSignalStatus: this.state.lastDecision?.status ?? null,
          lastSignal: this.state.lastSignal ?? null,
          openTrades,
          dailyPnl: this.state.dailyPnl,
          timestamp: Date.now()
        };
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify(payload, null, 2));
      } catch (error) {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: error instanceof Error ? error.message : String(error) }));
      }
    });

    this.server.listen(this.config.dashboard.port, this.config.dashboard.host, () => {
      logger.info("Dashboard started", {
        url: `http://${this.config.dashboard.host}:${this.config.dashboard.port}`
      });
    });
  }

  stop(): void {
    this.server?.close();
    this.server = undefined;
  }
}
