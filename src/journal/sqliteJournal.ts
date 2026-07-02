import fs from "node:fs";
import path from "node:path";
import { logger } from "../logger/logger";
import { AppConfig, Position, TradingMode, TradeSignal } from "../types";
import { SignalDecision } from "../strategy/signalEngine";

type DatabaseSyncConstructor = new (filename: string) => {
  exec(sql: string): void;
  prepare(sql: string): { run(...params: unknown[]): unknown; get(...params: unknown[]): unknown };
  close(): void;
};

interface SqliteModule {
  DatabaseSync: DatabaseSyncConstructor;
}

const sqlite = require("node:sqlite") as SqliteModule;

export class SqliteJournal {
  private db?: InstanceType<DatabaseSyncConstructor>;

  constructor(private readonly config: AppConfig) {}

  init(): void {
    if (!this.config.journal.enabled || this.db) return;
    const resolved = path.resolve(this.config.journal.path);
    fs.mkdirSync(path.dirname(resolved), { recursive: true });
    this.db = new sqlite.DatabaseSync(resolved);
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS bot_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        time INTEGER NOT NULL,
        event_type TEXT NOT NULL,
        symbol TEXT NOT NULL,
        details_json TEXT NOT NULL
      );

      CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        time INTEGER NOT NULL,
        symbol TEXT NOT NULL,
        status TEXT NOT NULL,
        action TEXT NOT NULL,
        direction TEXT NOT NULL,
        trend_direction TEXT NOT NULL,
        setup_type TEXT NOT NULL,
        score REAL NOT NULL,
        required_score REAL NOT NULL,
        allowed INTEGER NOT NULL,
        blocked_by_json TEXT NOT NULL,
        reasons_json TEXT NOT NULL,
        final_decision_json TEXT NOT NULL
      );

      CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        open_time INTEGER NOT NULL,
        position_id TEXT NOT NULL,
        symbol TEXT NOT NULL,
        direction TEXT NOT NULL,
        volume REAL NOT NULL,
        entry REAL NOT NULL,
        stop_loss REAL NOT NULL,
        take_profits_json TEXT NOT NULL,
        score REAL NOT NULL,
        setup_type TEXT NOT NULL,
        regime TEXT,
        signal_json TEXT NOT NULL,
        position_json TEXT NOT NULL
      );

      CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at INTEGER NOT NULL
      );
    `);
    logger.info("SQLite journal started", { path: resolved });
  }

  // ── Mode persistence ────────────────────────────────────────────────────────

  saveMode(mode: TradingMode): void {
    this.safeRun(() => {
      this.db?.prepare(`
        INSERT INTO settings (key, value, updated_at)
        VALUES ('trading_mode', ?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
      `).run(mode, Date.now());
      logger.info("Trading mode saved", { mode });
    });
  }

  loadMode(): TradingMode | undefined {
    if (!this.config.journal.enabled || !this.db) return undefined;
    try {
      const row = this.db.prepare(
        `SELECT value FROM settings WHERE key = 'trading_mode'`
      ).get() as { value: string } | undefined;
      if (!row) return undefined;
      const valid: TradingMode[] = ["beginner", "advanced", "expert"];
      return valid.includes(row.value as TradingMode) ? (row.value as TradingMode) : undefined;
    } catch {
      return undefined;
    }
  }

  // ── Event / signal / trade recording ────────────────────────────────────────

  recordBotEvent(eventType: "started" | "stopped" | "error", details: Record<string, unknown> = {}): void {
    this.safeRun(() => {
      this.db?.prepare(`
        INSERT INTO bot_events (time, event_type, symbol, details_json)
        VALUES (?, ?, ?, ?)
      `).run(Date.now(), eventType, this.config.symbol, JSON.stringify(details));
    });
  }

  recordSignal(decision: SignalDecision): void {
    this.safeRun(() => {
      const finalDecision = decision.finalDecision;
      this.db?.prepare(`
        INSERT INTO signals (
          time, symbol, status, action, direction, trend_direction, setup_type,
          score, required_score, allowed, blocked_by_json, reasons_json, final_decision_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `).run(
        Date.now(),
        this.config.symbol,
        decision.status,
        finalDecision.action,
        finalDecision.direction,
        finalDecision.trendDirection,
        finalDecision.setupType,
        decision.score,
        finalDecision.requiredScore,
        finalDecision.allowed ? 1 : 0,
        JSON.stringify(finalDecision.blockedBy),
        JSON.stringify(decision.reasons),
        JSON.stringify(finalDecision)
      );
    });
  }

  recordTradeOpened(position: Position, signal: TradeSignal, setupType: string): void {
    this.safeRun(() => {
      this.db?.prepare(`
        INSERT INTO trades (
          open_time, position_id, symbol, direction, volume, entry, stop_loss,
          take_profits_json, score, setup_type, regime, signal_json, position_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `).run(
        position.openedAt,
        position.id,
        position.symbol,
        position.direction,
        position.volume,
        position.entry,
        position.stopLoss,
        JSON.stringify(position.takeProfits),
        signal.score,
        setupType,
        signal.regime ?? null,
        JSON.stringify(signal),
        JSON.stringify(position)
      );
    });
  }

  close(): void {
    if (!this.db) return;
    this.db.close();
    this.db = undefined;
  }

  private safeRun(action: () => void): void {
    if (!this.config.journal.enabled || !this.db) return;
    try {
      action();
    } catch (error) {
      logger.warn("SQLite journal write failed", { message: error instanceof Error ? error.message : String(error) });
    }
  }
}
