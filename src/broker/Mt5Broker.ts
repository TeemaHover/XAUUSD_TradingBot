import { spawn } from "node:child_process";
import path from "node:path";
import { Broker } from "./Broker";
import { AppConfig, Candle, OrderRequest, Position, SymbolSpec, Timeframe, TradeHistoryItem } from "../types";
import { logger } from "../logger/logger";

interface BridgeResponse<T> {
  ok: boolean;
  result?: T;
  error?: string;
}

export class Mt5Broker implements Broker {
  private readonly maxRetries = 2;

  constructor(private readonly config: AppConfig) {}

  async connect(): Promise<void> {
    await this.call<{ connected: boolean }>("ping", {});
    logger.info("MT5 broker connected through Python bridge", {
      symbol: this.config.symbol,
      dryRun: this.config.mt5.dryRun
    });
  }

  async disconnect(): Promise<void> {
    logger.info("MT5 broker disconnected");
  }

  async getBalance(): Promise<number> {
    const result = await this.call<{ balance: number }>("balance", {});
    return result.balance;
  }

  async getSymbolSpec(symbol: string): Promise<SymbolSpec> {
    const result = await this.call<{ spec: SymbolSpec }>("symbol_info", { symbol });
    return result.spec;
  }

  async getSpread(symbol: string): Promise<number> {
    const result = await this.call<{ spread: number }>("spread", { symbol });
    return result.spread;
  }

  async getCandles(symbol: string, timeframe: Timeframe, limit: number): Promise<Candle[]> {
    const result = await this.call<{ candles: Candle[] }>("candles", { symbol, timeframe, limit });
    return result.candles;
  }

  async placeOrder(order: OrderRequest): Promise<Position> {
    const result = await this.call<{ dryRun?: boolean; ticket?: string; price?: number; volume?: number }>("order", {
      ...order,
      dryRun: this.config.mt5.dryRun,
      deviation: this.config.mt5.deviation,
      magic: this.config.mt5.magic
    });

    const position: Position = {
      ...order,
      id: result.ticket ?? `MT5-DRY-RUN-${Date.now()}`,
      openedAt: Date.now(),
      remainingVolume: result.volume ?? order.volume,
      entry: result.price ?? order.entry
    };

    logger.info(result.dryRun ? "MT5 dry-run order checked" : "MT5 order sent", position);
    return position;
  }

  async modifyOrder(orderId: string, _updates: Partial<OrderRequest>): Promise<Position | undefined> {
    logger.warn("MT5 modifyOrder is not implemented in V2 adapter", { orderId });
    return undefined;
  }

  async closeOrder(orderId: string): Promise<Position | undefined> {
    logger.warn("MT5 closeOrder is not implemented in V2 adapter", { orderId });
    return undefined;
  }

  async getOpenPositions(): Promise<Position[]> {
    const result = await this.call<{ positions: Position[] }>("positions", { symbol: this.config.symbol });
    return result.positions;
  }

  async getTradeHistory(): Promise<TradeHistoryItem[]> {
    const result = await this.call<{ history: TradeHistoryItem[] }>("history", {
      symbol: this.config.symbol,
      magic: this.config.mt5.magic
    });
    return result.history;
  }

  private async call<T>(command: string, args: Record<string, unknown>): Promise<T> {
    let lastError: unknown;
    for (let attempt = 0; attempt <= this.maxRetries; attempt += 1) {
      try {
        return await this.callOnce<T>(command, args);
      } catch (error) {
        lastError = error;
        logger.warn("MT5 bridge command failed", {
          command,
          attempt,
          message: error instanceof Error ? error.message : String(error)
        });
      }
    }
    throw lastError instanceof Error ? lastError : new Error(String(lastError));
  }

  private async callOnce<T>(command: string, args: Record<string, unknown>): Promise<T> {
    const bridgePath = path.resolve(this.config.mt5.bridgePath);
    const input = JSON.stringify({ command, args });
    const { stdout, stderr } = await new Promise<{ stdout: string; stderr: string }>((resolve, reject) => {
      const child = spawn(this.config.mt5.pythonPath, [bridgePath], {
        stdio: ["pipe", "pipe", "pipe"]
      });
      let stdout = "";
      let stderr = "";
      child.stdout.setEncoding("utf8");
      child.stderr.setEncoding("utf8");
      child.stdout.on("data", (chunk) => {
        stdout += chunk;
      });
      child.stderr.on("data", (chunk) => {
        stderr += chunk;
      });
      child.on("error", reject);
      child.on("close", () => resolve({ stdout, stderr }));
      child.stdin.write(input);
      child.stdin.end();
    });

    if (stderr.trim()) {
      logger.warn("MT5 bridge stderr", { stderr: stderr.trim() });
    }

    const output = stdout.trim();
    if (!output) {
      throw new Error(`MT5 bridge returned empty output for command ${command}`);
    }

    let parsed: BridgeResponse<T>;
    try {
      parsed = JSON.parse(output) as BridgeResponse<T>;
    } catch {
      throw new Error(`MT5 bridge returned invalid JSON for command ${command}: ${output.slice(0, 500)}`);
    }
    if (!parsed.ok) {
      throw new Error(parsed.error ?? `MT5 bridge command failed: ${command}`);
    }
    return parsed.result as T;
  }
}
