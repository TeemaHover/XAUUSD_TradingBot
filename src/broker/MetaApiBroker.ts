import { Broker } from "./Broker";
import { AppConfig, Candle, OrderRequest, Position, SymbolSpec, Timeframe, TradeHistoryItem } from "../types";
import { logger } from "../logger/logger";

/**
 * Broker adapter for MetaApi (metaapi.cloud) — connects to any MT4/MT5
 * account (demo or live) from macOS/Linux with no Windows machine.
 *
 * Setup:
 *   1. Sign up at https://metaapi.cloud and add your MT5 demo account
 *      (broker server name + login + password) in their dashboard.
 *   2. Copy the API token and the account id into .env:
 *        METAAPI_TOKEN=...
 *        METAAPI_ACCOUNT_ID=...
 *   3. Set broker.mode to "metaapi" in config/default.json (or BROKER_MODE=metaapi).
 *   4. npm install metaapi.cloud-sdk
 *
 * The SDK is loaded lazily via require() so the project still compiles and
 * runs in mock/mt5 mode without the dependency installed.
 */

// Minimal structural types for the parts of the SDK we use (avoids a hard
// compile-time dependency on the SDK's own type definitions).
interface MetaApiRpcConnection {
  connect(): Promise<void>;
  close(): Promise<void>;
  waitSynchronized(opts?: unknown): Promise<void>;
  getAccountInformation(): Promise<{ balance: number; currency?: string }>;
  getSymbolSpecification(symbol: string): Promise<Record<string, unknown>>;
  getSymbolPrice(symbol: string): Promise<{ bid: number; ask: number }>;
  getPositions(): Promise<Array<Record<string, unknown>>>;
  getDealsByTimeRange(start: Date, end: Date): Promise<{ deals?: Array<Record<string, unknown>> } | Array<Record<string, unknown>>>;
  createMarketBuyOrder(symbol: string, volume: number, stopLoss?: number, takeProfit?: number, options?: unknown): Promise<{ orderId?: string; positionId?: string; stringCode?: string }>;
  createMarketSellOrder(symbol: string, volume: number, stopLoss?: number, takeProfit?: number, options?: unknown): Promise<{ orderId?: string; positionId?: string; stringCode?: string }>;
  createLimitBuyOrder(symbol: string, volume: number, openPrice: number, stopLoss?: number, takeProfit?: number, options?: unknown): Promise<{ orderId?: string; positionId?: string; stringCode?: string }>;
  createLimitSellOrder(symbol: string, volume: number, openPrice: number, stopLoss?: number, takeProfit?: number, options?: unknown): Promise<{ orderId?: string; positionId?: string; stringCode?: string }>;
  modifyPosition(positionId: string, stopLoss?: number, takeProfit?: number): Promise<unknown>;
  closePosition(positionId: string, options?: unknown): Promise<unknown>;
}

interface MetaApiAccount {
  state: string;
  deploy(): Promise<void>;
  waitConnected(): Promise<void>;
  getRPCConnection(): MetaApiRpcConnection;
  getHistoricalCandles(symbol: string, timeframe: string, startTime?: Date, limit?: number): Promise<Array<Record<string, unknown>>>;
}

function num(value: unknown, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed !== 0 ? parsed : fallback;
}

export class MetaApiBroker implements Broker {
  private connection?: MetaApiRpcConnection;
  private account?: MetaApiAccount;

  constructor(private readonly config: AppConfig) {}

  async connect(): Promise<void> {
    const token = process.env.METAAPI_TOKEN;
    const accountId = process.env.METAAPI_ACCOUNT_ID;
    if (!token || !accountId) {
      throw new Error("MetaApi broker requires METAAPI_TOKEN and METAAPI_ACCOUNT_ID in .env");
    }

    // Lazy require so mock/mt5 modes work without the package installed.
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    let sdk: any;
    try {
      sdk = require("metaapi.cloud-sdk");
    } catch {
      throw new Error("metaapi.cloud-sdk is not installed. Run: npm install metaapi.cloud-sdk");
    }
    const MetaApi = sdk.default ?? sdk;
    const api = new MetaApi(token);

    logger.info("MetaApi: fetching account...", { accountId });
    const account: MetaApiAccount = await api.metatraderAccountApi.getAccount(accountId);
    if (account.state !== "DEPLOYED") {
      logger.info("MetaApi: deploying account (first time can take a minute)...");
      await account.deploy();
    }
    await account.waitConnected();

    const connection = account.getRPCConnection();
    await connection.connect();
    await connection.waitSynchronized();

    this.account = account;
    this.connection = connection;

    const info = await connection.getAccountInformation();
    logger.info("MetaApi broker connected", {
      symbol: this.config.symbol,
      balance: info.balance,
      currency: info.currency
    });
  }

  async disconnect(): Promise<void> {
    try {
      await this.connection?.close();
    } catch {
      /* ignore */
    }
    logger.info("MetaApi broker disconnected");
  }

  async getBalance(): Promise<number> {
    const info = await this.conn().getAccountInformation();
    return info.balance;
  }

  async getSymbolSpec(symbol: string): Promise<SymbolSpec> {
    const spec = await this.conn().getSymbolSpecification(symbol);
    const risk = this.config.risk;
    const digits = num(spec.digits, 2);
    const point = Math.pow(10, -digits);
    return {
      symbol,
      point,
      digits,
      tickSize: num(spec.tickSize, risk.tickSize),
      tickValue: num(spec.tickValue, risk.tickValue),
      contractSize: num(spec.contractSize ?? spec.lotSize, 100),
      volumeStep: num(spec.volumeStep ?? spec.lotStep, risk.volumeStep),
      minVolume: num(spec.minVolume ?? spec.volumeMin ?? spec.lotMin, risk.minVolume),
      maxVolume: num(spec.maxVolume ?? spec.volumeMax ?? spec.lotMax, risk.maxVolume),
      minStopDistance: num(spec.stopsLevel, 0) * point || risk.minStopDistance
    };
  }

  async getSpread(symbol: string): Promise<number> {
    const price = await this.conn().getSymbolPrice(symbol);
    return Math.max(0, price.ask - price.bid);
  }

  async getCandles(symbol: string, timeframe: Timeframe, limit: number): Promise<Candle[]> {
    if (!this.account) throw new Error("MetaApi broker is not connected");
    const raw = await this.account.getHistoricalCandles(symbol, timeframe, undefined, limit);
    return raw
      .map((c): Candle => ({
        time: new Date(c.time as string | number | Date).getTime(),
        open: Number(c.open),
        high: Number(c.high),
        low: Number(c.low),
        close: Number(c.close),
        volume: num(c.tickVolume ?? c.volume, 1)
      }))
      .filter((c) => Number.isFinite(c.time) && Number.isFinite(c.close))
      .sort((a, b) => a.time - b.time)
      .slice(-limit);
  }

  async placeOrder(order: OrderRequest): Promise<Position> {
    const conn = this.conn();
    const takeProfit = order.takeProfits[0];
    const options = { comment: (order.comment ?? "xauusd-bot").slice(0, 25) };
    const isLimit = order.entryType === "limit";

    let result: { orderId?: string; positionId?: string; stringCode?: string };
    if (order.direction === "long") {
      result = isLimit
        ? await conn.createLimitBuyOrder(order.symbol, order.volume, order.entry, order.stopLoss, takeProfit, options)
        : await conn.createMarketBuyOrder(order.symbol, order.volume, order.stopLoss, takeProfit, options);
    } else {
      result = isLimit
        ? await conn.createLimitSellOrder(order.symbol, order.volume, order.entry, order.stopLoss, takeProfit, options)
        : await conn.createMarketSellOrder(order.symbol, order.volume, order.stopLoss, takeProfit, options);
    }

    const position: Position = {
      ...order,
      id: result.positionId ?? result.orderId ?? `METAAPI-${Date.now()}`,
      openedAt: Date.now(),
      remainingVolume: order.volume
    };
    logger.info("MetaApi order sent", { id: position.id, code: result.stringCode, direction: order.direction, volume: order.volume });
    return position;
  }

  async modifyOrder(orderId: string, updates: Partial<OrderRequest>): Promise<Position | undefined> {
    await this.conn().modifyPosition(orderId, updates.stopLoss, updates.takeProfits?.[0]);
    const positions = await this.getOpenPositions();
    return positions.find((p) => p.id === orderId);
  }

  async closeOrder(orderId: string): Promise<Position | undefined> {
    const positions = await this.getOpenPositions();
    const position = positions.find((p) => p.id === orderId);
    await this.conn().closePosition(orderId, {});
    if (position) {
      return { ...position, closedAt: Date.now(), remainingVolume: 0 };
    }
    return undefined;
  }

  async getOpenPositions(): Promise<Position[]> {
    const raw = await this.conn().getPositions();
    return raw
      .filter((p) => p.symbol === this.config.symbol)
      .map((p): Position => {
        const direction = String(p.type).includes("SELL") ? "short" : "long";
        return {
          id: String(p.id),
          symbol: String(p.symbol),
          direction,
          volume: num(p.volume, 0),
          remainingVolume: num(p.currentVolume ?? p.volume, 0),
          entry: num(p.openPrice, 0),
          stopLoss: num(p.stopLoss, 0),
          takeProfits: p.takeProfit ? [Number(p.takeProfit)] : [],
          openedAt: new Date((p.time as string | number | Date) ?? Date.now()).getTime(),
          comment: p.comment ? String(p.comment) : undefined
        };
      });
  }

  /** Today's closed deals, mapped so RiskGuard can hydrate daily PnL. */
  async getTradeHistory(): Promise<TradeHistoryItem[]> {
    const start = new Date();
    start.setHours(0, 0, 0, 0);
    const response = await this.conn().getDealsByTimeRange(start, new Date());
    const deals = Array.isArray(response) ? response : response.deals ?? [];

    return deals
      .filter((d) => d.symbol === this.config.symbol && String(d.entryType ?? "").includes("OUT"))
      .map((d): TradeHistoryItem => {
        const closedAt = new Date((d.time as string | number | Date) ?? Date.now()).getTime();
        return {
          id: String(d.positionId ?? d.id),
          symbol: String(d.symbol),
          direction: String(d.type).includes("SELL") ? "long" : "short", // closing deal is opposite side
          volume: num(d.volume, 0),
          remainingVolume: 0,
          entry: num(d.price, 0),
          stopLoss: 0,
          takeProfits: [],
          openedAt: closedAt,
          closedAt,
          realizedR: 0,
          profit: num(d.profit, 0)
        };
      });
  }

  private conn(): MetaApiRpcConnection {
    if (!this.connection) throw new Error("MetaApi broker is not connected");
    return this.connection;
  }
}
