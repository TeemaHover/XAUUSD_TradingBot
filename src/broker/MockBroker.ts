import { Broker } from "./Broker";
import { Candle, OrderRequest, Position, SymbolSpec, Timeframe, TradeHistoryItem } from "../types";
import { logger } from "../logger/logger";

export class MockBroker implements Broker {
  private connected = false;
  private positions: Position[] = [];
  private history: TradeHistoryItem[] = [];

  constructor(
    private balance: number,
    private readonly spread: number,
    private readonly candlesByTimeframe: Partial<Record<Timeframe, Candle[]>> = {}
  ) {}

  async connect(): Promise<void> {
    this.connected = true;
    logger.info("MockBroker connected");
  }

  async disconnect(): Promise<void> {
    this.connected = false;
    logger.info("MockBroker disconnected");
  }

  async getBalance(): Promise<number> {
    this.ensureConnected();
    return this.balance;
  }

  async getSymbolSpec(symbol: string): Promise<SymbolSpec> {
    this.ensureConnected();
    return {
      symbol,
      point: 0.01,
      digits: 2,
      tickSize: 0.01,
      tickValue: 1,
      contractSize: 100,
      volumeStep: 0.01,
      minVolume: 0.01,
      maxVolume: 50,
      minStopDistance: 0.5
    };
  }

  async getSpread(_symbol: string): Promise<number> {
    this.ensureConnected();
    return this.spread;
  }

  async getCandles(_symbol: string, timeframe: Timeframe, limit: number): Promise<Candle[]> {
    this.ensureConnected();
    return (this.candlesByTimeframe[timeframe] ?? []).slice(-limit);
  }

  async placeOrder(order: OrderRequest): Promise<Position> {
    this.ensureConnected();
    if (this.positions.some((position) => position.symbol === order.symbol)) {
      throw new Error(`Duplicate position rejected for ${order.symbol}`);
    }
    const position: Position = {
      ...order,
      id: `MOCK-${Date.now()}-${Math.random().toString(16).slice(2)}`,
      openedAt: Date.now(),
      remainingVolume: order.volume
    };
    this.positions.push(position);
    logger.info("MockBroker order opened", position);
    return position;
  }

  async modifyOrder(orderId: string, updates: Partial<OrderRequest>): Promise<Position | undefined> {
    const position = this.positions.find((item) => item.id === orderId);
    if (!position) return undefined;
    Object.assign(position, updates);
    return position;
  }

  async closeOrder(orderId: string): Promise<Position | undefined> {
    const index = this.positions.findIndex((item) => item.id === orderId);
    if (index === -1) return undefined;
    const [position] = this.positions.splice(index, 1);
    const closed: TradeHistoryItem = { ...position, closedAt: Date.now(), realizedR: position.realizedR ?? 0 };
    this.history.push(closed);
    logger.info("MockBroker order closed", closed);
    return closed;
  }

  async getOpenPositions(): Promise<Position[]> {
    return [...this.positions];
  }

  async getTradeHistory(): Promise<TradeHistoryItem[]> {
    return [...this.history];
  }

  private ensureConnected(): void {
    if (!this.connected) throw new Error("MockBroker is not connected");
  }
}
