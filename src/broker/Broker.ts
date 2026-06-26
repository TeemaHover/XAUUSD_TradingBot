import { Candle, OrderRequest, Position, SymbolSpec, Timeframe, TradeHistoryItem } from "../types";

export interface Broker {
  connect(): Promise<void>;
  disconnect(): Promise<void>;
  getBalance(): Promise<number>;
  getSymbolSpec(symbol: string): Promise<SymbolSpec>;
  getSpread(symbol: string): Promise<number>;
  getCandles(symbol: string, timeframe: Timeframe, limit: number): Promise<Candle[]>;
  placeOrder(order: OrderRequest): Promise<Position>;
  modifyOrder(orderId: string, updates: Partial<OrderRequest>): Promise<Position | undefined>;
  closeOrder(orderId: string): Promise<Position | undefined>;
  getOpenPositions(): Promise<Position[]>;
  getTradeHistory(): Promise<TradeHistoryItem[]>;
}
