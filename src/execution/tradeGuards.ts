import { getSession } from "../market/sessions";
import { AppConfig } from "../types";

export class TradeGuard {
  private cooldownUntil = 0;
  private sessionCounts = new Map<string, number>();

  constructor(private readonly config: AppConfig) {}

  canOpen(timestamp: number): { allowed: boolean; reasons: string[] } {
    const reasons: string[] = [];
    if (timestamp < this.cooldownUntil) {
      reasons.push(`Cooldown active until ${new Date(this.cooldownUntil).toISOString()}`);
    }

    const key = this.sessionKey(timestamp);
    const count = this.sessionCounts.get(key) ?? 0;
    const maxTrades = this.config.tradeGuards.maxTradesPerSession;
    if (maxTrades > 0 && count >= maxTrades) {
      reasons.push(`Max trades reached for session ${key}`);
    }

    return { allowed: reasons.length === 0, reasons };
  }

  recordOpen(timestamp: number): void {
    const key = this.sessionKey(timestamp);
    this.sessionCounts.set(key, (this.sessionCounts.get(key) ?? 0) + 1);
  }

  recordClose(timestamp: number, profit: number): void {
    if (profit < 0) {
      this.cooldownUntil = timestamp + this.config.tradeGuards.cooldownAfterLossMinutes * 60 * 1000;
    }
  }

  state(): { cooldownUntil: number; sessionCounts: Record<string, number> } {
    return {
      cooldownUntil: this.cooldownUntil,
      sessionCounts: Object.fromEntries(this.sessionCounts)
    };
  }

  private sessionKey(timestamp: number): string {
    const date = new Date(timestamp).toISOString().slice(0, 10);
    return `${date}:${getSession(timestamp, this.config.sessions.utcOffsetMinutes)}`;
  }
}
