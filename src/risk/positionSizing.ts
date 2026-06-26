export interface PositionSizingSpec {
  tickSize: number;
  tickValue: number;
  volumeStep: number;
  minVolume: number;
  maxVolume: number;
}

function roundDownToStep(value: number, step: number): number {
  if (step <= 0) return value;
  return Math.floor(value / step) * step;
}

export function normalizeVolume(volume: number, spec: PositionSizingSpec): number {
  const stepped = roundDownToStep(volume, spec.volumeStep);
  const clamped = Math.min(Math.max(stepped, spec.minVolume), spec.maxVolume);
  return Number(clamped.toFixed(8));
}

export function calculatePositionSize(
  balance: number,
  riskPerTrade: number,
  entry: number,
  stopLoss: number,
  spec: PositionSizingSpec
): number {
  const riskAmount = balance * riskPerTrade;
  const stopDistance = Math.abs(entry - stopLoss);
  if (stopDistance <= 0) throw new Error("Stop distance must be greater than zero");
  if (spec.tickSize <= 0 || spec.tickValue <= 0) {
    throw new Error("Invalid tick size or tick value");
  }

  const lossPerLot = (stopDistance / spec.tickSize) * spec.tickValue;
  if (lossPerLot <= 0) throw new Error("Invalid loss per lot");

  return normalizeVolume(riskAmount / lossPerLot, spec);
}

export class RiskGuard {
  private consecutiveLosses = 0;
  private dailyLoss = 0;

  constructor(
    private readonly maxDailyLoss: number,
    private readonly maxConsecutiveLosses: number,
    private readonly startingBalance: number
  ) {}

  canTrade(): boolean {
    return this.dailyLoss < this.startingBalance * this.maxDailyLoss
      && this.consecutiveLosses < this.maxConsecutiveLosses;
  }

  status(): { canTrade: boolean; dailyLoss: number; consecutiveLosses: number; maxDailyLossAmount: number } {
    return {
      canTrade: this.canTrade(),
      dailyLoss: this.dailyLoss,
      consecutiveLosses: this.consecutiveLosses,
      maxDailyLossAmount: this.startingBalance * this.maxDailyLoss
    };
  }

  hydrateFromClosedProfits(profits: number[]): void {
    this.dailyLoss = profits.filter((profit) => profit < 0).reduce((sum, profit) => sum + Math.abs(profit), 0);
    this.consecutiveLosses = 0;
    for (const profit of [...profits].reverse()) {
      if (profit < 0) {
        this.consecutiveLosses += 1;
      } else {
        break;
      }
    }
  }

  recordResult(profit: number): void {
    if (profit < 0) {
      this.consecutiveLosses += 1;
      this.dailyLoss += Math.abs(profit);
      return;
    }

    this.consecutiveLosses = 0;
  }
}
