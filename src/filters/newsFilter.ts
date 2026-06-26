import { AppConfig, DetectorResult, NewsEvent } from "../types";

export interface NewsFilterResult extends DetectorResult {
  blocked: boolean;
  event?: NewsEvent;
}

export function newsFilter(timestamp: number, symbol: string, config: AppConfig): NewsFilterResult {
  if (!config.news.enabled) {
    return { blocked: false, score: 0, reasons: ["News filter disabled"] };
  }

  const beforeMs = config.news.blackoutMinutesBefore * 60 * 1000;
  const afterMs = config.news.blackoutMinutesAfter * 60 * 1000;
  const blockingEvent = config.news.events.find((event) => {
    const appliesToSymbol = !event.symbols || event.symbols.length === 0 || event.symbols.includes(symbol);
    return appliesToSymbol && timestamp >= event.time - beforeMs && timestamp <= event.time + afterMs;
  });

  if (!blockingEvent) {
    return { blocked: false, score: 0, reasons: ["No active news blackout"] };
  }

  return {
    blocked: true,
    event: blockingEvent,
    score: 0,
    reasons: [`News blackout active: ${blockingEvent.title}`]
  };
}
