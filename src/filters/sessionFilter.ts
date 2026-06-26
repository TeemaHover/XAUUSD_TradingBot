import { AppConfig, DetectorResult } from "../types";
import { getSession } from "../market/sessions";

export function sessionFilter(timestamp: number, config: AppConfig): DetectorResult {
  if (!config.sessions.enabled) {
    return { score: config.scoring.sessionAllowed, reasons: ["Session filter disabled"] };
  }

  const session = getSession(timestamp, config.sessions.utcOffsetMinutes);
  const allowed = config.sessions.allowed.includes(session);
  return {
    score: allowed ? config.scoring.sessionAllowed : 0,
    reasons: [allowed ? `Session allowed: ${session}` : `Session rejected: ${session}`]
  };
}
