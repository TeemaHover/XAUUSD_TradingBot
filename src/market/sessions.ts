import { SessionName } from "../types";

export function getSession(timestamp: number, utcOffsetMinutes = 0): SessionName {
  const hour = new Date(timestamp + utcOffsetMinutes * 60 * 1000).getUTCHours();
  if (hour >= 0 && hour < 7) return "Asian";
  if (hour >= 7 && hour < 13) return "London";
  if (hour >= 13 && hour < 21) return "NewYork";
  return "OffSession";
}
