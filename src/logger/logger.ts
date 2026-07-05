type LogLevel = "info" | "warn" | "error" | "debug";

function formatMeta(meta: unknown): string {
  if (meta === undefined) return "";
  if (typeof meta !== "object" || meta === null) return " " + String(meta);

  const lines: string[] = [];
  for (const [key, value] of Object.entries(meta as Record<string, unknown>)) {
    if (Array.isArray(value)) {
      if (value.length === 0) {
        lines.push(`  ${key}: []`);
      } else {
        lines.push(`  ${key}:`);
        for (const item of value) {
          lines.push(`    - ${String(item)}`);
        }
      }
    } else if (typeof value === "object" && value !== null) {
      lines.push(`  ${key}: ${JSON.stringify(value)}`);
    } else {
      lines.push(`  ${key}: ${String(value)}`);
    }
  }
  return "\n" + lines.join("\n");
}

const LEVEL_RANK: Record<LogLevel, number> = { debug: 0, info: 1, warn: 2, error: 3 };

function minLevel(): number {
  const configured = (process.env.LOG_LEVEL ?? "info").toLowerCase() as LogLevel;
  return LEVEL_RANK[configured] ?? LEVEL_RANK.info;
}

function write(level: LogLevel, message: string, meta?: unknown): void {
  if (LEVEL_RANK[level] < minLevel()) return;
  const timestamp = new Date().toISOString();
  console.log(`${timestamp} ${level.toUpperCase()} ${message}${formatMeta(meta)}`);
}

export const logger = {
  info: (message: string, meta?: unknown) => write("info", message, meta),
  warn: (message: string, meta?: unknown) => write("warn", message, meta),
  error: (message: string, meta?: unknown) => write("error", message, meta),
  debug: (message: string, meta?: unknown) => write("debug", message, meta),
  separator: () => console.log("\n" + "-".repeat(60) + "\n")
};
