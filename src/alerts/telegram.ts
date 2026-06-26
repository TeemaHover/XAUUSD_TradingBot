import https from "node:https";
import { AppConfig } from "../types";
import { logger } from "../logger/logger";

export class TelegramAlerts {
  constructor(private readonly config: AppConfig) {}

  async send(message: string): Promise<void> {
    const telegram = this.config.alerts.telegram;
    if (!telegram.enabled) return;
    if (!telegram.botToken || !telegram.chatId) {
      logger.warn("Telegram alert skipped: missing botToken or chatId");
      return;
    }

    const body = JSON.stringify({ chat_id: telegram.chatId, text: message });
    const options: https.RequestOptions = {
      method: "POST",
      hostname: "api.telegram.org",
      path: `/bot${telegram.botToken}/sendMessage`,
      headers: {
        "Content-Type": "application/json",
        "Content-Length": Buffer.byteLength(body)
      },
      timeout: 5000
    };

    await new Promise<void>((resolve) => {
      const req = https.request(options, (res) => {
        res.resume();
        res.on("end", () => {
          if ((res.statusCode ?? 500) >= 400) {
            logger.warn("Telegram alert failed", { statusCode: res.statusCode });
          }
          resolve();
        });
      });
      req.on("error", (error) => {
        logger.warn("Telegram alert error", { message: error.message });
        resolve();
      });
      req.write(body);
      req.end();
    });
  }
}
