import logging
import os
from dataclasses import dataclass
from typing import Optional

import MetaTrader5 as mt5
import pandas as pd


LOGGER = logging.getLogger(__name__)


@dataclass
class OrderRequest:
    symbol: str
    side: str
    volume: float
    entry: float
    sl: float
    tp: float
    comment: str = "python-xauusd-bot"


class MT5Broker:
    def __init__(self, dry_run: Optional[bool] = None, magic: int = 26062026):
        self.dry_run = self._env_bool("DRY_RUN", True) if dry_run is None else dry_run
        self.magic = magic

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}

    def connect(self) -> None:
        if not mt5.initialize():
            raise RuntimeError(f"MetaTrader5 initialize failed: {mt5.last_error()}")
        LOGGER.info("Connected to MetaTrader 5. DRY_RUN=%s", self.dry_run)

    def shutdown(self) -> None:
        mt5.shutdown()
        LOGGER.info("Disconnected from MetaTrader 5")

    def account_info(self):
        account = mt5.account_info()
        if account is None:
            raise RuntimeError(f"Could not read account info: {mt5.last_error()}")
        return account

    def symbol_info(self, symbol: str):
        info = mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"Symbol not found: {symbol}")
        if not info.visible and not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Could not select symbol {symbol}: {mt5.last_error()}")
        return info

    def rates(self, symbol: str, timeframe: int, bars: int) -> pd.DataFrame:
        self.symbol_info(symbol)
        raw = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        if raw is None or len(raw) == 0:
            raise RuntimeError(f"No rates returned for {symbol}: {mt5.last_error()}")

        df = pd.DataFrame(raw)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df

    def tick(self, symbol: str):
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"Could not read tick for {symbol}: {mt5.last_error()}")
        return tick

    def open_positions(self, symbol: str):
        positions = mt5.positions_get(symbol=symbol)
        if positions is None:
            return []
        return list(positions)

    def has_open_position(self, symbol: str) -> bool:
        return len(self.open_positions(symbol)) > 0

    def send_market_order(self, request: OrderRequest):
        side = request.side.upper()
        if side not in {"BUY", "SELL"}:
            raise ValueError(f"Invalid order side: {request.side}")

        if self.dry_run:
            LOGGER.info(
                "DRY_RUN order skipped: %s %.2f %s entry=%.2f sl=%.2f tp=%.2f",
                side,
                request.volume,
                request.symbol,
                request.entry,
                request.sl,
                request.tp,
            )
            return {"dry_run": True, "request": request}

        order_type = mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL
        symbol_info = self.symbol_info(request.symbol)
        filling_modes = [
            int(symbol_info.filling_mode),
            mt5.ORDER_FILLING_IOC,
            mt5.ORDER_FILLING_FOK,
            mt5.ORDER_FILLING_RETURN,
        ]
        seen = set()
        filling_modes = [mode for mode in filling_modes if not (mode in seen or seen.add(mode))]

        payload = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": request.symbol,
            "volume": request.volume,
            "type": order_type,
            "price": request.entry,
            "sl": request.sl,
            "tp": request.tp,
            "deviation": 30,
            "magic": self.magic,
            "comment": request.comment,
            "type_time": mt5.ORDER_TIME_GTC,
        }

        errors = []
        for filling_mode in filling_modes:
            payload["type_filling"] = filling_mode
            check = mt5.order_check(payload)
            if check is not None and check.retcode != 0:
                errors.append(f"check filling={filling_mode} retcode={check.retcode} comment={check.comment}")
                continue

            result = mt5.order_send(payload)
            if result is None:
                errors.append(f"send filling={filling_mode} returned None last_error={mt5.last_error()}")
                continue
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                LOGGER.info("Order sent: %s", result)
                return result
            errors.append(f"send filling={filling_mode} retcode={result.retcode} comment={result.comment}")

        raise RuntimeError("order_send failed with all filling modes: " + " | ".join(errors))
