import logging
import os
import time

import MetaTrader5 as mt5
from dotenv import load_dotenv

from broker import MT5Broker, OrderRequest
from risk import calculate_position_size
from strategy import build_signal


load_dotenv()

SYMBOL = os.getenv("SYMBOL", "GOLD")
RISK_PERCENT = float(os.getenv("RISK_PERCENT", "0.5"))
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "60"))
M5_BARS = int(os.getenv("M5_BARS", "300"))
H1_BARS = int(os.getenv("H1_BARS", "300"))
STARTUP_ORDER_ENABLED = os.getenv("STARTUP_ORDER_ENABLED", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
STARTUP_ORDER_SIDE = os.getenv("STARTUP_ORDER_SIDE", "BUY").strip().upper()
STARTUP_SL_PIPS = float(os.getenv("STARTUP_SL_PIPS", "5"))
STARTUP_TP_PIPS = float(os.getenv("STARTUP_TP_PIPS", "5"))
PIP_SIZE = float(os.getenv("PIP_SIZE", "0.1"))
ENFORCE_MIN_STOP = os.getenv("ENFORCE_MIN_STOP", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def fmt_level(value) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def round_price(price: float, symbol_info) -> float:
    digits = int(symbol_info.digits)
    return round(price, digits)


def min_stop_distance(symbol_info) -> float:
    point = float(symbol_info.point)
    stops_level = int(symbol_info.trade_stops_level or 0)
    freeze_level = int(symbol_info.trade_freeze_level or 0)
    return max(stops_level, freeze_level) * point


def enforce_minimum_stops(entry: float, sl: float, tp: float, side: str, symbol_info):
    minimum = min_stop_distance(symbol_info)
    if not ENFORCE_MIN_STOP or minimum <= 0:
        return round_price(sl, symbol_info), round_price(tp, symbol_info), False, minimum

    changed = False
    if side == "BUY":
        if entry - sl < minimum:
            sl = entry - minimum
            changed = True
        if tp - entry < minimum:
            tp = entry + minimum
            changed = True
    else:
        if sl - entry < minimum:
            sl = entry + minimum
            changed = True
        if entry - tp < minimum:
            tp = entry - minimum
            changed = True

    return round_price(sl, symbol_info), round_price(tp, symbol_info), changed, minimum


def run_startup_order_once(broker: MT5Broker) -> None:
    logger = logging.getLogger("bot")
    if not STARTUP_ORDER_ENABLED:
        logger.info("STARTUP ORDER disabled")
        return

    if STARTUP_ORDER_SIDE not in {"BUY", "SELL"}:
        logger.warning("STARTUP ORDER skipped: invalid STARTUP_ORDER_SIDE=%s", STARTUP_ORDER_SIDE)
        return

    if broker.has_open_position(SYMBOL):
        logger.info("STARTUP ORDER skipped: open position already exists on %s", SYMBOL)
        return

    symbol_info = broker.symbol_info(SYMBOL)
    account = broker.account_info()
    tick = broker.tick(SYMBOL)
    entry = float(tick.ask if STARTUP_ORDER_SIDE == "BUY" else tick.bid)
    sl_distance = STARTUP_SL_PIPS * PIP_SIZE
    tp_distance = STARTUP_TP_PIPS * PIP_SIZE

    if STARTUP_ORDER_SIDE == "BUY":
        sl = entry - sl_distance
        tp = entry + tp_distance
    else:
        sl = entry + sl_distance
        tp = entry - tp_distance
    sl, tp, stops_changed, minimum = enforce_minimum_stops(entry, sl, tp, STARTUP_ORDER_SIDE, symbol_info)

    volume = calculate_position_size(
        balance=float(account.balance),
        risk_percent=RISK_PERCENT,
        entry=entry,
        stop_loss=sl,
        symbol_info=symbol_info,
    )

    logger.info(
        (
            "STARTUP %s %s: entry=%.2f sl=%.2f tp=%.2f "
            "sl_pips=%.1f tp_pips=%.1f pip_size=%.4f volume=%.2f"
        ),
        STARTUP_ORDER_SIDE,
        SYMBOL,
        entry,
        sl,
        tp,
        STARTUP_SL_PIPS,
        STARTUP_TP_PIPS,
        PIP_SIZE,
        volume,
    )
    if stops_changed:
        logger.warning(
            "STARTUP ORDER adjusted SL/TP to broker minimum stop distance %.5f. New sl=%.2f tp=%.2f",
            minimum,
            sl,
            tp,
        )

    broker.send_market_order(
        OrderRequest(
            symbol=SYMBOL,
            side=STARTUP_ORDER_SIDE,
            volume=volume,
            entry=entry,
            sl=sl,
            tp=tp,
            comment="startup-test-order",
        )
    )


def run_once(broker: MT5Broker) -> None:
    m5 = broker.rates(SYMBOL, mt5.TIMEFRAME_M5, M5_BARS)
    h1 = broker.rates(SYMBOL, mt5.TIMEFRAME_H1, H1_BARS)
    signal = build_signal(m5, h1)

    logger = logging.getLogger("bot")
    if signal.action == "WAIT":
        logger.info("WAIT %s: %s", SYMBOL, signal.reason)
        return

    if broker.has_open_position(SYMBOL):
        logger.info("WAIT %s: open position already exists", SYMBOL)
        return

    symbol_info = broker.symbol_info(SYMBOL)
    account = broker.account_info()
    tick = broker.tick(SYMBOL)
    entry = float(tick.ask if signal.action == "BUY" else tick.bid)
    risk_distance = abs(entry - float(signal.sl))
    tp = entry + (risk_distance * 2.0) if signal.action == "BUY" else entry - (risk_distance * 2.0)
    sl, tp, stops_changed, minimum = enforce_minimum_stops(entry, float(signal.sl), tp, signal.action, symbol_info)

    volume = calculate_position_size(
        balance=float(account.balance),
        risk_percent=RISK_PERCENT,
        entry=entry,
        stop_loss=sl,
        symbol_info=symbol_info,
    )
    if stops_changed:
        logger.warning(
            "%s %s: adjusted SL/TP to broker minimum stop distance %.5f. New sl=%.2f tp=%.2f",
            signal.action,
            SYMBOL,
            minimum,
            sl,
            tp,
        )

    logger.info(
        (
            "%s %s: %s entry=%.2f sl=%.2f tp=%.2f volume=%.2f "
            "fvg=%s-%s ob=%s-%s ob_ok=%s poc=%s poc_ok=%s liquidity=%s"
        ),
        signal.action,
        SYMBOL,
        signal.reason,
        entry,
        sl,
        tp,
        volume,
        fmt_level(signal.fvg_low),
        fmt_level(signal.fvg_high),
        fmt_level(signal.order_block_low),
        fmt_level(signal.order_block_high),
        signal.order_block_confluence,
        fmt_level(signal.poc),
        signal.poc_confluence,
        fmt_level(signal.liquidity_level),
    )

    broker.send_market_order(
        OrderRequest(
            symbol=SYMBOL,
            side=signal.action,
            volume=volume,
            entry=entry,
            sl=sl,
            tp=float(tp),
        )
    )


def main() -> None:
    configure_logging()
    broker = MT5Broker()
    broker.connect()
    try:
        try:
            run_startup_order_once(broker)
        except Exception:
            logging.exception("Startup order failed")
        while True:
            try:
                run_once(broker)
            except Exception:
                logging.exception("Bot cycle failed")
            time.sleep(POLL_SECONDS)
    finally:
        broker.shutdown()


if __name__ == "__main__":
    main()
