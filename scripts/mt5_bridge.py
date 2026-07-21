import json
import math
import sys
from datetime import datetime, time, timezone
from typing import Any, Dict

import MetaTrader5 as mt5


TIMEFRAMES = {
    "5m": mt5.TIMEFRAME_M5,
    "15m": mt5.TIMEFRAME_M15,
    "1h": mt5.TIMEFRAME_H1,
    "4h": mt5.TIMEFRAME_H4,
    "1d": mt5.TIMEFRAME_D1,
}


def fail(message: str) -> None:
    print(json.dumps({"ok": False, "error": message}))
    raise SystemExit(1)


def ensure_symbol(symbol: str):
    info = mt5.symbol_info(symbol)
    if info is None:
        fail(f"Symbol not found in MT5: {symbol}")
    if not info.visible and not mt5.symbol_select(symbol, True):
        fail(f"Could not select symbol: {symbol}")
    return info


def candle_from_rate(rate) -> Dict[str, Any]:
    return {
        "time": int(rate["time"]) * 1000,
        "open": float(rate["open"]),
        "high": float(rate["high"]),
        "low": float(rate["low"]),
        "close": float(rate["close"]),
        "volume": float(rate["tick_volume"]),
    }


def position_to_dict(position) -> Dict[str, Any]:
    direction = "long" if position.type == mt5.POSITION_TYPE_BUY else "short"
    return {
        "id": str(position.ticket),
        "symbol": position.symbol,
        "direction": direction,
        "volume": float(position.volume),
        "remainingVolume": float(position.volume),
        "entry": float(position.price_open),
        "stopLoss": float(position.sl),
        "takeProfits": [float(position.tp)] if float(position.tp) > 0 else [],
        "openedAt": int(position.time) * 1000,
        "comment": position.comment,
    }


def deal_to_history_item(deal) -> Dict[str, Any]:
    direction = "long" if deal.type == mt5.DEAL_TYPE_BUY else "short"
    profit = float(deal.profit + deal.commission + deal.swap)
    return {
        "id": str(deal.ticket),
        "symbol": deal.symbol,
        "direction": direction,
        "volume": float(deal.volume),
        "remainingVolume": 0,
        "entry": float(deal.price),
        "stopLoss": 0,
        "takeProfits": [],
        "openedAt": int(deal.time) * 1000,
        "closedAt": int(deal.time) * 1000,
        "realizedR": 0,
        "profit": profit,
        "comment": deal.comment,
    }


def symbol_spec(symbol: str) -> Dict[str, Any]:
    info = ensure_symbol(symbol)
    point = float(info.point)
    stops_level = int(info.trade_stops_level or 0)
    freeze_level = int(info.trade_freeze_level or 0)
    return {
        "symbol": symbol,
        "point": point,
        "digits": int(info.digits),
        "tickSize": float(info.trade_tick_size),
        "tickValue": float(info.trade_tick_value),
        "contractSize": float(info.trade_contract_size),
        "volumeStep": float(info.volume_step),
        "minVolume": float(info.volume_min),
        "maxVolume": float(info.volume_max),
        "minStopDistance": max(stops_level, freeze_level) * point,
    }


def round_volume(volume: float, info) -> float:
    step = float(info.volume_step)
    min_volume = float(info.volume_min)
    max_volume = float(info.volume_max)
    if step <= 0:
        return min(max(volume, min_volume), max_volume)
    rounded = math.floor(volume / step) * step
    return round(min(max(rounded, min_volume), max_volume), 8)


def validate_stops(direction: str, price: float, stop_loss: float, take_profit: float, info) -> None:
    point = float(info.point)
    minimum = max(int(info.trade_stops_level or 0), int(info.trade_freeze_level or 0)) * point
    if minimum <= 0:
        return
    if direction == "long":
        if price - stop_loss < minimum:
            fail(f"Stop loss too close. Required minimum distance={minimum}")
        if take_profit > 0 and take_profit - price < minimum:
            fail(f"Take profit too close. Required minimum distance={minimum}")
    else:
        if stop_loss - price < minimum:
            fail(f"Stop loss too close. Required minimum distance={minimum}")
        if take_profit > 0 and price - take_profit < minimum:
            fail(f"Take profit too close. Required minimum distance={minimum}")


def order_type(direction: str):
    if direction == "long":
        return mt5.ORDER_TYPE_BUY
    if direction == "short":
        return mt5.ORDER_TYPE_SELL
    fail(f"Invalid direction: {direction}")


def handle(command: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    symbol = payload.get("symbol")

    if command == "ping":
        account = mt5.account_info()
        return {"connected": account is not None}

    if command == "balance":
        account = mt5.account_info()
        if account is None:
            fail(f"Could not read account info: {mt5.last_error()}")
        return {"balance": float(account.balance)}

    if command == "spread":
        info = ensure_symbol(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            fail(f"Could not read tick for {symbol}: {mt5.last_error()}")
        spread = float(tick.ask - tick.bid)
        if spread <= 0 and info.point > 0:
            spread = float(info.spread * info.point)
        return {"spread": spread, "bid": float(tick.bid), "ask": float(tick.ask)}

    if command == "symbol_info":
        return {"spec": symbol_spec(symbol)}

    if command == "candles":
        ensure_symbol(symbol)
        timeframe = payload.get("timeframe")
        limit = int(payload.get("limit", 500))
        from_ts = payload.get("fromTimestamp")   # Unix ms; if set, use date-range fetch
        mt5_timeframe = TIMEFRAMES.get(timeframe)
        if mt5_timeframe is None:
            fail(f"Unsupported timeframe: {timeframe}")

        rates = None
        if from_ts is not None:
            # Try date-range first
            from_dt = datetime.utcfromtimestamp(int(from_ts) / 1000)
            to_dt   = datetime.utcnow()
            rates   = mt5.copy_rates_range(symbol, mt5_timeframe, from_dt, to_dt)

        if rates is None or len(rates) == 0:
            # Fall back: try progressively smaller bar counts until one succeeds
            # (MT5 terminal "Max bars in chart" setting varies; default often 50000)
            # Start at pos 1, NOT 0: pos 0 is the still-forming candle. Models and
            # backtests only ever see closed bars, and the bot's once-per-candle
            # gate keys off the last candle time — a partial bar would break both.
            for try_limit in [99000, 50000, 30000, 10000, limit]:
                rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 1, try_limit)
                if rates is not None and len(rates) > 0:
                    break

        if rates is None or len(rates) == 0:
            fail(f"No candles returned for {symbol}: {mt5.last_error()}")
        return {"candles": [candle_from_rate(rate) for rate in rates]}

    if command == "positions":
        positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        if positions is None:
            return {"positions": []}
        return {"positions": [position_to_dict(position) for position in positions]}

    if command == "history":
        from_dt = datetime.combine(datetime.now(timezone.utc).date(), time.min, tzinfo=timezone.utc)
        to_dt = datetime.now(timezone.utc)
        deals = mt5.history_deals_get(from_dt, to_dt)
        if deals is None:
            return {"history": []}
        magic = int(payload.get("magic", 26062026))
        filtered = [
            deal for deal in deals
            if (not symbol or deal.symbol == symbol)
            and int(deal.magic) == magic
            and deal.entry in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT, mt5.DEAL_ENTRY_OUT_BY)
        ]
        return {"history": [deal_to_history_item(deal) for deal in filtered]}

    if command == "order":
        symbol_info = ensure_symbol(symbol)
        direction = payload["direction"]
        volume = round_volume(float(payload["volume"]), symbol_info)
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            fail(f"Could not read tick for {symbol}: {mt5.last_error()}")
        is_limit = payload.get("entryType") == "limit"
        market_price = float(tick.ask if direction == "long" else tick.bid)
        stop_loss = float(payload["stopLoss"])
        take_profits = payload.get("takeProfits") or []
        take_profit = float(take_profits[0]) if take_profits else 0.0
        dry_run = bool(payload.get("dryRun", True))
        magic = int(payload.get("magic", 26062026))

        if is_limit:
            entry_price = float(payload.get("entry") or 0.0)
            if entry_price <= 0:
                fail("Limit order requires a positive entry price")
            # Guard: never stack pending orders — the bot loop re-emits the same
            # signal every few seconds while price approaches the zone.
            existing = mt5.orders_get(symbol=symbol) or []
            for pending in existing:
                if int(pending.magic) == magic:
                    return {
                        "ticket": str(pending.ticket),
                        "price": float(pending.price_open),
                        "volume": float(pending.volume_initial),
                        "pending": True,
                        "duplicate": True,
                    }
            # A buy limit must sit below the current ask, sell limit above the bid;
            # otherwise fall back to a market order.
            if (direction == "long" and entry_price >= market_price) or (
                direction == "short" and entry_price <= market_price
            ):
                is_limit = False

        price = entry_price if is_limit else market_price
        validate_stops(direction, price, stop_loss, take_profit, symbol_info)

        if is_limit:
            pending_type = mt5.ORDER_TYPE_BUY_LIMIT if direction == "long" else mt5.ORDER_TYPE_SELL_LIMIT
            base_request = {
                "action": mt5.TRADE_ACTION_PENDING,
                "symbol": symbol,
                "volume": volume,
                "type": pending_type,
                "price": price,
                "sl": stop_loss,
                "tp": take_profit,
                "magic": magic,
                "comment": payload.get("comment", "typescript-mt5-bot"),
                "type_time": mt5.ORDER_TIME_GTC,
            }
            expiry_seconds = payload.get("expirySeconds")
            if expiry_seconds:
                from datetime import timedelta
                base_request["type_time"] = mt5.ORDER_TIME_SPECIFIED
                base_request["expiration"] = datetime.now(timezone.utc) + timedelta(seconds=int(expiry_seconds))
        else:
            base_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type(direction),
                "price": price,
                "sl": stop_loss,
                "tp": take_profit,
                "deviation": int(payload.get("deviation", 30)),
                "magic": magic,
                "comment": payload.get("comment", "typescript-mt5-bot"),
                "type_time": mt5.ORDER_TIME_GTC,
            }
        # symbol_info.filling_mode is a bitmask:
        #   bit 0 (value 1) = FOK supported  -> ORDER_FILLING_FOK = 0
        #   bit 1 (value 2) = IOC supported  -> ORDER_FILLING_IOC = 1
        #   bit 2 (value 4) = RETURN supported -> ORDER_FILLING_RETURN = 2
        filling_mask = int(symbol_info.filling_mode)
        filling_modes = []
        if filling_mask & 1:
            filling_modes.append(mt5.ORDER_FILLING_FOK)
        if filling_mask & 2:
            filling_modes.append(mt5.ORDER_FILLING_IOC)
        if filling_mask & 4:
            filling_modes.append(mt5.ORDER_FILLING_RETURN)
        # Fallback: try all three if mask gave nothing useful
        if not filling_modes:
            filling_modes = [mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN]

        errors = []
        for filling_mode in filling_modes:
            request = dict(base_request)
            request["type_filling"] = filling_mode
            check = mt5.order_check(request)
            if check is not None and check.retcode not in (0, mt5.TRADE_RETCODE_DONE):
                errors.append(f"check filling={filling_mode} retcode={check.retcode} comment={check.comment}")
                continue

            if dry_run:
                return {"dryRun": True, "request": request, "volume": volume, "price": price}

            result = mt5.order_send(request)
            if result is None:
                errors.append(f"send filling={filling_mode} returned None error={mt5.last_error()}")
                continue
            # Some brokers reject ORDER_TIME_SPECIFIED on pendings — retry as GTC
            if result.retcode != mt5.TRADE_RETCODE_DONE and "expiration" in request:
                request = dict(request)
                request.pop("expiration", None)
                request["type_time"] = mt5.ORDER_TIME_GTC
                result = mt5.order_send(request)
                if result is None:
                    errors.append(f"send filling={filling_mode} GTC retry returned None error={mt5.last_error()}")
                    continue
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                return {
                    "ticket": str(result.order),
                    "price": float(result.price) or price,
                    "volume": float(result.volume) or volume,
                    "pending": is_limit,
                }
            errors.append(f"send filling={filling_mode} retcode={result.retcode} comment={result.comment}")

        fail("Order failed with all filling modes: " + " | ".join(errors))


    if command == "ai_predict":
        import os as _os
        import sys as _sys
        import numpy as _np
        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        try:
            from ai_features import extract_sequence, extract_multi_tf_sequence, SEQ_LEN, CONTEXT_LOOKBACK
            from ai_model    import load_model, MTF_TFS
        except ImportError as e:
            fail(f"AI modules not found: {e}")

        model_path = payload.get("modelPath", "models/ai_model.npz")
        symbol     = payload.get("symbol", "GOLD")

        if not _os.path.exists(model_path):
            return {"direction": "hold", "confidence": 0.0,
                    "reason": f"model not found at {model_path} -- run ai_train.py first"}

        # Detect model type from the .npz
        _meta = _np.load(model_path, allow_pickle=True)
        _mtype = str(_meta.get("model_type", _np.array("single_tf")))

        model = load_model(model_path)
        _features = model.feature_names   # extract exactly what the model was trained on
        needed = SEQ_LEN + 1
        fetch = max(needed, CONTEXT_LOOKBACK)   # context features want more history

        if _mtype == "multi_tf":
            # ── Multi-TF: fetch 5m, 1h, 4h candles from MT5 directly ──
            tf_map = {"5m": mt5.TIMEFRAME_M5,
                      "1h": mt5.TIMEFRAME_H1,
                      "4h": mt5.TIMEFRAME_H4}
            candles_dict = {}
            for tf_name, mt5_tf in tf_map.items():
                # Start at pos 1, NOT 0: pos 0 is the still-forming candle.
                # The model is trained on closed candles only (close-time
                # alignment), so feeding it a partial bar would mismatch
                # training and reintroduce inconsistency with the backtest.
                rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 1, fetch)
                if rates is None or len(rates) == 0:
                    return {"direction": "hold", "confidence": 0.0,
                            "reason": f"no {tf_name} candles from MT5 for {symbol}"}
                candles_dict[tf_name] = [
                    {"time":   int(r["time"]) * 1000,
                     "open":   float(r["open"]),
                     "high":   float(r["high"]),
                     "low":    float(r["low"]),
                     "close":  float(r["close"]),
                     "volume": float(r["tick_volume"])}
                    for r in rates
                ]
            seqs = extract_multi_tf_sequence(candles_dict, features=_features)
            direction, confidence = model.predict_one(seqs)

        else:
            # ── Single-TF: use candles passed in payload (or fetch 5m) ──
            candles = payload.get("candles", [])
            if len(candles) < needed:
                # Fallback: fetch from MT5 (pos 1 = closed candles only,
                # consistent with training)
                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 1, fetch)
                if rates is None or len(rates) == 0:
                    return {"direction": "hold", "confidence": 0.0,
                            "reason": f"no candles from MT5 for {symbol}"}
                candles = [
                    {"time":   int(r["time"]) * 1000,
                     "open":   float(r["open"]),
                     "high":   float(r["high"]),
                     "low":    float(r["low"]),
                     "close":  float(r["close"]),
                     "volume": float(r["tick_volume"])}
                    for r in rates
                ]
            seq = extract_sequence(candles, features=_features)
            direction, confidence = model.predict_one(seq)

        return {"direction": direction, "confidence": confidence}

    fail(f"Unsupported command: {command}")


def main() -> None:
    raw = sys.stdin.read()
    payload = json.loads(raw or "{}")
    command = payload.get("command")
    args = payload.get("args", {})

    if not mt5.initialize():
        fail(f"MetaTrader5 initialize failed: {mt5.last_error()}")

    try:
        result = handle(command, args)
        print(json.dumps({"ok": True, "result": result}))
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
