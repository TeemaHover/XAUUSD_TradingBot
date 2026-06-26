import math


def round_volume(volume: float, min_volume: float, max_volume: float, step: float) -> float:
    if step <= 0:
        return max(min_volume, min(volume, max_volume))

    steps = math.floor(volume / step)
    rounded = steps * step
    return max(min_volume, min(rounded, max_volume))


def calculate_position_size(
    balance: float,
    risk_percent: float,
    entry: float,
    stop_loss: float,
    symbol_info,
) -> float:
    risk_amount = balance * (risk_percent / 100.0)
    price_distance = abs(entry - stop_loss)
    if price_distance <= 0:
        raise ValueError("Stop loss distance must be greater than zero")

    tick_size = float(symbol_info.trade_tick_size)
    tick_value = float(symbol_info.trade_tick_value)
    if tick_size <= 0 or tick_value <= 0:
        raise ValueError("Symbol tick size/value is invalid")

    loss_per_lot = (price_distance / tick_size) * tick_value
    if loss_per_lot <= 0:
        raise ValueError("Calculated loss per lot is invalid")

    raw_volume = risk_amount / loss_per_lot
    return round_volume(
        raw_volume,
        float(symbol_info.volume_min),
        float(symbol_info.volume_max),
        float(symbol_info.volume_step),
    )
