# GOLD TypeScript Trading Bot

Modular TypeScript trading bot for GOLD. This is a production-quality first version with clean interfaces, a mock broker, configurable strategy values, unit tests, and a CSV backtester.

The older Python MT5 files in this folder are separate from this TypeScript app.

## Features

- Clean modular architecture
- Config-driven strategy values in `config/default.json`
- Mock broker that runs without real money
- Flexible scoring system instead of all-or-nothing rules
- Trend, liquidity, MSS/CHOCH, order block, FVG, volume, session, and volatility modules
- Risk-based position sizing
- Gold-style tick size/tick value and volume-step sizing from config
- TP1/TP2/TP3 using 1R/2R/3R
- Basic candle-by-candle backtesting
- Spread, slippage, commission, partial TPs, and breakeven simulation in backtests
- Walk-forward backtesting
- News blackout windows
- Market regime detection: trending, ranging, high volatility, low volatility
- Adaptive scoring by regime
- Cooldown after losses and max trades per session
- Optional Telegram alerts
- Dashboard endpoint for signal score, open trades, and daily PnL
- Console logs for signals, rejected trades, executed mock trades, and backtest results

## Install

```powershell
npm install
```

## Build

```powershell
npm run build
```

## Run With MockBroker

```powershell
npm run build
npm start
```

This uses generated sample candles and the `MockBroker`. It does not connect to a real broker and does not place real trades.

To use the mock broker, set:

```json
"broker": {
  "mode": "mock"
}
```

## Run With MetaTrader 5 Demo Data

The TypeScript bot can read candles, balance, spread, and open positions from your locally running MT5 desktop terminal through `scripts/mt5_bridge.py`.

Requirements:

```powershell
pip install MetaTrader5
```

MT5 must be:

- installed on the same Windows computer
- open and logged into your demo account
- showing the configured symbol in Market Watch

Set:

```json
"symbol": "GOLD",
"broker": {
  "mode": "mt5"
},
"mt5": {
  "pythonPath": "python",
  "bridgePath": "scripts/mt5_bridge.py",
  "dryRun": true
}
```

This project is configured for your MT5 symbol:

```json
"symbol": "GOLD"
```

Run:

```powershell
npm run build
npm start
```

By default, the bot scans immediately and then repeats every 60 seconds:

```json
"bot": {
  "loopEnabled": true,
  "intervalSeconds": 60
}
```

You can override this without editing JSON:

```powershell
$env:BOT_LOOP_ENABLED="true"
$env:BOT_INTERVAL_SECONDS="60"
npm start
```

Set `BOT_LOOP_ENABLED="false"` if you only want one signal calculation.

With `dryRun: true`, the bot reads real MT5 demo data but does not place orders. Only set `dryRun: false` after confirming the logs and only on demo.

If the dashboard is enabled, `npm start` keeps running until `Ctrl+C`.

Dashboard:

```text
http://127.0.0.1:8787
```

## Run Tests

```powershell
npm run build
npm test
```

## Run Backtest

Prepare a CSV file with:

```text
time,open,high,low,close,volume
```

`time` should be a Unix millisecond timestamp.

Run:

```powershell
npm run build
npm run backtest -- data/xauusd.csv backtest-results.json
```

Walk-forward:

```powershell
npm run backtest -- data/xauusd.csv walk-forward-results.json --walk-forward
```

Backtest output includes:

- total trades
- win rate
- profit factor
- max drawdown
- expectancy
- total return
- average R

## Config

Edit [config/default.json](D:/TradingBot/config/default.json).

Important values:

```json
{
  "symbol": "GOLD",
  "bot": {
    "loopEnabled": true,
    "intervalSeconds": 60
  },
  "risk": {
    "riskPerTrade": 0.005,
    "maxDailyLoss": 0.03,
    "maxConsecutiveLosses": 3,
    "minStopDistance": 0.5,
    "stopBufferAtr": 0.1,
    "tickSize": 0.01,
    "tickValue": 1,
    "volumeStep": 0.01,
    "minVolume": 0.01,
    "maxVolume": 50
  },
  "strategy": {
    "minScore": 75,
    "watchlistScore": 60,
    "maxSpread": 1,
    "allowCounterTrendTrades": false,
    "counterTrendMinScore": 85
  },
  "mockBroker": {
    "spread": 0.25,
    "slippage": 0.05,
    "commissionPerLot": 7
  },
  "news": {
    "enabled": true,
    "blackoutMinutesBefore": 30,
    "blackoutMinutesAfter": 30,
    "events": []
  },
  "tradeGuards": {
    "cooldownAfterLossMinutes": 60,
    "maxTradesPerSession": 2
  },
  "dashboard": {
    "enabled": true,
    "host": "127.0.0.1",
    "port": 8787
  },
  "sessions": {
    "allowed": ["London", "NewYork"]
  }
}
```

## Scoring

Each module contributes points:

- Trend alignment: 30
- Liquidity sweep: 20
- MSS/CHOCH: 20
- Order block: 15
- FVG: 10
- Volume confirmation: 15
- Session allowed: 10
- Volatility valid: 10

The final score is normalized to `0-100`.

- `>= 75`: valid trade
- `60-74`: watchlist only
- `< 60`: rejected

The engine logs rejection reasons such as missing directional context, missing confirmation candle, low score, invalid stop distance, session rejection, or volatility rejection.

## V2 Controls

News event example:

```json
{
  "time": 1767182400000,
  "title": "US CPI",
  "impact": "high",
  "symbols": ["GOLD"]
}
```

Telegram alerts are disabled by default. To enable:

```json
{
  "alerts": {
    "telegram": {
      "enabled": true,
      "botToken": "YOUR_TOKEN",
      "chatId": "YOUR_CHAT_ID"
    }
  }
}
```

Or set them from PowerShell before starting the bot:

```powershell
$env:TELEGRAM_ENABLED="true"
$env:TELEGRAM_BOT_TOKEN="YOUR_TOKEN"
$env:TELEGRAM_CHAT_ID="YOUR_CHAT_ID"
npm start
```

The bot sends Telegram messages when:

- the bot starts successfully
- an order is opened by the broker
- the bot stops

It does not send an open-position alert for rejected signals or watchlist signals.

Adaptive scoring is configured under `adaptiveScoring.regimeScoreMultipliers`. Regime detection uses ADX plus ATR relative to recent ATR.

## Backtest Assumptions

The backtester avoids same-close fills by entering on the next candle open with configured spread and slippage. If stop loss and targets could both be touched inside the same candle, stop loss is handled conservatively first.

It simulates:

- TP1/TP2/TP3 partial exits
- optional breakeven after TP1
- commission per lot
- spread and slippage
- max daily loss guard
- consecutive loss guard

It is still a simple OHLC backtester. It does not know tick order inside a candle, real broker rejection rules, news spread spikes, or exact MT5 contract metadata.

## Real Broker Later

Add a real broker by implementing [src/broker/Broker.ts](D:/TradingBot/src/broker/Broker.ts):

- `connect()`
- `disconnect()`
- `getBalance()`
- `getSpread(symbol)`
- `getCandles(symbol, timeframe, limit)`
- `placeOrder(order)`
- `modifyOrder(orderId, updates)`
- `closeOrder(orderId)`
- `getOpenPositions()`
- `getTradeHistory()`

Keep real broker code behind this interface so strategy, risk, and backtesting remain testable.

## Warning

This is educational software, not financial advice. Backtest and forward-test on demo before using any strategy with real money.
