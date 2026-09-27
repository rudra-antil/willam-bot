# WILLAM - Forex/CFD Trading Bot

> **Fresh implementation - No legacy dependencies**

A fully autonomous forex/CFD trading bot with news-aware strategy execution, risk management, and continuous evolution.

## Features

### 🎯 News-Aware Trading
- **1-hour lookahead** for high-impact economic events
- **Pre-news pause** (60 min) - stops trading before major releases
- **Surprise analysis** - compares actual vs forecast vs previous
- **Expected news** → waits for trend confirmation before entering
- **Surprising news** → trades data direction immediately (bullish/bearish)

### 🧠 Strategy Engine
- **7 built-in strategies**: Trend Follow, Mean Reversion, Breakout, RSI Divergence, MACD, Bollinger Bands, Ichimoku
- **Automatic backtesting** on 1-year history per symbol
- **Winrate filtering** - only deploys strategies >55% winrate
- **Per-symbol optimization** - each pair gets its best strategy

### ⚖️ Risk Management
- **ATR-based stops** - volatility-adjusted SL/TP
- **Fixed risk per trade** (default 1%, max 2%)
- **Risk:Reward ratio** enforcement (default 2:1)
- **Daily drawdown limit** (5%)
- **Position limits** - max per symbol & total
- **Correlation awareness** - avoids overexposure

### 🤖 JEV Decision Engine
- **Judgment** - signal quality analysis
- **Evaluation** - portfolio/risk context check
- **Validation** - market conditions (spread, liquidity, session)
- **Confidence threshold** - only executes high-conviction trades

### 🧬 Evolution Engine
- **Learns from every trade** - win or loss
- **Gap analysis** - identifies strategy weaknesses by session, condition
- **Culls weak strategies** - kills negative expectancy after 50 trades
- **Continuous adaptation** - fills gaps with new variants
- **Survival mandate** - "evolve or die" philosophy

## Architecture

```
willam-bot/
├── main.py                 # Main orchestrator
├── config/
│   └── settings.yaml       # All configuration
├── news/
│   └── checker.py          # Economic calendar & surprise detection
├── strategies/
│   └── manager.py          # Backtesting, selection, signals
├── risk/
│   └── manager.py          # Position sizing, SL/TP, limits
├── jev/
│   └── engine.py           # Judgment/Evaluation/Validation
├── evolution/
│   └── engine.py           # Learning, gap analysis, culling
├── logging/
│   └── trade_logger.py     # CSV/JSON per symbol/strategy
├── backtest/               # Backtest results (auto-generated)
├── logs/                   # Trade logs (auto-generated)
└── evolution/              # Evolution memory (auto-generated)
```

## Quick Start

### 1. Prerequisites
- Windows with MetaTrader 5 terminal installed
- MT5 logged into your broker account
- Python 3.11+

### 2. Install
```bash
cd willam-bot
pip install -r requirements.txt
```

### 3. Configure
Edit `config/settings.yaml`:
```yaml
mt5:
  login: YOUR_ACCOUNT_NUMBER
  password: "YOUR_PASSWORD"
  server: "YOUR_BROKER_SERVER"

trading:
  symbols:
    - "EURUSD"
    - "GBPUSD"
    - "USDJPY"
    # ... add your pairs
```

### 4. Run
```bash
python main.py
```

## Configuration

Key settings in `config/settings.yaml`:

| Section | Key | Description |
|---------|-----|-------------|
| `news` | `lookahead_hours` | Hours ahead to check for news (default: 1) |
| `news` | `pause_before_minutes` | Minutes to pause before news (default: 60) |
| `risk` | `default_risk_percent` | Risk per trade % (default: 1.0) |
| `risk` | `risk_reward_ratio` | Minimum R:R (default: 2.0) |
| `strategy` | `min_winrate` | Minimum winrate to deploy (default: 55) |
| `evolution` | `enabled` | Enable learning (default: true) |

## Logging & Analysis

- **Live trades**: `logs/trades/{SYMBOL}_{STRATEGY}_closed.csv`
- **Backtests**: `backtest/backtest_{SYMBOL}_{STRATEGY}_{TIMEFRAME}_{TIMESTAMP}.json`
- **Evolution**: `evolution/memory.json`, `evolution/performance.json`, `evolution/gaps.json`

## Strategy Details

| Strategy | Type | Best For |
|----------|------|----------|
| `trend_follow` | EMA crossover | Trending markets |
| `mean_reversion` | Bollinger Bands | Ranging markets |
| `breakout` | Donchian channels | Volatility expansion |
| `rsi_divergence` | RSI levels | Reversal points |
| `macd_trend` | MACD crossover | Momentum trends |
| `bollinger_bands` | BB + trend filter | Trend + reversion |
| `ichimoku` | Ichimoku Cloud | Complete system |

## Evolution Philosophy

> **"It wants to survive with 100% will and generate money continuously"**

The evolution engine:
1. Records every trade with full context
2. Tracks performance per strategy per symbol
3. Identifies gaps (low winrate sessions, high drawdown, negative expectancy)
4. Culls strategies that fail survival threshold (>50% expectancy)
5. Adapts by filling gaps with new variants
6. Never stops learning - each trade makes it smarter

## Disclaimer

**Trading forex/CFD carries high risk. This bot trades real money. Use at your own risk. Test thoroughly in demo first.**

## License

MIT - Use freely, modify, evolve.