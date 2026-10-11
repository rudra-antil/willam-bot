# WILLAM — Config Reference

> Source of truth: `config/settings.yaml`

## mt5 — connection
| Key | Meaning |
|---|---|
| `login` / `password` / `server` | Broker account (REQUIRED) |
| `path` | MT5 terminal.exe path — blank = auto-detect |

## trading
| Key | Meaning |
|---|---|
| `symbols` | Instruments traded (each gets its own best strategy) |
| `timeframe` | Signal timeframe (M15 default) |
| `session_hours` | Session windows (UTC) |

## news
| Key | Meaning |
|---|---|
| `lookahead_hours` | How far ahead to gate new entries (1h per spec) |
| `pause_before_minutes` | Stop NEW entries this long before release |
| `wait_after_minutes` | Cool-down after release |
| `confirmation_wait_minutes` | How long to wait for trend confirmation on expected-news releases |

**v2 behaviour:** gates are **per symbol**. EURUSD news no longer stops GBPUSD trading.

## risk
| Key | Meaning |
|---|---|
| `default_risk_percent` | Risk per trade (1% default) |
| `max_risk_percent` | Absolute ceiling (2%) |
| `max_daily_drawdown` | Sticky halt for the UTC day (5%) |
| `max_positions_per_symbol` | 1 (no stacking) |
| `max_total_positions` | Portfolio cap (5) |
| `risk_reward_ratio` | Minimum R:R (2.0) |
| `atr_multiplier_sl` / `_tp` | Volatility-adaptive stops (1.5 / 3.0) |

## strategy
| Key | Meaning |
|---|---|
| `min_winrate` | Strategies below this are never deployed (55%) |
| `min_trades_for_validation` | Minimum sample to trust a backtest (100) |
| `lookback_days` | Backtest history (365) |

## evolution
| Key | Meaning |
|---|---|
| `loss_learning_multiplier` | **Losses taught this many times harder (3.0)** |
| `frustration_threshold` | Consecutive losses before a deep dive (3) |
| `curiosity_growth_rate` | Curiosity added per trade (0.05) |
| `survival_threshold` | Expectancy floor to stay alive (0.5) |

## jev
| Key | Meaning |
|---|---|
| `model` | Primary LLM provider key (`nemotron`) |
| `fallback_model` | Fallback provider (`free-llm-api`) |
| `confidence_threshold` | Minimum score to approve (0.7) |
| `llm_cooldown_minutes` | Bench a failing provider this long (10) |

## Environment (`.env`)
```bash
NVIDIA_API_KEY=      # nemotron
FREELLMAPI_KEY=      # fallback
```
No keys ⇒ rules-only mode. Trading continues normally.
