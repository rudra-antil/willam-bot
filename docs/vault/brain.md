# WILLAM — Central Brain (v2)

> Forex/CFD trading bot · news-aware · LLM-decided · self-evolving
> Repo: https://github.com/rudra-antil/willam-bot

## Quick links
- [[config]] — every setting, what it does
- [[evolution]] — the psychology loop (loss-weighted learning)
- [[jev-decisions]] — how the LLM gatekeeper decides
- [[risk]] — sizing, stops, hard limits
- [[strategies/index]] — the 7 strategies + selection

## The loop

```
every 30s
 ├─ 1. manage open positions → on close: log + LEARN
 ├─ 2. refresh news calendar (15 min)
 ├─ 3. tick per-symbol news windows
 ├─ 4. strategies for every symbol NOT gated by its own news
 ├─ 5. JEV decides (LLM, or rules if LLM is down)
 ├─ 6. risk manager sizes + sets SL/TP
 ├─ 7. execute via MT5
 └─ 8. evolution: frustration → deep dive → solutions
```

## Non-negotiables (v2)

1. **News never halts the whole bot.** A pair with imminent news is gated by itself; all other pairs keep trading.
2. **Every entry has SL/TP.** Default 1.5×ATR stop, 2:1 reward.
3. **Hard gates beat the LLM.** Duplicate symbol, bad R:R, max exposure → blocked before the model is consulted.
4. **The LLM is optional.** No API key ⇒ deterministic rules. Trading never stops because a model is down.
5. **Losses teach 3×.** Every loss is fingerprinted, tracked, and solved.
6. **Unfit strategies die.** 50 trades with expectancy ≤ 0 → culled.

## Status

| Item | Value |
|---|---|
| Version | 2.0.0 |
| Tests | `python tests/test_willam.py` → 65 checks |
| Symbols | EURUSD GBPUSD USDJPY AUDUSD USDCAD NZDUSD XAUUSD US30 |
| Timeframe | M15 |
| Risk/trade | 1% (max 2%) |
| Rewards | 2:1 |
| LLM chain | nemotron → free-llm-api → rules |

## Getting started

1. Log into MT5 terminal (keep it running).
2. `pip install -r requirements.txt`
3. Fill `mt5.login/password/server` in `config/settings.yaml`
4. (optional) `.env` with `NVIDIA_API_KEY` / `FREELLMAPI_KEY`
5. `python tests/test_willam.py` → expect 65 passed
6. `python main.py`
