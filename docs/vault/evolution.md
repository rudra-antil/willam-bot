# WILLAM — Evolution: the psychology loop

> *"Learn more from losses than wins. Get frustrated. Find the setup. Solve it.*
> *Never lose to the same setup twice. Curiosity grows. Survive."*

## The loop

```
TRADE CLOSES
      │
      ├─ WIN  → weight 1×  → relief: frustration −0.2, streak reset
      └─ LOSS → weight 3×  → frustration += 0.15 × streak
                    │
                    ▼
        FINGERPRINT the setup
        symbol · strategy · direction · session · regime · volatility · spread
                    │
                    ▼
        KNOWN PATTERN?
          no  → create pattern + ask 3 root-cause questions
          yes → frustration climbs
                    │
                    ▼
        frustration ≥ 3 consecutive losses
                    │
                    ▼
        🧠 DEEP DIVE
        analyse every matching trade → generate targeted fixes
        (wider SL · volatility filter · spread filter · session filter)
                    │
                    ▼
        🧪 SOLUTIONS enter the exploration queue
                    │
                    ▼
        WIN while testing → effectiveness += 0.3
        effectiveness > 0.7 → ✅ PATTERN SOLVED
                    │
                    ▼
        💀 SURVIVAL: 50 trades & expectancy ≤ 0 → strategy CULLED
```

## Tuning

| Parameter | Default | Effect |
|---|---|---|
| `loss_learning_multiplier` | 3.0 | How much harder losses teach |
| `frustration_threshold` | 3 | Consecutive losses before deep dive |
| `curiosity_growth_rate` | 0.05 | Curiosity added per trade |
| `max_frustration` | 1.0 | Cap |
| `solution_test_trades` | 20 | Validation window |
| `profit_per_trade_target` | 0.001 | Per-trade expectancy goal |

## State files (never committed — per installation)

| File | Contents |
|---|---|
| `evolution/memory.json` | every trade with its fingerprint + weight |
| `evolution/losing_patterns.json` | patterns, frustration, solutions, status |
| `evolution/performance.json` | per strategy × symbol metrics |
| `evolution/gaps.json` | identified weaknesses |
| `evolution/daily_learning.json` | per-day lessons, score, frustration peaks |
| `evolution/curiosity.json` | open questions + hypotheses |

## Verified behaviour

From the test suite (`tests/test_willam.py`):
- loss weighted **3.0×** ✓
- 4 losses → frustration **1.00** ✓
- deep dive fired → **2 solutions** generated ✓
- **15** curiosity questions raised ✓
- a win → frustration **1.00 → 0.80**, streak reset ✓
- memory persisted to disk ✓
