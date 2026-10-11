# WILLAM — JEV decision engine

> **J**udgment → **E**valuation → **V**alidation
> Rules guard the floor. The LLM handles judgment. Rules catch the fall.

## Three phases

### 1 · JUDGMENT — hard gates (rules only, the LLM cannot override)
- duplicate position in symbol → **block**
- max total exposure reached → **block**
- SL/TP invalid or risk ≤ 0 → **block**
- R:R below 80% of the configured minimum → **block**

### 2 · EVALUATION — deterministic score
Starts at 0.50, then:
| Factor | Δ |
|---|---|
| strategy confidence | ±(conf−0.5)×0.4 |
| correlated exposure | −0.10 |
| London/NY overlap (12–16 UTC) | +0.05 |
| Asian session (00–07 UTC) | −0.05 |
| spread normal / blown out | +0.05 / −0.15 |
| news surprise / expected news | +0.15 / −0.10 |

### 3 · VALIDATION — the LLM verdict
Sends signal + portfolio + score + market context to the model, asks for strict JSON:

```json
{"approved": true, "confidence": 0.82, "reasoning": "one short sentence"}
```

Final rule:
```
approved = llm.approved AND deterministic_score ≥ 0.7
confidence = min(llm.confidence, deterministic_score)
```

**If no provider answers → the deterministic score alone decides. Trading never halts.**

## Provider chain

| Order | Provider | Key | Endpoint |
|---|---|---|---|
| 1 | nemotron (NVIDIA NIM) | `NVIDIA_API_KEY` | `integrate.api.nvidia.com/v1` |
| 2 | free-llm-api | `FREELLMAPI_KEY` | `api.freellmapi.com/v1` |
| 3 | rules | — | always available |

Health handling: 3 consecutive failures bench a provider for `llm_cooldown_minutes`.
Rate cap: 120 calls/hour per provider.

## Audit trail

Decisions persist every loop cycle:
```
jev/decisions/decisions_YYYYMMDD.json
```

Each entry: timestamp · symbol · direction · strategy · approved · confidence · source · reasoning.

## Verified behaviour

- duplicate symbol → blocked with `judgment_failed` ✓
- R:R too low → blocked ✓
- invalid SL/TP → blocked ✓
- no API keys → decision still produced, `source=rules` ✓
- correlation detected: `['GBPUSD(USD)']` ✓
