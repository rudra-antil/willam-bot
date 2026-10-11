"""
JEV Engine Module for WILLAM — v2
Judgment, Evaluation, Validation — now powered by a REAL LLM.

Primary: NVIDIA NIM (nemotron) via OpenAI-compatible API.
Fallback: FreeLLMAPI (https://freeLLMAPI.com) — same OpenAI shape.
If both are unreachable, degrades to the deterministic rule engine
so the bot never stops trading because a model is down.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("willam.jev")

# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

PROVIDERS: Dict[str, Dict[str, Any]] = {
    "nemotron": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "model": "nvidia/llama-3.1-nemotron-70b-instruct",
        "api_key_env": "NVIDIA_API_KEY",
        "timeout": 20.0,
        "max_calls_per_hour": 120,  # self-imposed safety cap
    },
    "free-llm-api": {
        "base_url": "https://api.freellmapi.com/v1",
        "model": "gpt-4o-mini",
        "api_key_env": "FREELLMAPI_KEY",
        "timeout": 20.0,
        "max_calls_per_hour": 120,
    },
}


@dataclass
class JEVDecision:
    approved: bool
    confidence: float
    reasoning: str
    source: str = "rules"  # "nemotron" | "free-llm-api" | "rules"
    adjustments: Dict = field(default_factory=dict)
    conditions: List[str] = field(default_factory=list)


@dataclass
class GateResult:
    passed: bool
    confidence: float


@dataclass
class EvalResult:
    passed: bool
    confidence: float


class JEVEngine:
    """
    Three-phase decision engine:

    Phase 1 JUDGMENT   — hard risk/portfolio gates. Never skipped, never LLM.
    Phase 2 EVALUATION — deterministic score components (spread, session, correlation...).
    Phase 3 VALIDATION — LLM reviews the whole context and returns the final
                         verdict + confidence + reasoning. Falls back to the
                         deterministic score if the LLM is unavailable.

    Why LLM at phase 3 only: rules catch violations the model might rationalize
    away; the model catches soft-context judgment rules can't express.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.jev_config = config.get("jev", {})
        self.confidence_threshold = self.jev_config.get("confidence_threshold", 0.7)
        self.max_reasoning_steps = self.jev_config.get("max_reasoning_steps", 5)

        # Provider chain: primary -> fallback -> rules
        self.primary = self.jev_config.get("model", "nemotron")
        self.fallback = self.jev_config.get("fallback_model", "free-llm-api")
        self.providers = [p for p in (self.primary, self.fallback) if p in PROVIDERS]

        # Rate limiting + health tracking per provider
        self.call_times: Dict[str, List[float]] = {p: [] for p in PROVIDERS}
        self.fail_counts: Dict[str, int] = {p: 0 for p in PROVIDERS}
        self.cooldown_until: Dict[str, float] = {p: 0.0 for p in PROVIDERS}
        self.cooldown_minutes = self.jev_config.get("llm_cooldown_minutes", 10)

        self.decision_history: List[Dict] = []
        self.log_dir = os.path.join("jev", "decisions")
        os.makedirs(self.log_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def validate(self, signal: Any, positions: Dict) -> bool:
        """Quick validation for main loop — returns True/False."""
        decision = await self.make_decision(signal, positions)
        return decision.approved

    async def make_decision(self, signal: Any, positions: Dict, market_context: Dict = None) -> JEVDecision:
        reasoning_steps: List[str] = []

        # ===== PHASE 1: JUDGMENT — hard gates (never LLM) =====
        gate = self._judge_signal(signal, positions, reasoning_steps)
        if not gate.passed:
            return self._record(
                JEVDecision(approved=False, confidence=gate.confidence,
                            reasoning="; ".join(reasoning_steps),
                            conditions=["judgment_failed"], source="rules"),
                signal)

        # ===== PHASE 2: EVALUATION — deterministic score =====
        evaluation = self._evaluate_context(signal, positions, reasoning_steps)
        if not evaluation.passed:
            return self._record(
                JEVDecision(approved=False, confidence=evaluation.confidence,
                            reasoning="; ".join(reasoning_steps),
                            conditions=["evaluation_failed"], source="rules"),
                signal)

        # ===== PHASE 3: VALIDATION — LLM verdict with fallback =====
        context = self._build_llm_context(signal, positions, evaluation, market_context)
        llm_verdict = await self._llm_validate(context, reasoning_steps)

        if llm_verdict is not None:
            approved = llm_verdict["approved"] and evaluation.confidence >= self.confidence_threshold
            decision = JEVDecision(
                approved=approved,
                confidence=min(llm_verdict["confidence"], evaluation.confidence),
                reasoning=llm_verdict.get("reasoning", "") or "; ".join(reasoning_steps),
                source=llm_verdict.get("source", "nemotron"),
                conditions=[],
            )
        else:
            # Deterministic fallback — bot keeps trading without LLM
            approved = evaluation.confidence >= self.confidence_threshold
            decision = JEVDecision(
                approved=approved,
                confidence=evaluation.confidence,
                reasoning="; ".join(reasoning_steps) + "; llm-unavailable-deterministic-fallback",
                source="rules",
                conditions=["llm_unavailable"],
            )

        return self._record(decision, signal)

    # ------------------------------------------------------------------
    # Phase 1 — hard gates
    # ------------------------------------------------------------------

    def _judge_signal(self, signal: Any, positions: Dict, reasoning: List[str]) -> GateResult:
        """Hard portfolio gates — no LLM can override these."""
        # Existing position in same symbol
        same_symbol = [p for p in positions.values() if p.symbol == signal.symbol]
        if same_symbol:
            reasoning.append(f"BLOCK: {len(same_symbol)} position(s) already open in {signal.symbol}")
            return GateResult(False, 0.3)

        # Total exposure
        max_positions = self.config.get("risk", {}).get("max_total_positions", 5)
        if len(positions) >= max_positions:
            reasoning.append(f"BLOCK: max positions reached {len(positions)}/{max_positions}")
            return self._GateResult(False, 0.3)

        # SL/TP sanity
        if signal.sl <= 0 or signal.tp <= 0:
            reasoning.append("BLOCK: invalid SL/TP")
            return GateResult(False, 0.2)

        # Risk/reward
        if signal.direction == "buy":
            risk = signal.entry_price - signal.sl
            reward = signal.tp - signal.entry_price
        else:
            risk = signal.sl - signal.entry_price
            reward = signal.entry_price - signal.tp
        if risk <= 0:
            reasoning.append("BLOCK: non-positive risk distance")
            return GateResult(False, 0.2)

        min_rr = self.config.get("risk", {}).get("risk_reward_ratio", 2.0) * 0.8
        if reward / risk < min_rr:
            reasoning.append(f"BLOCK: R:R {reward / risk:.2f} below minimum {min_rr:.2f}")
            return GateResult(False, 0.3)

        reasoning.append(f"Gates passed: no dup position, exposure ok, R:R {reward / risk:.2f}")
        return GateResult(True, 0.5)

    # ------------------------------------------------------------------
    # Phase 2 — deterministic evaluation
    # ------------------------------------------------------------------

    def _evaluate_context(self, signal: Any, positions: Dict, reasoning: List[str]) -> EvalResult:
        confidence = 0.5
        # Strategy confidence
        if hasattr(signal, "confidence") and signal.confidence:
            confidence += (signal.confidence - 0.5) * 0.4
            reasoning.append(f"Strategy confidence {signal.confidence:.2f}")

        # Correlation penalty
        correlated = self._check_correlation(signal.symbol, positions)
        if correlated:
            confidence -= 0.10
            reasoning.append(f"Correlated exposure: {correlated}")

        # Session quality
        hour = datetime.now(timezone.utc).hour
        if 12 <= hour <= 16:
            confidence += 0.05
            reasoning.append("London/NY overlap — high liquidity")
        elif 0 <= hour <= 7:
            confidence -= 0.05
            reasoning.append("Asian session — thin liquidity")

        # Spread check via MT5
        try:
            import MetaTrader5 as mt5
            info = mt5.symbol_info(signal.symbol)
            tick = mt5.symbol_info_tick(signal.symbol)
            if info and tick and info.point:
                spread = (tick.ask - tick.bid) / info.point
                if info.spread and spread > info.spread * 2:
                    confidence -= 0.15
                    reasoning.append(f"Wide spread {spread:.1f} vs avg {info.spread:.1f}")
                else:
                    confidence += 0.05
                    reasoning.append(f"Spread normal ({spread:.1f})")
        except Exception:
            pass

        # News context
        if getattr(signal, "news_context", None):
            reason = signal.news_context.get("reason", "")
            if "surprise" in reason:
                confidence += 0.15
                reasoning.append(f"News surprise: {reason}")
            elif "expected" in reason:
                confidence -= 0.10
                reasoning.append("Expected news — follow trend only after confirmation")

        reasoning.append(f"Deterministic score: {confidence:.2f}")
        return EvalResult(True, max(0.0, min(confidence, 1.0)))

    # ------------------------------------------------------------------
    # Phase 3 — LLM validation
    # ------------------------------------------------------------------

    def _build_llm_context(self, signal: Any, positions: Dict, evaluation: EvalResult, market_context: Optional[Dict] = None) -> Dict:
        return {
            "signal": {
                "symbol": signal.symbol,
                "direction": signal.direction,
                "entry": signal.entry_price,
                "sl": signal.sl,
                "tp": signal.tp,
                "lots": signal.lot_size,
                "strategy": signal.strategy_name,
                "strategy_confidence": signal.confidence,
                "reasoning": signal.reasoning,
                "news_context": getattr(signal, "news_context", None),
            },
            "portfolio": {
                "open_positions": [
                    {"symbol": p.symbol, "direction": p.direction, "pnl": p.current_pnl}
                    for p in positions.values()
                ],
                "count": len(positions),
            },
            "deterministic_score": evaluation.confidence,
            "evaluation_reasoning": "",
            "market_context": market_context or {},
            "threshold": self.confidence_threshold,
        }

    def _system_prompt(self) -> str:
        return (
            "You are JEV, the risk-gatekeeper of a forex/CFD trading bot. "
            "You receive a trade candidate that already passed hard risk gates and a deterministic score. "
            "Your job: catch what rules cannot — regime mismatches, contradictory context, suspicious setups. "
            'Respond ONLY with JSON: {"approved": bool, "confidence": 0.0-1.0, "reasoning": "one short sentence"}'
        )

    def _user_prompt(self, context: Dict) -> str:
        return (
            "Decide whether to APPROVE this trade.\n"
            f"Deterministic score: {context['deterministic_score']:.2f} (approval threshold {context['threshold']})\n"
            f"Signal: {json.dumps(context['signal'])}\n"
            f"Portfolio: {json.dumps(context['portfolio'])}\n"
            f"Market context: {json.dumps(context.get('market_context', {}))}\n"
            "Be conservative: when uncertain, reject."
        )

    async def _llm_validate(self, context: Dict, reasoning_steps: List[str]) -> Optional[Dict]:
        """Try providers in order; return verdict dict or None if all fail."""
        for provider in self.providers:
            if not self._provider_available(provider):
                continue
            try:
                verdict = await self._call_provider(provider, context)
                if verdict is not None:
                    self.fail_counts[provider] = 0
                    reasoning_steps.append(f"LLM[{provider}] approved={verdict['approved']} conf={verdict['confidence']:.2f}")
                    return verdict
            except Exception as e:
                self._register_failure(provider, e)
        return None

    def _provider_available(self, provider: str) -> bool:
        now = time.monotonic()
        if now < self.cooldown_until.get(provider, 0):
            return False
        key = os.environ.get(PROVIDERS[provider]["api_key_env"], "")
        if not key:
            return False
        # hourly rate cap
        calls = [t for t in self.call_times[provider] if now - t < 3600]
        self.call_times[provider] = calls
        return len(calls) < PROVIDERS[provider]["max_calls_per_hour"]

    def _register_failure(self, provider: str, error: Exception):
        self.fail_counts[provider] += 1
        if self.fail_counts[provider] >= 3:
            self.cooldown_until[provider] = time.monotonic() + self.cooldown_minutes * 60
            self.fail_counts[provider] = 0
            logger.warning(f"JEV LLM '{provider}' in cooldown {self.cooldown_minutes}min after repeated failures: {error}")

    async def _call_provider(self, provider: str, context: Dict) -> Optional[Dict]:
        cfg = PROVIDERS[provider]
        api_key = os.environ[cfg["api_key_env"]]
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "model": cfg["model"],
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": self._user_prompt(context)},
            ],
            "temperature": 0.2,
            "max_tokens": 150,
        }
        self.call_times[provider].append(time.monotonic())

        async with httpx.AsyncClient(timeout=cfg["timeout"], verify=False) as client:
            resp = await client.post(f"{cfg['base_url']}/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        content = data["choices"][0]["message"]["content"]
        return self._parse_verdict(content, provider)

    def _parse_verdict(self, content: str, provider: str) -> Optional[Dict]:
        """Tolerant JSON extraction — models wrap JSON in prose/markdown."""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            start, end = content.find("{"), content.rfind("}")
            if start >= 0 and end > start:
                try:
                    parsed = json.loads(content[start:end + 1])
                except json.JSONDecodeError:
                    return None
            else:
                return None
        if not isinstance(parsed, dict) or "approved" not in parsed:
            return None
        try:
            confidence = float(parsed.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        return {
            "approved": bool(parsed.get("approved")),
            "confidence": max(0.0, min(confidence, 1.0)),
            "reasoning": str(parsed.get("reasoning", ""))[:300],
            "source": provider,
        }

    # ------------------------------------------------------------------
    # Persistence + stats
    # ------------------------------------------------------------------

    def _record(self, decision: JEVDecision, signal: Any) -> JEVDecision:
        self.decision_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": signal.symbol,
            "direction": signal.direction,
            "strategy": signal.strategy_name,
            "approved": decision.approved,
            "confidence": decision.confidence,
            "source": decision.source,
            "reasoning": decision.reasoning,
        })
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]
        return decision

    def save_decisions(self) -> None:
        """Persist decisions — called from main loop every cycle."""
        date = datetime.now(timezone.utc).strftime("%Y%m%d")
        path = os.path.join(self.log_dir, f"decisions_{date}.json")
        try:
            existing = []
            if os.path.exists(path):
                with open(path) as f:
                    existing = json.load(f)
            with open(path, "w") as f:
                json.dump(existing + self.decision_history, f, indent=2)
            self.decision_history.clear()
        except Exception as e:
            logger.error(f"Failed to save JEV decisions: {e}")

    def _check_correlation(self, symbol: str, positions: Dict) -> List[str]:
        """Flag correlated exposure across currency/commodity/index groups."""
        corr_groups = {
            "USD": ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"],
            "JPY": ["USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY"],
            "EUR": ["EURUSD", "EURJPY", "EURGBP", "EURAUD", "EURNZD", "EURCAD", "EURCHF"],
            "GBP": ["GBPUSD", "GBPJPY", "EURGBP", "GBPAUD", "GBPNZD", "GBPCAD", "GBPCHF"],
            "COMMODITY": ["XAUUSD", "XAGUSD", "USOIL", "UKOIL"],
            "INDICES": ["US30", "US500", "USTEC", "GER40", "UK100", "JPN225"],
        }
        symbol_groups = [g for g, symbols in corr_groups.items() if symbol in symbols]
        if not symbol_groups:
            return []
        correlated = []
        for pos in positions.values():
            for group in symbol_groups:
                if pos.symbol in corr_groups.get(group, []):
                    correlated.append(f"{pos.symbol}({group})")
        return correlated

    def get_decision_stats(self) -> Dict:
        if not self.decision_history:
            return {}
        total = len(self.decision_history)
        approved = sum(1 for d in self.decision_history if d["approved"])
        return {
            "total_decisions": total,
            "approved": approved,
            "rejected": total - approved,
            "approval_rate": approved / total * 100,
            "avg_confidence": sum(d["confidence"] for d in self.decision_history) / total,
        }
