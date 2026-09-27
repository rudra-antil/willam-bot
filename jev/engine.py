"""
JEV Engine Module for WILLAM
Judgment, Evaluation, Validation - makes final trading decisions
"""

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class JEVDecision:
    approved: bool
    confidence: float
    reasoning: str
    adjustments: Dict = None
    conditions: List[str] = None


class JEVEngine:
    """
    JEV (Judgment, Evaluation, Validation) Engine
    
    Makes final trading decisions by:
    1. Judgment: Analyzing signal quality and context
    2. Evaluation: Checking against risk rules and portfolio state
    3. Validation: Confirming with market conditions
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.jev_config = config.get("jev", {})
        self.confidence_threshold = self.jev_config.get("confidence_threshold", 0.7)
        self.max_reasoning_steps = self.jev_config.get("max_reasoning_steps", 5)
        
        # Decision history for learning
        self.decision_history: List[Dict] = []
    
    async def validate(self, signal: Any, positions: Dict) -> bool:
        """Quick validation for main loop - returns True/False."""
        decision = await self.make_decision(signal, positions)
        return decision.approved
    
    async def make_decision(self, signal: Any, positions: Dict, market_context: Dict = None) -> JEVDecision:
        """
        Full JEV decision process.
        Returns detailed decision with reasoning.
        """
        reasoning_steps = []
        confidence = 0.5  # Base confidence
        
        # ===== JUDGMENT PHASE =====
        # Analyze signal quality
        judgment = await self._judge_signal(signal, reasoning_steps)
        confidence += judgment.confidence_delta
        
        # ===== EVALUATION PHASE =====
        # Check risk, portfolio, correlations
        evaluation = await self._evaluate_context(signal, positions, reasoning_steps)
        confidence += evaluation.confidence_delta
        
        if not evaluation.passed:
            return JEVDecision(
                approved=False,
                confidence=confidence,
                reasoning="; ".join(reasoning_steps),
                conditions=["evaluation_failed"]
            )
        
        # ===== VALIDATION PHASE =====
        # Validate against current market conditions
        validation = await self._validate_market(signal, market_context, reasoning_steps)
        confidence += validation.confidence_delta
        
        if not validation.passed:
            return JEVDecision(
                approved=False,
                confidence=confidence,
                reasoning="; ".join(reasoning_steps),
                conditions=["validation_failed"]
            )
        
        # Final decision
        approved = confidence >= self.confidence_threshold
        
        # Record decision for learning
        self.decision_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "signal": {
                "symbol": signal.symbol,
                "direction": signal.direction,
                "strategy": signal.strategy_name,
                "confidence": signal.confidence
            },
            "decision": {
                "approved": approved,
                "final_confidence": confidence,
                "reasoning": reasoning_steps
            }
        })
        
        # Keep last 1000 decisions
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]
        
        return JEVDecision(
            approved=approved,
            confidence=min(confidence, 1.0),
            reasoning="; ".join(reasoning_steps),
            adjustments={},
            conditions=[]
        )
    
    async def _judge_signal(self, signal: Any, reasoning: List[str]) -> Any:
        """Judge the signal quality."""
        @dataclass
        class JudgmentResult:
            confidence_delta: float
        
        delta = 0.0
        
        # Signal confidence from strategy
        if hasattr(signal, 'confidence'):
            delta += (signal.confidence - 0.5) * 0.5
            reasoning.append(f"Strategy confidence: {signal.confidence:.2f}")
        
        # Strategy track record
        if hasattr(signal, 'strategy_name'):
            reasoning.append(f"Strategy: {signal.strategy_name}")
        
        # News context bonus/penalty
        if hasattr(signal, 'news_context') and signal.news_context:
            reason = signal.news_context.get('reason', '')
            if 'surprise' in reason:
                delta += 0.15
                reasoning.append(f"News surprise factor: {reason}")
            elif 'expected' in reason:
                delta -= 0.1  # Expected news less reliable
                reasoning.append("Expected news - lower confidence")
        
        # Time of day filter
        hour = datetime.utcnow().hour
        if 8 <= hour <= 17:  # London/NY overlap
            delta += 0.05
            reasoning.append("Active session hours")
        elif 0 <= hour <= 7:  # Asian session
            delta -= 0.05
            reasoning.append("Asian session - lower volatility")
        
        return JudgmentResult(confidence_delta=delta)
    
    async def _evaluate_context(self, signal: Any, positions: Dict, reasoning: List[str]) -> Any:
        """Evaluate portfolio and risk context."""
        @dataclass
        class EvaluationResult:
            passed: bool
            confidence_delta: float
        
        delta = 0.0
        
        # Check existing positions in same symbol
        same_symbol = [p for p in positions.values() if p.symbol == signal.symbol]
        if same_symbol:
            reasoning.append(f"Already have {len(same_symbol)} position(s) in {signal.symbol}")
            return EvaluationResult(passed=False, confidence_delta=-0.3)
        
        # Check correlation with existing positions
        correlated = self._check_correlation(signal.symbol, positions)
        if correlated:
            reasoning.append(f"Correlated positions exist: {correlated}")
            delta -= 0.1
        
        # Check total exposure
        total_positions = len(positions)
        max_positions = self.config.get("risk", {}).get("max_total_positions", 5)
        if total_positions >= max_positions:
            reasoning.append(f"Max positions reached: {total_positions}/{max_positions}")
            return EvaluationResult(passed=False, confidence_delta=-0.3)
        
        # Check daily drawdown
        # (Risk manager handles this, but double-check)
        reasoning.append(f"Portfolio check passed: {total_positions}/{max_positions} positions")
        delta += 0.05
        
        return EvaluationResult(passed=True, confidence_delta=delta)
    
    async def _validate_market(self, signal: Any, market_context: Dict, reasoning: List[str]) -> Any:
        """Validate against current market conditions."""
        @dataclass
        class ValidationResult:
            passed: bool
            confidence_delta: float
        
        delta = 0.0
        
        # Spread check
        try:
            import MetaTrader5 as mt5
            info = mt5.symbol_info(signal.symbol)
            tick = mt5.symbol_info_tick(signal.symbol)
            
            if info and tick:
                spread = (tick.ask - tick.bid) / info.point
                avg_spread = info.spread  # Typical spread
                
                if spread > avg_spread * 2:
                    reasoning.append(f"Wide spread detected: {spread:.1f} vs avg {avg_spread:.1f}")
                    delta -= 0.15
                else:
                    reasoning.append(f"Spread normal: {spread:.1f}")
                    delta += 0.05
        except:
            pass
        
        # Volatility check (ATR)
        # High volatility = wider stops needed = lower position size
        # Could adjust confidence based on volatility regime
        
        # Session liquidity
        hour = datetime.utcnow().hour
        if 12 <= hour <= 16:  # London/NY overlap - best liquidity
            delta += 0.05
            reasoning.append("High liquidity session")
        
        return ValidationResult(passed=True, confidence_delta=delta)
    
    def _check_correlation(self, symbol: str, positions: Dict) -> List[str]:
        """Check for correlated positions."""
        # Major correlation groups
        corr_groups = {
            "USD": ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"],
            "JPY": ["USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY"],
            "EUR": ["EURUSD", "EURJPY", "EURGBP", "EURAUD", "EURNZD", "EURCAD", "EURCHF"],
            "GBP": ["GBPUSD", "GBPJPY", "EURGBP", "GBPAUD", "GBPNZD", "GBPCAD", "GBPCHF"],
            "COMMODITY": ["XAUUSD", "XAGUSD", "USOIL", "UKOIL"],
            "INDICES": ["US30", "US500", "USTEC", "GER40", "UK100", "JPN225"],
        }
        
        # Find which group this symbol belongs to
        symbol_groups = []
        for group, symbols in corr_groups.items():
            if symbol in symbols:
                symbol_groups.append(group)
        
        if not symbol_groups:
            return []
        
        # Check existing positions for same group
        correlated = []
        for pos in positions.values():
            for group in symbol_groups:
                if pos.symbol in corr_groups.get(group, []):
                    correlated.append(f"{pos.symbol}({group})")
        
        return correlated
    
    def get_decision_stats(self) -> Dict:
        """Get statistics on JEV decisions."""
        if not self.decision_history:
            return {}
        
        total = len(self.decision_history)
        approved = sum(1 for d in self.decision_history if d["decision"]["approved"])
        avg_confidence = sum(d["decision"]["final_confidence"] for d in self.decision_history) / total
        
        return {
            "total_decisions": total,
            "approved": approved,
            "rejected": total - approved,
            "approval_rate": approved / total * 100,
            "avg_confidence": avg_confidence
        }
    
    def save_decisions(self, filepath: str):
        """Save decision history to file."""
        with open(filepath, "w") as f:
            json.dump(self.decision_history, f, indent=2)
    
    def load_decisions(self, filepath: str):
        """Load decision history from file."""
        try:
            with open(filepath) as f:
                self.decision_history = json.load(f)
        except:
            self.decision_history = []