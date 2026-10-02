"""
Evolution Engine Module for WILLAM - Enhanced with Psychological Learning
Learns MORE from losses than wins, gets "frustrated" with repeated failures,
analyzes specific losing setups, generates solutions, and evolves daily
with a relentless profit-per-trade mindset.
"""

import asyncio
import json
import numpy as np
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
from collections import defaultdict, Counter


@dataclass
class TradeRecord:
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    sl: float
    tp: float
    lot_size: float
    strategy_name: str
    pnl: float
    close_reason: str  # "sl", "tp", "manual"
    duration_minutes: float
    timestamp: str
    news_context: Optional[Dict] = None
    
    # Enhanced: Setup fingerprint for pattern matching
    setup_fingerprint: str = ""
    market_regime: str = ""
    session: str = ""
    volatility_percentile: float = 0.0
    spread_pips: float = 0.0
    
    # Psychological weight
    learning_weight: float = 1.0  # Losses get higher weight


@dataclass
class LosingPattern:
    """A specific losing setup that the bot has identified and is trying to solve."""
    pattern_id: str
    fingerprint: str
    symbol: str
    strategy: str
    regime: str
    session: str
    description: str
    
    # Statistics
    occurrences: int = 0
    total_loss: float = 0.0
    avg_loss: float = 0.0
    max_loss: float = 0.0
    winrate: float = 0.0
    
    # Frustration tracking
    consecutive_losses: int = 0
    frustration_level: float = 0.0  # 0-1, increases with repeated failures
    last_occurrence: str = ""
    
    # Solution attempts
    attempted_solutions: List[Dict] = field(default_factory=list)
    current_solution: Optional[str] = None
    solution_effectiveness: float = 0.0
    
    # Status
    status: str = "active"  # "active", "solved", "monitoring", "archived"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    solved_at: Optional[str] = None


@dataclass
class StrategyPerformance:
    name: str
    symbol: str
    total_trades: int
    wins: int
    losses: int
    winrate: float
    expectancy: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    max_drawdown: float
    sharpe: float
    last_updated: str
    trades_since_update: int = 0
    
    # Enhanced: Psychological metrics
    loss_aversion_weight: float = 2.0  # Learn 2x more from losses
    frustration_index: float = 0.0  # Increases with consecutive losses
    curiosity_drive: float = 1.0  # Increases with learning
    improvement_trajectory: List[float] = field(default_factory=list)  # Daily expectancy change
    profit_streak: int = 0  # Consecutive profitable trades
    loss_streak: int = 0  # Consecutive losing trades


class EvolutionEngine:
    """
    Evolution Engine - The Survival Mechanism with Psychological Learning
    
    Core Philosophy:
    - "Learn MORE from losses than wins" - Loss aversion weighting
    - "Get frustrated with losses" - Frustration drives deeper analysis
    - "Find the losing setup and solve it" - Pattern-specific solutions
    - "Never lose to the same setup twice" - Memory + adaptation
    - "Curiosity increases with learning" - More knowledge = more questions
    - "Profit per trade mindset" - Every trade must improve the system
    - "Survive with 100% will" - Relentless adaptation
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.evo_config = config.get("evolution", {})
        self.enabled = self.evo_config.get("enabled", True)
        self.learning_rate = self.evo_config.get("learning_rate", 0.1)
        self.memory_window = self.evo_config.get("memory_window_trades", 500)
        self.survival_threshold = self.evo_config.get("survival_threshold", 0.5)
        self.gap_analysis_enabled = self.evo_config.get("gap_analysis_enabled", True)
        
        # PSYCHOLOGICAL PARAMETERS
        self.loss_learning_multiplier = self.evo_config.get("loss_learning_multiplier", 3.0)  # Learn 3x from losses
        self.frustration_threshold = self.evo_config.get("frustration_threshold", 3)  # Consecutive losses before deep dive
        self.curiosity_growth_rate = self.evo_config.get("curiosity_growth_rate", 0.05)  # Curiosity grows with learning
        self.max_frustration = self.evo_config.get("max_frustration", 1.0)
        self.solution_test_trades = self.evo_config.get("solution_test_trades", 20)  # Trades to validate a solution
        self.profit_per_trade_target = self.evo_config.get("profit_per_trade_target", 0.001)  # Target expectancy per trade
        
        # Memory
        self.trade_memory: List[TradeRecord] = []
        self.strategy_performance: Dict[str, StrategyPerformance] = {}
        self.gap_analysis: Dict[str, List[Dict]] = defaultdict(list)
        
        # NEW: Losing pattern database
        self.losing_patterns: Dict[str, LosingPattern] = {}
        self.pattern_solutions: Dict[str, List[Dict]] = defaultdict(list)
        
        # NEW: Daily learning journal
        self.daily_learning: Dict[str, Dict] = {}  # date -> {trades, lessons, improvements, frustration, curiosity}
        self.current_day = datetime.utcnow().date().isoformat()
        
        # NEW: Curiosity engine
        self.curiosity_questions: List[Dict] = []  # Questions the bot asks itself
        self.exploration_queue: List[Dict] = []  # Hypotheses to test
        
        # Paths
        self.memory_path = Path("evolution/memory.json")
        self.performance_path = Path("evolution/performance.json")
        self.gaps_path = Path("evolution/gaps.json")
        self.patterns_path = Path("evolution/losing_patterns.json")
        self.daily_path = Path("evolution/daily_learning.json")
        self.curiosity_path = Path("evolution/curiosity.json")
        self.memory_path.parent.mkdir(exist_ok=True)
    
    async def load_memory(self):
        """Load all evolution memory from disk."""
        try:
            if self.memory_path.exists():
                with open(self.memory_path) as f:
                    data = json.load(f)
                    self.trade_memory = [TradeRecord(**t) for t in data.get("trades", [])]
            
            if self.performance_path.exists():
                with open(self.performance_path) as f:
                    data = json.load(f)
                    for k, v in data.items():
                        self.strategy_performance[k] = StrategyPerformance(**v)
            
            if self.gaps_path.exists():
                with open(self.gaps_path) as f:
                    self.gap_analysis = defaultdict(list, json.load(f))
            
            if self.patterns_path.exists():
                with open(self.patterns_path) as f:
                    data = json.load(f)
                    self.losing_patterns = {k: LosingPattern(**v) for k, v in data.items()}
            
            if self.daily_path.exists():
                with open(self.daily_path) as f:
                    self.daily_learning = json.load(f)
            
            if self.curiosity_path.exists():
                with open(self.curiosity_path) as f:
                    data = json.load(f)
                    self.curiosity_questions = data.get("questions", [])
                    self.exploration_queue = data.get("explorations", [])
            
            self.current_day = datetime.utcnow().date().isoformat()
            self._init_daily_learning()
            
            print(f"EVOLUTION: Memory loaded - {len(self.trade_memory)} trades, "
                  f"{len(self.strategy_performance)} strategies, "
                  f"{len(self.losing_patterns)} losing patterns tracked")
            print(f"EVOLUTION: Curiosity level: {self._get_curiosity_level():.2f}, "
                  f"Active patterns: {sum(1 for p in self.losing_patterns.values() if p.status == 'active')}")
        except Exception as e:
            print(f"EVOLUTION: Failed to load memory: {e}")
    
    def _init_daily_learning(self):
        """Initialize today's learning journal."""
        if self.current_day not in self.daily_learning:
            self.daily_learning[self.current_day] = {
                "date": self.current_day,
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "net_pnl": 0.0,
                "lessons_learned": [],
                "patterns_identified": [],
                "solutions_tested": [],
                "frustration_peaks": [],
                "curiosity_questions": [],
                "improvement_score": 0.0,
                "profit_per_trade": 0.0,
            }
    
    def _get_curiosity_level(self) -> float:
        """Calculate current curiosity level based on knowledge and unsolved problems."""
        base = 1.0
        knowledge_bonus = len(self.trade_memory) * 0.001
        unsolved_problems = sum(1 for p in self.losing_patterns.values() if p.status == "active")
        frustration_bonus = sum(p.frustration_level for p in self.losing_patterns.values())
        return min(base + knowledge_bonus + frustration_bonus * 0.5 + unsolved_problems * 0.1, 10.0)
    
    async def save_memory(self):
        """Save all evolution memory to disk."""
        try:
            # Save trades
            with open(self.memory_path, "w") as f:
                json.dump({
                    "trades": [asdict(t) for t in self.trade_memory[-self.memory_window:]],
                    "saved_at": datetime.utcnow().isoformat()
                }, f, indent=2)
            
            # Save performance
            with open(self.performance_path, "w") as f:
                json.dump({
                    k: asdict(v) for k, v in self.strategy_performance.items()
                }, f, indent=2)
            
            # Save gaps
            with open(self.gaps_path, "w") as f:
                json.dump(dict(self.gap_analysis), f, indent=2)
            
            # Save losing patterns
            with open(self.patterns_path, "w") as f:
                json.dump({k: asdict(v) for k, v in self.losing_patterns.items()}, f, indent=2)
            
            # Save daily learning
            with open(self.daily_path, "w") as f:
                json.dump(self.daily_learning, f, indent=2)
            
            # Save curiosity
            with open(self.curiosity_path, "w") as f:
                json.dump({
                    "questions": self.curiosity_questions[-100:],
                    "explorations": self.exploration_queue[-50:],
                    "curiosity_level": self._get_curiosity_level(),
                    "saved_at": datetime.utcnow().isoformat()
                }, f, indent=2)
                
        except Exception as e:
            print(f"EVOLUTION: Failed to save memory: {e}")
    
    def _generate_setup_fingerprint(self, position: Any, market_data: Dict = None) -> str:
        """Generate a unique fingerprint for the trade setup."""
        # Create a hash of the setup conditions
        components = [
            position.symbol,
            position.strategy_name,
            market_data.get("regime", "unknown") if market_data else "unknown",
            market_data.get("session", "unknown") if market_data else "unknown",
            f"vol_{int(market_data.get('volatility_percentile', 50) / 10) * 10}" if market_data else "vol_50",
            f"spread_{int(market_data.get('spread_pips', 1) * 10)}" if market_data else "spread_10",
            position.direction,
        ]
        fingerprint_str = "|".join(components)
        return hashlib.md5(fingerprint_str.encode()).hexdigest()[:12]
    
    def _get_market_context(self, position: Any) -> Dict:
        """Extract market context at trade time."""
        # This would ideally pull from market data at entry time
        # For now, infer from timestamp
        try:
            entry_time = datetime.fromisoformat(position.open_time.replace('Z', '+00:00'))
            hour = entry_time.hour
            
            if 0 <= hour < 8:
                session = "asian"
            elif 8 <= hour < 16:
                session = "london"
            elif 13 <= hour < 21:
                session = "ny"
            else:
                session = "asian"
            
            if 12 <= hour < 16:
                session += "_overlap"
        except:
            session = "unknown"
        
        return {
            "session": session,
            "regime": "unknown",  # Would need market data
            "volatility_percentile": 50,
            "spread_pips": 1.0,
        }
    
    async def learn_from_trade(self, position: Any, exit_price: float, close_reason: str, pnl: float):
        """
        Learn from a completed trade - WITH PSYCHOLOGICAL WEIGHTING.
        Losses teach 3x more than wins. Frustration builds with repeated failures.
        """
        if not self.enabled:
            return
        
        # Check day rollover
        today = datetime.utcnow().date().isoformat()
        if today != self.current_day:
            await self._finalize_day()
            self.current_day = today
            self._init_daily_learning()
        
        market_context = self._get_market_context(position)
        fingerprint = self._generate_setup_fingerprint(position, market_context)
        
        # Determine learning weight - LOSSES TEACH MORE
        is_loss = pnl <= 0
        learning_weight = self.loss_learning_multiplier if is_loss else 1.0
        
        # Create enhanced trade record
        record = TradeRecord(
            symbol=position.symbol,
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=exit_price,
            sl=position.sl,
            tp=position.tp,
            lot_size=position.lot_size,
            strategy_name=position.strategy_name,
            pnl=pnl,
            close_reason=close_reason,
            duration_minutes=(datetime.utcnow() - position.open_time).total_seconds() / 60,
            timestamp=datetime.utcnow().isoformat(),
            news_context=position.news_context if hasattr(position, 'news_context') else None,
            setup_fingerprint=fingerprint,
            market_regime=market_context.get("regime", "unknown"),
            session=market_context.get("session", "unknown"),
            volatility_percentile=market_context.get("volatility_percentile", 50),
            spread_pips=market_context.get("spread_pips", 1.0),
            learning_weight=learning_weight
        )
        
        self.trade_memory.append(record)
        
        # Update strategy performance with psychological weighting
        key = f"{position.symbol}_{position.strategy_name}"
        await self._update_performance_psychological(key, position.symbol, position.strategy_name, record, learning_weight)
        
        # CRITICAL: Track losing patterns
        if is_loss:
            await self._process_loss(position, record, fingerprint, market_context, pnl, close_reason)
        else:
            await self._process_win(position, record, fingerprint, pnl)
        
        # Update daily journal
        daily = self.daily_learning[self.current_day]
        daily["trades"] += 1
        daily["net_pnl"] += pnl
        if is_loss:
            daily["losses"] += 1
        else:
            daily["wins"] += 1
        
        # Generate curiosity question from this trade
        await self._generate_curiosity(record, is_loss)
        
        # Analyze gaps periodically
        if self.gap_analysis_enabled and len(self.trade_memory) % 10 == 0:
            await self._analyze_gaps(position.symbol)
        
        # Trim memory
        if len(self.trade_memory) > self.memory_window:
            self.trade_memory = self.trade_memory[-self.memory_window:]
    
    async def _update_performance_psychological(self, key: str, symbol: str, strategy: str, record: TradeRecord, weight: float):
        """Update performance with psychological weighting."""
        if key not in self.strategy_performance:
            self.strategy_performance[key] = StrategyPerformance(
                name=strategy, symbol=symbol, total_trades=0, wins=0, losses=0,
                winrate=0.0, expectancy=0.0, profit_factor=0.0, avg_win=0.0, avg_loss=0.0,
                max_drawdown=0.0, sharpe=0.0, last_updated=datetime.utcnow().isoformat(),
                loss_aversion_weight=self.loss_learning_multiplier
            )
        
        perf = self.strategy_performance[key]
        perf.total_trades += 1
        perf.trades_since_update += 1
        
        if record.pnl > 0:
            perf.wins += 1
            perf.loss_streak = 0
            perf.profit_streak += 1
            perf.frustration_index = max(0, perf.frustration_index - 0.05)  # Wins reduce frustration
        else:
            perf.losses += 1
            perf.profit_streak = 0
            perf.loss_streak += 1
            perf.frustration_index = min(1.0, perf.frustration_index + 0.1)  # Losses increase frustration
        
        perf.winrate = perf.wins / perf.total_trades * 100
        
        # Weighted expectancy calculation
        wins = [t for t in self.trade_memory if f"{t.symbol}_{t.strategy_name}" == key and t.pnl > 0]
        losses = [t for t in self.trade_memory if f"{t.symbol}_{t.strategy_name}" == key and t.pnl <= 0]
        
        if wins:
            perf.avg_win = np.mean([t.pnl * t.learning_weight for t in wins])
        if losses:
            perf.avg_loss = np.mean([t.pnl * t.learning_weight for t in losses])
        
        if perf.avg_loss != 0:
            perf.profit_factor = abs(perf.avg_win * perf.wins / (perf.avg_loss * perf.losses)) if perf.losses > 0 else 999
        
        # Standard expectancy (unweighted for actual PnL tracking)
        raw_wins = [t.pnl for t in wins]
        raw_losses = [t.pnl for t in losses]
        if raw_wins and raw_losses:
            perf.expectancy = (perf.winrate / 100 * np.mean(raw_wins)) - ((100 - perf.winrate) / 100 * abs(np.mean(raw_losses)))
        
        # Curiosity grows with learning
        perf.curiosity_drive = min(5.0, 1.0 + len(self.trade_memory) * 0.002 + perf.frustration_index)
        
        # Track daily improvement trajectory
        if len(self.trade_memory) % 20 == 0:
            perf.improvement_trajectory.append(perf.expectancy)
            if len(perf.improvement_trajectory) > 30:
                perf.improvement_trajectory = perf.improvement_trajectory[-30:]
        
        perf.last_updated = datetime.utcnow().isoformat()
    
    async def _process_loss(self, position: Any, record: TradeRecord, fingerprint: str, 
                           market_context: Dict, pnl: float, close_reason: str):
        """
        PROCESS A LOSS - This is where the real learning happens.
        Identify the pattern, increase frustration, generate solutions.
        """
        pattern_key = f"{position.symbol}_{position.strategy_name}_{fingerprint}"
        
        if pattern_key not in self.losing_patterns:
            # NEW LOSING PATTERN DISCOVERED
            self.losing_patterns[pattern_key] = LosingPattern(
                pattern_id=pattern_key,
                fingerprint=fingerprint,
                symbol=position.symbol,
                strategy=position.strategy_name,
                regime=market_context.get("regime", "unknown"),
                session=market_context.get("session", "unknown"),
                description=self._describe_pattern(position, market_context, fingerprint),
                occurrences=1,
                total_loss=pnl,
                avg_loss=pnl,
                max_loss=pnl,
                consecutive_losses=1,
                frustration_level=0.2,
                last_occurrence=record.timestamp,
            )
            print(f"EVOLUTION: 🔴 NEW LOSING PATTERN: {pattern_key} - {self.losing_patterns[pattern_key].description}")
            
            # Generate immediate curiosity questions
            await self._generate_pattern_questions(pattern_key)
            
        else:
            # EXISTING PATTERN - FRUSTRATION BUILDS
            pattern = self.losing_patterns[pattern_key]
            pattern.occurrences += 1
            pattern.total_loss += pnl
            pattern.avg_loss = pattern.total_loss / pattern.occurrences
            pattern.max_loss = min(pattern.max_loss, pnl)
            pattern.consecutive_losses += 1
            pattern.frustration_level = min(self.max_frustration, 
                pattern.frustration_level + 0.15 * pattern.consecutive_losses)
            pattern.last_occurrence = record.timestamp
            
            print(f"EVOLUTION: 🔴 REPEATED LOSS PATTERN: {pattern_key} "
                  f"(#{pattern.occurrences}, frustration={pattern.frustration_level:.2f})")
            
            # FRUSTRATION TRIGGERS DEEP ANALYSIS
            if pattern.consecutive_losses >= self.frustration_threshold:
                await self._deep_dive_analysis(pattern)
            
            # Log to daily
            self.daily_learning[self.current_day]["frustration_peaks"].append({
                "pattern": pattern_key,
                "frustration": pattern.frustration_level,
                "consecutive": pattern.consecutive_losses
            })
        
        # Track in daily lessons
        self.daily_learning[self.current_day]["lessons_learned"].append({
            "type": "loss_analysis",
            "pattern": pattern_key,
            "lesson": f"Lost {pnl:.2f} on {position.symbol} {position.strategy_name} - {close_reason}",
            "action": "Analyzing setup fingerprint for solution"
        })
    
    async def _process_win(self, position: Any, record: TradeRecord, fingerprint: str, pnl: float):
        """Process a win - validate solutions, reduce frustration."""
        pattern_key = f"{position.symbol}_{position.strategy_name}_{fingerprint}"
        
        # Check if this win validates a solution for a known losing pattern
        if pattern_key in self.losing_patterns:
            pattern = self.losing_patterns[pattern_key]
            pattern.consecutive_losses = 0  # Reset consecutive losses
            pattern.frustration_level = max(0, pattern.frustration_level - 0.2)
            
            # If we were testing a solution, evaluate it
            if pattern.current_solution:
                pattern.solution_effectiveness = (pattern.solution_effectiveness * 0.7) + 0.3
                if pattern.solution_effectiveness > 0.7 and pattern.occurrences >= 5:
                    pattern.status = "solved"
                    pattern.solved_at = datetime.utcnow().isoformat()
                    print(f"EVOLUTION: ✅ PATTERN SOLVED: {pattern_key} via {pattern.current_solution}")
                    self.daily_learning[self.current_day]["lessons_learned"].append({
                        "type": "pattern_solved",
                        "pattern": pattern_key,
                        "solution": pattern.current_solution
                    })
    
    def _describe_pattern(self, position: Any, market_context: Dict, fingerprint: str) -> str:
        """Human-readable description of the losing setup."""
        parts = [
            f"{position.symbol}",
            f"{position.strategy_name}",
            f"{position.direction}",
            f"session:{market_context.get('session', 'unknown')}",
            f"regime:{market_context.get('regime', 'unknown')}",
            f"fp:{fingerprint[:8]}"
        ]
        return " | ".join(parts)
    
    async def _generate_pattern_questions(self, pattern_key: str):
        """Generate curiosity questions for a new losing pattern."""
        pattern = self.losing_patterns[pattern_key]
        questions = [
            {
                "pattern": pattern_key,
                "question": f"Why does {pattern.strategy} fail on {pattern.symbol} in {pattern.session} session?",
                "type": "root_cause",
                "priority": "high",
                "created": datetime.utcnow().isoformat()
            },
            {
                "pattern": pattern_key,
                "question": f"What market condition (volatility, spread, regime) makes this setup fail?",
                "type": "condition_analysis",
                "priority": "high",
                "created": datetime.utcnow().isoformat()
            },
            {
                "pattern": pattern_key,
                "question": f"Should we add a filter, change SL/TP, or abandon this setup entirely?",
                "type": "solution_design",
                "priority": "critical",
                "created": datetime.utcnow().isoformat()
            },
        ]
        self.curiosity_questions.extend(questions)
        self.daily_learning[self.current_day]["curiosity_questions"].extend(questions)
    
    async def _deep_dive_analysis(self, pattern: LosingPattern):
        """
        DEEP DIVE - Triggered by high frustration.
        Analyzes ALL trades matching this pattern to find the root cause.
        """
        print(f"EVOLUTION: 🧠 DEEP DIVE TRIGGERED for {pattern.pattern_id} (frustration={pattern.frustration_level:.2f})")
        
        # Get all trades for this pattern
        pattern_trades = [
            t for t in self.trade_memory 
            if t.setup_fingerprint == pattern.fingerprint
        ]
        
        if len(pattern_trades) < 3:
            return
        
        # Analyze by close reason
        sl_losses = [t for t in pattern_trades if t.close_reason == "sl"]
        tp_hits = [t for t in pattern_trades if t.close_reason == "tp"]
        
        # Analyze by volatility
        high_vol = [t for t in pattern_trades if t.volatility_percentile > 70]
        low_vol = [t for t in pattern_trades if t.volatility_percentile < 30]
        
        # Analyze by spread
        high_spread = [t for t in pattern_trades if t.spread_pips > 2.0]
        
        # Generate specific solutions based on analysis
        solutions = []
        
        if len(sl_losses) / len(pattern_trades) > 0.7:
            solutions.append({
                "type": "wider_sl",
                "description": "Increase SL multiplier from 1.5x to 2.0x ATR",
                "parameter_change": {"atr_multiplier_sl": 2.0},
                "hypothesis": "SL too tight for this volatility regime"
            })
        
        if high_vol and sum(t.pnl for t in high_vol) < 0:
            solutions.append({
                "type": "volatility_filter",
                "description": "Skip this setup when volatility > 70th percentile",
                "parameter_change": {"max_volatility_percentile": 70},
                "hypothesis": "Strategy fails in high volatility"
            })
        
        if high_spread and sum(t.pnl for t in high_spread) < 0:
            solutions.append({
                "type": "spread_filter",
                "description": "Skip when spread > 2 pips",
                "parameter_change": {"max_spread_pips": 2.0},
                "hypothesis": "Spread eats profits on this setup"
            })
        
        if pattern.session in ["asian", "asian_overlap"] and sum(t.pnl for t in pattern_trades) < 0:
            solutions.append({
                "type": "session_filter",
                "description": f"Disable {pattern.strategy} during {pattern.session} session",
                "parameter_change": {"disabled_sessions": [pattern.session]},
                "hypothesis": f"Strategy incompatible with {pattern.session} liquidity"
            })
        
        # Store solutions for testing
        for sol in solutions:
            sol["pattern_id"] = pattern.pattern_id
            sol["proposed_at"] = datetime.utcnow().isoformat()
            sol["status"] = "proposed"
            self.pattern_solutions[pattern.pattern_id].append(sol)
            self.exploration_queue.append(sol)
        
        pattern.attempted_solutions.extend(solutions)
        self.daily_learning[self.current_day]["solutions_tested"].extend(solutions)
        
        print(f"EVOLUTION: 💡 Generated {len(solutions)} solutions for {pattern.pattern_id}")
    
    async def _generate_curiosity(self, record: TradeRecord, is_loss: bool):
        """Generate curiosity questions from every trade."""
        base_questions = []
        
        if is_loss:
            base_questions = [
                f"What specific condition made this {record.strategy_name} setup fail on {record.symbol}?",
                f"Was the SL hit due to noise or genuine trend reversal?",
                f"Did the news context ({record.news_context}) mislead the entry?",
            ]
        else:
            base_questions = [
                f"Why did this {record.strategy_name} setup work on {record.symbol}?",
                f"Can we replicate this exact setup more often?",
                f"Was the win due to skill or luck (market moved our way)?",
            ]
        
        for q in base_questions:
            self.curiosity_questions.append({
                "trade_id": f"{record.symbol}_{record.timestamp[:19]}",
                "question": q,
                "type": "loss_analysis" if is_loss else "win_analysis",
                "priority": "high" if is_loss else "medium",
                "created": datetime.utcnow().isoformat()
            })
        
        # Curiosity grows with every trade
        curiosity_growth = self.curiosity_growth_rate * (self.loss_learning_multiplier if is_loss else 1.0)
    
    async def _analyze_gaps(self, symbol: str):
        """Enhanced gap analysis with psychological insights."""
        symbol_strategies = {
            k: v for k, v in self.strategy_performance.items()
            if v.symbol == symbol and v.total_trades >= 20
        }
        
        if not symbol_strategies:
            return
        
        gaps = []
        
        for key, perf in symbol_strategies.items():
            # Standard gaps
            if perf.winrate < 50:
                gaps.append({"type": "low_winrate", "strategy": perf.name, "value": perf.winrate, "severity": "high" if perf.winrate < 40 else "medium"})
            
            if perf.max_drawdown > 15:
                gaps.append({"type": "high_drawdown", "strategy": perf.name, "value": perf.max_drawdown, "severity": "high" if perf.max_drawdown > 25 else "medium"})
            
            if perf.expectancy <= 0:
                gaps.append({"type": "negative_expectancy", "strategy": perf.name, "value": perf.expectancy, "severity": "critical"})
            
            # PSYCHOLOGICAL GAPS
            if perf.frustration_index > 0.5:
                gaps.append({"type": "high_frustration", "strategy": perf.name, "value": perf.frustration_index, "severity": "high"})
            
            if perf.loss_streak >= 3:
                gaps.append({"type": "loss_streak", "strategy": perf.name, "value": perf.loss_streak, "severity": "high"})
            
            if perf.curiosity_drive > 3.0 and perf.expectancy < self.profit_per_trade_target:
                gaps.append({"type": "curiosity_unmet", "strategy": perf.name, "value": perf.curiosity_drive, "severity": "medium"})
            
            # Session analysis
            session_gaps = await self._analyze_session_performance(symbol, perf.name)
            gaps.extend(session_gaps)
        
        self.gap_analysis[symbol] = gaps[-50:]
    
    async def _analyze_session_performance(self, symbol: str, strategy: str) -> List[Dict]:
        """Analyze performance by trading session."""
        gaps = []
        trades = [t for t in self.trade_memory if t.symbol == symbol and t.strategy_name == strategy]
        
        if len(trades) < 15:
            return gaps
        
        sessions = {"asian": (0, 8), "london": (8, 16), "ny": (13, 21), "overlap": (12, 16)}
        
        for session_name, (start, end) in sessions.items():
            session_trades = [t for t in trades if start <= datetime.fromisoformat(t.timestamp).hour < end]
            if len(session_trades) >= 8:
                winrate = sum(1 for t in session_trades if t.pnl > 0) / len(session_trades) * 100
                expectancy = np.mean([t.pnl for t in session_trades])
                if winrate < 45 or expectancy < 0:
                    gaps.append({
                        "type": f"session_weakness_{session_name}",
                        "strategy": strategy, "session": session_name,
                        "winrate": winrate, "expectancy": expectancy,
                        "trades": len(session_trades),
                        "severity": "high" if expectancy < 0 else "medium"
                    })
        return gaps
    
    async def step(self, closed_trades: List[Any]):
        """Periodic evolution step - cull, test solutions, evolve."""
        if not self.enabled:
            return
        
        await self._cull_weak_strategies()
        await self._test_solutions()
        await self._generate_new_variants()
        await self._update_daily_improvement_score()
        await self.save_memory()
    
    async def _cull_weak_strategies(self):
        """Kill strategies that don't meet survival threshold."""
        to_remove = []
        for key, perf in self.strategy_performance.items():
            if perf.total_trades >= 50:
                if perf.expectancy <= 0 or perf.winrate < 45:
                    print(f"EVOLUTION: 💀 CULLING {key} - expectancy={perf.expectancy:.4f}, winrate={perf.winrate:.1f}%, frustration={perf.frustration_index:.2f}")
                    to_remove.append(key)
        
        for key in to_remove:
            # Archive the losing patterns for this strategy
            for pattern in self.losing_patterns.values():
                if pattern.strategy == self.strategy_performance[key].name and pattern.symbol == self.strategy_performance[key].symbol:
                    pattern.status = "archived"
            del self.strategy_performance[key]
    
    async def _test_solutions(self):
        """Test proposed solutions against recent trades."""
        for pattern_id, solutions in self.pattern_solutions.items():
            for sol in solutions:
                if sol.get("status") == "proposed":
                    # Mark as testing
                    sol["status"] = "testing"
                    sol["test_start"] = datetime.utcnow().isoformat()
                    sol["test_trades"] = 0
                    sol["test_pnl"] = 0.0
                    print(f"EVOLUTION: 🧪 TESTING SOLUTION: {sol['description']} for {pattern_id}")
    
    async def _generate_new_variants(self):
        """Generate new strategy variants from curiosity and gaps."""
        curiosity = self._get_curiosity_level()
        
        if curiosity > 3.0 and len(self.exploration_queue) < 5:
            # High curiosity + room for exploration = generate hypotheses
            for symbol in self.config.get("trading", {}).get("symbols", []):
                for strat_name in ["trend_follow", "mean_reversion", "breakout"]:
                    key = f"{symbol}_{strat_name}"
                    if key in self.strategy_performance:
                        perf = self.strategy_performance[key]
                        if perf.expectancy > 0 and perf.frustration_index < 0.3:
                            # Successful strategy - explore variants
                            self.exploration_queue.append({
                                "type": "variant_exploration",
                                "base_strategy": strat_name,
                                "symbol": symbol,
                                "hypothesis": f"Can we improve {strat_name} on {symbol} by adjusting parameters?",
                                "priority": "medium",
                                "created": datetime.utcnow().isoformat()
                            })
    
    async def _update_daily_improvement_score(self):
        """Calculate daily improvement score."""
        daily = self.daily_learning[self.current_day]
        if daily["trades"] > 0:
            daily["profit_per_trade"] = daily["net_pnl"] / daily["trades"]
            
            # Improvement score: profit per trade + lessons learned - frustration
            lessons = len(daily["lessons_learned"])
            frustration_penalty = len(daily["frustration_peaks"]) * 0.1
            daily["improvement_score"] = daily["profit_per_trade"] + (lessons * 0.01) - frustration_penalty
            
            print(f"EVOLUTION: 📊 Daily Score: {daily['improvement_score']:.4f} "
                  f"(PnL/trade: {daily['profit_per_trade']:.4f}, "
                  f"Lessons: {lessons}, Frustration: {frustration_penalty:.2f})")
    
    async def _finalize_day(self):
        """Finalize daily learning and prepare for next day."""
        if self.current_day in self.daily_learning:
            daily = self.daily_learning[self.current_day]
            print(f"EVOLUTION: 🌅 DAY COMPLETE: {self.current_day} - "
                  f"Trades: {daily['trades']}, PnL: {daily['net_pnl']:.2f}, "
                  f"Improvement: {daily['improvement_score']:.4f}, "
                  f"Patterns: {len(daily['patterns_identified'])}, "
                  f"Solutions: {len(daily['solutions_tested'])}")
    
    def get_best_strategy(self, symbol: str) -> Optional[str]:
        """Get best strategy - considers psychological fitness."""
        symbol_strategies = {
            k: v for k, v in self.strategy_performance.items()
            if v.symbol == symbol and v.total_trades >= 20
        }
        
        if not symbol_strategies:
            return None
        
        # Fitness = expectancy * winrate * (1 - frustration) * curiosity
        best = max(symbol_strategies.values(), 
                   key=lambda p: p.expectancy * p.winrate / 100 * (1 - p.frustration_index * 0.5) * min(p.curiosity_drive, 2.0))
        return best.name
    
    def get_evolution_report(self) -> Dict:
        """Comprehensive evolution status report."""
        active_patterns = [p for p in self.losing_patterns.values() if p.status == "active"]
        solved_patterns = [p for p in self.losing_patterns.values() if p.status == "solved"]
        
        return {
            "total_trades_learned": len(self.trade_memory),
            "strategies_tracked": len(self.strategy_performance),
            "active_losing_patterns": len(active_patterns),
            "solved_patterns": len(solved_patterns),
            "symbols_with_gaps": len(self.gap_analysis),
            "curiosity_level": self._get_curiosity_level(),
            "total_frustration": sum(p.frustration_level for p in self.losing_patterns.values()),
            "survival_threshold": self.survival_threshold,
            "learning_rate": self.learning_rate,
            "loss_learning_multiplier": self.loss_learning_multiplier,
            "profit_per_trade_target": self.profit_per_trade_target,
            "top_strategies": self._get_top_strategies(),
            "worst_patterns": sorted(active_patterns, key=lambda p: p.frustration_level, reverse=True)[:5],
            "pending_explorations": len(self.exploration_queue),
            "daily_score": self.daily_learning.get(self.current_day, {}).get("improvement_score", 0)
        }
    
    def _get_top_strategies(self) -> List[Dict]:
        strategies = [
            {
                "key": k, "symbol": v.symbol, "strategy": v.name,
                "winrate": v.winrate, "expectancy": v.expectancy,
                "profit_factor": v.profit_factor, "trades": v.total_trades,
                "frustration": v.frustration_index, "curiosity": v.curiosity_drive,
                "fitness": v.expectancy * v.winrate / 100 * (1 - v.frustration_index * 0.5)
            }
            for k, v in self.strategy_performance.items()
            if v.total_trades >= 20
        ]
        strategies.sort(key=lambda x: x["fitness"], reverse=True)
        return strategies[:5]
    
    def get_pattern_report(self, symbol: str = None) -> List[Dict]:
        """Get detailed losing pattern report."""
        patterns = []
        for p in self.losing_patterns.values():
            if symbol is None or p.symbol == symbol:
                patterns.append({
                    "pattern_id": p.pattern_id,
                    "symbol": p.symbol,
                    "strategy": p.strategy,
                    "description": p.description,
                    "occurrences": p.occurrences,
                    "avg_loss": p.avg_loss,
                    "max_loss": p.max_loss,
                    "consecutive_losses": p.consecutive_losses,
                    "frustration_level": p.frustration_level,
                    "status": p.status,
                    "current_solution": p.current_solution,
                    "solution_effectiveness": p.solution_effectiveness,
                    "solutions_tested": len(p.attempted_solutions)
                })
        patterns.sort(key=lambda x: x["frustration_level"], reverse=True)
        return patterns