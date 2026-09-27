"""
Evolution Engine Module for WILLAM
Learns from trades, fills gaps, evolves strategies
"""

import asyncio
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict


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


class EvolutionEngine:
    """
    Evolution Engine - The survival mechanism
    
    Principles:
    1. Learn from EVERY trade (win or loss)
    2. Identify gaps in strategy performance
    3. Fill gaps by combining/adapting strategies
    4. Kill strategies that don't evolve
    5. Survive and generate money continuously
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.evo_config = config.get("evolution", {})
        self.enabled = self.evo_config.get("enabled", True)
        self.learning_rate = self.evo_config.get("learning_rate", 0.1)
        self.memory_window = self.evo_config.get("memory_window_trades", 500)
        self.survival_threshold = self.evo_config.get("survival_threshold", 0.5)
        self.gap_analysis_enabled = self.evo_config.get("gap_analysis_enabled", True)
        
        # Memory
        self.trade_memory: List[TradeRecord] = []
        self.strategy_performance: Dict[str, StrategyPerformance] = {}  # key: f"{symbol}_{strategy}"
        self.gap_analysis: Dict[str, List[Dict]] = defaultdict(list)  # symbol -> gaps
        
        # Paths
        self.memory_path = Path("evolution/memory.json")
        self.performance_path = Path("evolution/performance.json")
        self.gaps_path = Path("evolution/gaps.json")
        self.memory_path.parent.mkdir(exist_ok=True)
    
    async def load_memory(self):
        """Load evolution memory from disk."""
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
            
            print(f"Evolution memory loaded: {len(self.trade_memory)} trades, {len(self.strategy_performance)} strategies")
        except Exception as e:
            print(f"Failed to load evolution memory: {e}")
    
    async def save_memory(self):
        """Save evolution memory to disk."""
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
                
        except Exception as e:
            print(f"Failed to save evolution memory: {e}")
    
    async def learn_from_trade(self, position: Any, exit_price: float, close_reason: str, pnl: float):
        """Learn from a completed trade."""
        if not self.enabled:
            return
        
        # Create trade record
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
            timestamp=datetime.utcnow().isoformat()
        )
        
        self.trade_memory.append(record)
        
        # Update strategy performance
        key = f"{position.symbol}_{position.strategy_name}"
        await self._update_performance(key, position.symbol, position.strategy_name, record)
        
        # Analyze gaps if enabled
        if self.gap_analysis_enabled and len(self.trade_memory) % 20 == 0:
            await self._analyze_gaps(position.symbol)
        
        # Trim memory
        if len(self.trade_memory) > self.memory_window:
            self.trade_memory = self.trade_memory[-self.memory_window:]
    
    async def _update_performance(self, key: str, symbol: str, strategy: str, record: TradeRecord):
        """Update strategy performance metrics."""
        if key not in self.strategy_performance:
            self.strategy_performance[key] = StrategyPerformance(
                name=strategy,
                symbol=symbol,
                total_trades=0,
                wins=0,
                losses=0,
                winrate=0.0,
                expectancy=0.0,
                profit_factor=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                max_drawdown=0.0,
                sharpe=0.0,
                last_updated=datetime.utcnow().isoformat()
            )
        
        perf = self.strategy_performance[key]
        perf.total_trades += 1
        perf.trades_since_update += 1
        
        if record.pnl > 0:
            perf.wins += 1
        else:
            perf.losses += 1
        
        perf.winrate = perf.wins / perf.total_trades * 100
        
        # Recalculate expectancy
        wins = [t for t in self.trade_memory if f"{t.symbol}_{t.strategy_name}" == key and t.pnl > 0]
        losses = [t for t in self.trade_memory if f"{t.symbol}_{t.strategy_name}" == key and t.pnl <= 0]
        
        if wins:
            perf.avg_win = np.mean([t.pnl for t in wins])
        if losses:
            perf.avg_loss = np.mean([t.pnl for t in losses])
        
        if perf.avg_loss != 0:
            perf.profit_factor = abs(perf.avg_win * perf.wins / (perf.avg_loss * perf.losses)) if perf.losses > 0 else 999
        
        perf.expectancy = (perf.winrate / 100 * perf.avg_win) - ((100 - perf.winrate) / 100 * abs(perf.avg_loss))
        
        # Simple Sharpe approximation
        all_pnls = [t.pnl for t in self.trade_memory if f"{t.symbol}_{t.strategy_name}" == key]
        if len(all_pnls) > 10:
            perf.sharpe = np.mean(all_pnls) / (np.std(all_pnls) + 1e-6) * np.sqrt(252)
        
        perf.last_updated = datetime.utcnow().isoformat()
    
    async def _analyze_gaps(self, symbol: str):
        """Analyze performance gaps for a symbol."""
        # Get all strategies for this symbol
        symbol_strategies = {
            k: v for k, v in self.strategy_performance.items()
            if v.symbol == symbol and v.total_trades >= 30
        }
        
        if not symbol_strategies:
            return
        
        gaps = []
        
        for key, perf in symbol_strategies.items():
            # Gap 1: Low winrate in certain conditions
            if perf.winrate < 50:
                gaps.append({
                    "type": "low_winrate",
                    "strategy": perf.name,
                    "value": perf.winrate,
                    "severity": "high" if perf.winrate < 40 else "medium",
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            # Gap 2: High drawdown
            if perf.max_drawdown > 15:
                gaps.append({
                    "type": "high_drawdown",
                    "strategy": perf.name,
                    "value": perf.max_drawdown,
                    "severity": "high" if perf.max_drawdown > 25 else "medium",
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            # Gap 3: Negative expectancy
            if perf.expectancy <= 0:
                gaps.append({
                    "type": "negative_expectancy",
                    "strategy": perf.name,
                    "value": perf.expectancy,
                    "severity": "critical",
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            # Gap 4: Time-based performance (session analysis)
            session_gaps = await self._analyze_session_performance(symbol, perf.name)
            gaps.extend(session_gaps)
        
        self.gap_analysis[symbol] = gaps[-50:]  # Keep last 50 gaps
    
    async def _analyze_session_performance(self, symbol: str, strategy: str) -> List[Dict]:
        """Analyze performance by trading session."""
        gaps = []
        
        # Get trades for this strategy/symbol
        trades = [t for t in self.trade_memory if t.symbol == symbol and t.strategy_name == strategy]
        
        if len(trades) < 20:
            return gaps
        
        # Group by session
        sessions = {
            "asian": (0, 8),
            "london": (8, 16),
            "ny": (13, 21),
            "overlap": (12, 16)  # London/NY overlap
        }
        
        for session_name, (start, end) in sessions.items():
            session_trades = [
                t for t in trades
                if start <= datetime.fromisoformat(t.timestamp).hour < end
            ]
            
            if len(session_trades) >= 10:
                winrate = sum(1 for t in session_trades if t.pnl > 0) / len(session_trades) * 100
                expectancy = np.mean([t.pnl for t in session_trades])
                
                if winrate < 45 or expectancy < 0:
                    gaps.append({
                        "type": f"session_weakness_{session_name}",
                        "strategy": strategy,
                        "session": session_name,
                        "winrate": winrate,
                        "expectancy": expectancy,
                        "trades": len(session_trades),
                        "severity": "high" if expectancy < 0 else "medium",
                        "timestamp": datetime.utcnow().isoformat()
                    })
        
        return gaps
    
    async def step(self, closed_trades: List[Any]):
        """Periodic evolution step."""
        if not self.enabled:
            return
        
        # Check for strategies that should be killed
        await self._cull_weak_strategies()
        
        # Generate new strategy variants for gaps
        await self._fill_gaps()
        
        # Save memory
        await self.save_memory()
    
    async def _cull_weak_strategies(self):
        """Kill strategies that don't meet survival threshold."""
        to_remove = []
        
        for key, perf in self.strategy_performance.items():
            # Must have minimum trades and positive expectancy
            if perf.total_trades >= 50:
                if perf.expectancy <= 0 or perf.winrate < 45:
                    print(f"EVOLUTION: Culling {key} - expectancy={perf.expectancy:.4f}, winrate={perf.winrate:.1f}%")
                    to_remove.append(key)
        
        for key in to_remove:
            del self.strategy_performance[key]
    
    async def _fill_gaps(self):
        """Generate new strategy variants to fill identified gaps."""
        # This is where we'd create new strategy combinations
        # For now, log the gaps for manual review
        for symbol, gaps in self.gap_analysis.items():
            critical_gaps = [g for g in gaps if g.get("severity") == "critical"]
            if critical_gaps:
                print(f"EVOLUTION: Critical gaps for {symbol}: {len(critical_gaps)}")
                for g in critical_gaps[:3]:
                    print(f"  - {g['type']} in {g['strategy']}: {g.get('value', 'N/A')}")
    
    def get_best_strategy(self, symbol: str) -> Optional[str]:
        """Get the best performing strategy for a symbol."""
        symbol_strategies = {
            k: v for k, v in self.strategy_performance.items()
            if v.symbol == symbol and v.total_trades >= 30
        }
        
        if not symbol_strategies:
            return None
        
        # Rank by expectancy * winrate (survival fitness)
        best = max(symbol_strategies.values(), key=lambda p: p.expectancy * p.winrate / 100)
        return best.name
    
    def get_evolution_report(self) -> Dict:
        """Get evolution status report."""
        return {
            "total_trades_learned": len(self.trade_memory),
            "strategies_tracked": len(self.strategy_performance),
            "symbols_with_gaps": len(self.gap_analysis),
            "survival_threshold": self.survival_threshold,
            "learning_rate": self.learning_rate,
            "top_strategies": self._get_top_strategies()
        }
    
    def _get_top_strategies(self) -> List[Dict]:
        """Get top 5 strategies by fitness."""
        strategies = [
            {
                "key": k,
                "symbol": v.symbol,
                "strategy": v.name,
                "winrate": v.winrate,
                "expectancy": v.expectancy,
                "profit_factor": v.profit_factor,
                "trades": v.total_trades,
                "fitness": v.expectancy * v.winrate / 100
            }
            for k, v in self.strategy_performance.items()
            if v.total_trades >= 30
        ]
        
        strategies.sort(key=lambda x: x["fitness"], reverse=True)
        return strategies[:5]