"""
Trade Logger Module for WILLAM
Logs all trades to CSV/JSON with pair and strategy separation
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class OpenTradeLog:
    ticket: int
    symbol: str
    direction: str
    entry_price: float
    sl: float
    tp: float
    lot_size: float
    strategy_name: str
    open_time: str
    confidence: float
    reasoning: str
    news_context: Optional[Dict] = None


@dataclass
class ClosedTradeLog:
    ticket: int
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    sl: float
    tp: float
    lot_size: float
    strategy_name: str
    open_time: str
    close_time: str
    duration_minutes: float
    pnl: float
    close_reason: str
    confidence: float
    reasoning: str
    news_context: Optional[Dict] = None


class TradeLogger:
    def __init__(self, config: Dict):
        self.config = config
        self.log_config = config.get("logging", {})
        
        # Paths
        self.trade_log_dir = Path(self.log_config.get("trade_log_path", "logs/trades"))
        self.backtest_log_dir = Path(self.log_config.get("backtest_log_path", "backtest"))
        
        self.trade_log_dir.mkdir(parents=True, exist_ok=True)
        self.backtest_log_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory for quick access
        self.open_trades: Dict[int, OpenTradeLog] = {}
        self.closed_trades: List[ClosedTradeLog] = []
        
        # CSV files (one per symbol/strategy combo)
        self.csv_files: Dict[str, Path] = {}
        self.csv_writers: Dict[str, csv.writer] = {}
        self.csv_headers_written: Dict[str, bool] = {}
    
    def log_open(self, position: Any, signal: Any):
        """Log an opened trade."""
        log = OpenTradeLog(
            ticket=position.ticket,
            symbol=position.symbol,
            direction=position.direction,
            entry_price=position.entry_price,
            sl=position.sl,
            tp=position.tp,
            lot_size=position.lot_size,
            strategy_name=position.strategy_name,
            open_time=position.open_time.isoformat(),
            confidence=signal.confidence,
            reasoning=signal.reasoning,
            news_context=signal.news_context
        )
        
        self.open_trades[position.ticket] = log
        self._write_csv(log, "open")
    
    def log_close(self, position: Any, exit_price: float, close_reason: str, pnl: float):
        """Log a closed trade."""
        open_log = self.open_trades.get(position.ticket)
        if not open_log:
            return
        
        close_log = ClosedTradeLog(
            ticket=position.ticket,
            symbol=position.symbol,
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=exit_price,
            sl=position.sl,
            tp=position.tp,
            lot_size=position.lot_size,
            strategy_name=position.strategy_name,
            open_time=open_log.open_time,
            close_time=datetime.utcnow().isoformat(),
            duration_minutes=(datetime.utcnow() - position.open_time).total_seconds() / 60,
            pnl=pnl,
            close_reason=close_reason,
            confidence=open_log.confidence,
            reasoning=open_log.reasoning,
            news_context=open_log.news_context
        )
        
        self.closed_trades.append(close_log)
        self._write_csv(close_log, "closed")
        
        # Remove from open
        del self.open_trades[position.ticket]
    
    def _write_csv(self, log: Any, log_type: str):
        """Write trade to CSV file (per symbol/strategy)."""
        key = f"{log.symbol}_{log.strategy_name}_{log_type}"
        
        if key not in self.csv_files:
            # Create CSV file
            filename = f"{log.symbol}_{log.strategy_name}_{log_type}.csv"
            filepath = self.trade_log_dir / filename
            self.csv_files[key] = filepath
            
            # Open file in append mode
            f = open(filepath, "a", newline="")
            writer = csv.writer(f)
            self.csv_writers[key] = writer
            self.csv_headers_written[key] = filepath.stat().st_size > 0
            
            # Store file handle for closing later
            if not hasattr(self, '_file_handles'):
                self._file_handles = {}
            self._file_handles[key] = f
        
        writer = self.csv_writers[key]
        
        # Write header if needed
        if not self.csv_headers_written[key]:
            if log_type == "open":
                header = [
                    "ticket", "symbol", "direction", "entry_price", "sl", "tp",
                    "lot_size", "strategy_name", "open_time", "confidence",
                    "reasoning", "news_context"
                ]
            else:
                header = [
                    "ticket", "symbol", "direction", "entry_price", "exit_price",
                    "sl", "tp", "lot_size", "strategy_name", "open_time",
                    "close_time", "duration_minutes", "pnl", "close_reason",
                    "confidence", "reasoning", "news_context"
                ]
            writer.writerow(header)
            self.csv_headers_written[key] = True
        
        # Write row
        if log_type == "open":
            row = [
                log.ticket, log.symbol, log.direction, log.entry_price, log.sl, log.tp,
                log.lot_size, log.strategy_name, log.open_time, log.confidence,
                log.reasoning, json.dumps(log.news_context) if log.news_context else ""
            ]
        else:
            row = [
                log.ticket, log.symbol, log.direction, log.entry_price, log.exit_price,
                log.sl, log.tp, log.lot_size, log.strategy_name, log.open_time,
                log.close_time, log.duration_minutes, log.pnl, log.close_reason,
                log.confidence, log.reasoning, json.dumps(log.news_context) if log.news_context else ""
            ]
        
        writer.writerow(row)
        
        # Flush to disk
        self._file_handles[key].flush()
    
    def log_backtest_result(self, result: Any):
        """Log backtest result to separate file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backtest_{result.symbol}_{result.name}_{result.timeframe}_{timestamp}.json"
        filepath = self.backtest_log_dir / filename
        
        data = {
            "symbol": result.symbol,
            "strategy": result.name,
            "timeframe": result.timeframe,
            "winrate": result.winrate,
            "expectancy": result.expectancy,
            "profit_factor": result.profit_factor,
            "max_drawdown": result.max_drawdown,
            "total_trades": result.total_trades,
            "avg_trade": result.avg_trade,
            "sharpe": result.sharpe,
            "parameters": result.parameters,
            "timestamp": datetime.utcnow().isoformat(),
            "equity_curve": result.equity_curve[-1000:] if result.equity_curve else []
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
    
    def get_symbol_stats(self, symbol: str) -> Dict:
        """Get statistics for a symbol."""
        trades = [t for t in self.closed_trades if t.symbol == symbol]
        
        if not trades:
            return {"symbol": symbol, "trades": 0}
        
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        
        return {
            "symbol": symbol,
            "total_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "winrate": len(wins) / len(trades) * 100,
            "total_pnl": sum(t.pnl for t in trades),
            "avg_win": sum(t.pnl for t in wins) / len(wins) if wins else 0,
            "avg_loss": sum(t.pnl for t in losses) / len(losses) if losses else 0,
            "profit_factor": abs(sum(t.pnl for t in wins) / sum(t.pnl for t in losses)) if losses else 999,
            "by_strategy": self._get_strategy_stats(symbol)
        }
    
    def _get_strategy_stats(self, symbol: str) -> Dict:
        """Get stats per strategy for a symbol."""
        trades = [t for t in self.closed_trades if t.symbol == symbol]
        strategies = {}
        
        for t in trades:
            if t.strategy_name not in strategies:
                strategies[t.strategy_name] = []
            strategies[t.strategy_name].append(t)
        
        result = {}
        for name, strat_trades in strategies.items():
            wins = [t for t in strat_trades if t.pnl > 0]
            losses = [t for t in strat_trades if t.pnl <= 0]
            
            result[name] = {
                "trades": len(strat_trades),
                "wins": len(wins),
                "losses": len(losses),
                "winrate": len(wins) / len(strat_trades) * 100,
                "total_pnl": sum(t.pnl for t in strat_trades),
                "avg_pnl": sum(t.pnl for t in strat_trades) / len(strat_trades)
            }
        
        return result
    
    def get_overall_stats(self) -> Dict:
        """Get overall portfolio statistics."""
        if not self.closed_trades:
            return {"total_trades": 0}
        
        wins = [t for t in self.closed_trades if t.pnl > 0]
        losses = [t for t in self.closed_trades if t.pnl <= 0]
        
        # By symbol
        symbols = {}
        for t in self.closed_trades:
            if t.symbol not in symbols:
                symbols[t.symbol] = []
            symbols[t.symbol].append(t)
        
        symbol_stats = {}
        for sym, trades in symbols.items():
            w = [x for x in trades if x.pnl > 0]
            l = [x for x in trades if x.pnl <= 0]
            symbol_stats[sym] = {
                "trades": len(trades),
                "winrate": len(w) / len(trades) * 100,
                "pnl": sum(x.pnl for x in trades)
            }
        
        return {
            "total_trades": len(self.closed_trades),
            "wins": len(wins),
            "losses": len(losses),
            "winrate": len(wins) / len(self.closed_trades) * 100,
            "total_pnl": sum(t.pnl for t in self.closed_trades),
            "avg_win": sum(t.pnl for t in wins) / len(wins) if wins else 0,
            "avg_loss": sum(t.pnl for t in losses) / len(losses) if losses else 0,
            "profit_factor": abs(sum(t.pnl for t in wins) / sum(t.pnl for t in losses)) if losses else 999,
            "by_symbol": symbol_stats
        }
    
    def close(self):
        """Close all file handles."""
        if hasattr(self, '_file_handles'):
            for f in self._file_handles.values():
                f.close()