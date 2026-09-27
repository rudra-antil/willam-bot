"""
WILLAM - Forex/CFD Trading Bot
Fresh implementation - no legacy dependencies
"""

import asyncio
import yaml
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json

# Core imports
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None

# Local modules
import sys
sys.path.append(str(Path(__file__).parent))
from news.checker import NewsChecker
from strategies.manager import StrategyManager
from risk.manager import RiskManager
from jev.engine import JEVEngine
from evolution.engine import EvolutionEngine
from logging.trade_logger import TradeLogger


class BotState(Enum):
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED_NEWS = "paused_news"
    WAITING_CONFIRMATION = "waiting_confirmation"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class TradeSignal:
    symbol: str
    direction: str  # "buy" or "sell"
    entry_price: float
    sl: float
    tp: float
    lot_size: float
    strategy_name: str
    confidence: float
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    news_context: Optional[Dict] = None


@dataclass
class Position:
    ticket: int
    symbol: str
    direction: str
    entry_price: float
    sl: float
    tp: float
    lot_size: float
    strategy_name: str
    open_time: datetime
    current_pnl: float = 0.0


class WillamBot:
    """
    Main WILLAM trading bot orchestrator.
    
    Flow:
    1. Check news (1hr lookahead)
    2. If high-impact news -> pause, wait for release
    3. Analyze actual vs expected -> surprise = trade data direction
    4. If expected -> wait for trend confirmation
    5. If no news -> run validated strategies
    6. JEV makes final decision with risk management
    7. Execute, log, evolve
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.state = BotState.INITIALIZING
        self.running = False
        
        # Core components
        self.news_checker = NewsChecker(self.config)
        self.strategy_manager = StrategyManager(self.config)
        self.risk_manager = RiskManager(self.config)
        self.jev = JEVEngine(self.config)
        self.evolution = EvolutionEngine(self.config)
        self.trade_logger = TradeLogger(self.config)
        
        # Runtime state
        self.positions: Dict[int, Position] = {}
        self.pending_signals: List[TradeSignal] = []
        self.last_news_check = datetime.utcnow()
        self.symbol_strategies: Dict[str, str] = {}  # symbol -> best strategy
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger("willam")
        
    def _load_config(self) -> Dict:
        with open(self.config_path) as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self):
        log_dir = Path(self.config["logging"]["trade_log_path"])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, self.config["logging"]["level"]),
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            handlers=[
                logging.FileHandler(log_dir / f"willam_{datetime.now():%Y%m%d}.log"),
                logging.StreamHandler()
            ]
        )
    
    async def initialize(self) -> bool:
        """Initialize MT5 connection and load strategies."""
        self.logger.info("=" * 60)
        self.logger.info(f"WILLAM v{self.config['bot']['version']} INITIALIZING")
        self.logger.info("=" * 60)
        
        # Connect MT5
        if not await self._connect_mt5():
            self.state = BotState.ERROR
            return False
        
        # Load/backtest strategies for each symbol
        await self._initialize_strategies()
        
        # Load evolution memory
        await self.evolution.load_memory()
        
        self.state = BotState.RUNNING
        self.running = True
        self.logger.info("WILLAM READY - Starting main loop")
        return True
    
    async def _connect_mt5(self) -> bool:
        if not MT5_AVAILABLE:
            self.logger.error("MetaTrader5 package not installed")
            return False
        
        cfg = self.config["mt5"]
        if not mt5.initialize(
            path=cfg.get("path"),
            login=cfg.get("login"),
            password=cfg.get("password"),
            server=cfg.get("server"),
            timeout=cfg.get("timeout", 60000)
        ):
            err = mt5.last_error()
            self.logger.error(f"MT5 init failed: {err}")
            return False
        
        acc = mt5.account_info()
        if acc:
            self.logger.info(f"MT5 Connected: {acc.login} | {acc.server} | Balance: {acc.balance:.2f} {acc.currency}")
            return True
        return False
    
    async def _initialize_strategies(self):
        """Backtest and select best strategy per symbol."""
        self.logger.info("Initializing strategies for all symbols...")
        
        for symbol in self.config["trading"]["symbols"]:
            self.logger.info(f"  Backtesting {symbol}...")
            best_strat = await self.strategy_manager.find_best_strategy(symbol)
            if best_strat:
                self.symbol_strategies[symbol] = best_strat
                self.logger.info(f"    -> Selected: {best_strat}")
            else:
                self.logger.warning(f"    -> No valid strategy found for {symbol}")
    
    async def run(self):
        """Main trading loop."""
        if not await self.initialize():
            return
        
        try:
            while self.running:
                await self._main_loop_iteration()
                await asyncio.sleep(30)  # Check every 30 seconds
        except KeyboardInterrupt:
            self.logger.info("Shutdown signal received")
        except Exception as e:
            self.logger.exception(f"Fatal error: {e}")
            self.state = BotState.ERROR
        finally:
            await self.shutdown()
    
    async def _main_loop_iteration(self):
        """Single iteration of the main loop."""
        now = datetime.utcnow()
        
        # 1. Manage existing positions
        await self._manage_positions()
        
        # 2. Check news (every 15 min)
        if (now - self.last_news_check).total_seconds() >= 900:
            await self._check_news_and_decide()
            self.last_news_check = now
        
        # 3. Process pending signals
        await self._process_pending_signals()
        
        # 4. Evolution step (periodic)
        if len(self.trade_logger.closed_trades) % 10 == 0:
            await self.evolution.step(self.trade_logger.closed_trades)
    
    async def _check_news_and_decide(self):
        """Check for upcoming high-impact news and decide action."""
        news_events = await self.news_checker.get_upcoming_news(
            hours_ahead=self.config["news"]["lookahead_hours"]
        )
        
        high_impact = [n for n in news_events if n.impact == "high"]
        
        if high_impact:
            self.logger.warning(f"HIGH IMPACT NEWS DETECTED: {len(high_impact)} events")
            self.state = BotState.PAUSED_NEWS
            
            for event in high_impact:
                await self._handle_news_event(event)
        else:
            self.logger.info("No high-impact news in next hour - running strategies")
            await self._run_strategies()
    
    async def _handle_news_event(self, event):
        """Handle a single news event: wait, analyze, decide."""
        self.logger.info(f"Waiting for news: {event.name} at {event.time}")
        
        # Pause trading until news release
        await self._wait_until(event.time)
        
        # Get actual vs forecast
        actual = await self.news_checker.get_actual_value(event)
        forecast = event.forecast
        previous = event.previous
        
        self.logger.info(f"News: {event.name} | Actual: {actual} | Forecast: {forecast} | Prev: {previous}")
        
        # Determine surprise
        surprise = self._analyze_surprise(actual, forecast, previous)
        
        if surprise == "expected":
            # Wait for trend confirmation
            self.state = BotState.WAITING_CONFIRMATION
            await self._wait_for_trend_confirmation(event.symbol)
        elif surprise == "positive":
            # Bullish surprise - go long
            await self._generate_news_signal(event.symbol, "buy", "bullish_surprise", actual, forecast)
        elif surprise == "negative":
            # Bearish surprise - go short
            await self._generate_news_signal(event.symbol, "sell", "bearish_surprise", actual, forecast)
    
    def _analyze_surprise(self, actual: float, forecast: float, previous: float) -> str:
        """Classify news surprise."""
        if actual is None or forecast is None:
            return "unknown"
        
        diff = actual - forecast
        threshold = abs(forecast * 0.05) if forecast != 0 else 0.001
        
        if abs(diff) <= threshold:
            return "expected"
        elif diff > 0:
            return "positive"
        else:
            return "negative"
    
    async def _wait_for_trend_confirmation(self, symbol: str):
        """Wait for price action to confirm trend continuation."""
        self.logger.info(f"Waiting for trend confirmation on {symbol}...")
        
        wait_minutes = self.config["news"]["confirmation_wait_minutes"]
        start = datetime.utcnow()
        
        while (datetime.utcnow() - start).total_seconds() < wait_minutes * 60:
            # Check if trend is continuing
            trend = await self.strategy_manager.get_current_trend(symbol)
            if trend.confirmed:
                self.logger.info(f"Trend confirmed on {symbol}: {trend.direction}")
                await self._run_strategies()
                return
            await asyncio.sleep(60)
        
        self.logger.warning(f"No trend confirmation on {symbol} - resuming normal")
        self.state = BotState.RUNNING
    
    async def _generate_news_signal(self, symbol: str, direction: str, reason: str, actual: float, forecast: float):
        """Generate trade signal from news surprise."""
        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return
        
        entry = tick.ask if direction == "buy" else tick.bid
        
        # Calculate SL/TP via risk manager
        sl, tp, lot = self.risk_manager.calculate_levels(
            symbol, direction, entry, self.config["trading"]["timeframe"]
        )
        
        signal = TradeSignal(
            symbol=symbol,
            direction=direction,
            entry_price=entry,
            sl=sl,
            tp=tp,
            lot_size=lot,
            strategy_name=f"news_{reason}",
            confidence=0.85,
            reasoning=f"News surprise: actual={actual}, forecast={forecast}",
            news_context={"actual": actual, "forecast": forecast, "reason": reason}
        )
        
        self.pending_signals.append(signal)
        self.logger.info(f"NEWS SIGNAL: {symbol} {direction} @ {entry:.5f} SL:{sl:.5f} TP:{tp:.5f} Lot:{lot:.2f}")
    
    async def _run_strategies(self):
        """Run validated strategies for all symbols (no news)."""
        if self.state != BotState.RUNNING:
            return
        
        for symbol in self.config["trading"]["symbols"]:
            if symbol not in self.symbol_strategies:
                continue
            
            strategy_name = self.symbol_strategies[symbol]
            signal = await self.strategy_manager.generate_signal(symbol, strategy_name)
            
            if signal and self.jev.validate(signal, self.positions):
                self.pending_signals.append(signal)
    
    async def _process_pending_signals(self):
        """Execute validated pending signals."""
        for signal in self.pending_signals[:]:
            if self.risk_manager.can_open_position(signal.symbol, self.positions):
                success = await self._execute_signal(signal)
                if success:
                    self.pending_signals.remove(signal)
            else:
                self.logger.warning(f"Risk check failed for {signal.symbol} - skipping")
                self.pending_signals.remove(signal)
    
    async def _execute_signal(self, signal: TradeSignal) -> bool:
        """Execute trade via MT5."""
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": signal.symbol,
            "volume": signal.lot_size,
            "type": mt5.ORDER_TYPE_BUY if signal.direction == "buy" else mt5.ORDER_TYPE_SELL,
            "price": signal.entry_price,
            "sl": signal.sl,
            "tp": signal.tp,
            "deviation": 20,
            "magic": 20240927,  # WILLAM magic number
            "comment": f"WILLAM_{signal.strategy_name}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            pos = Position(
                ticket=result.order,
                symbol=signal.symbol,
                direction=signal.direction,
                entry_price=signal.entry_price,
                sl=signal.sl,
                tp=signal.tp,
                lot_size=signal.lot_size,
                strategy_name=signal.strategy_name,
                open_time=datetime.utcnow()
            )
            self.positions[result.order] = pos
            self.trade_logger.log_open(pos, signal)
            self.logger.info(f"EXECUTED: {signal.symbol} {signal.direction} Ticket:{result.order}")
            return True
        else:
            self.logger.error(f"Order failed: {result.retcode} - {result.comment}")
            return False
    
    async def _manage_positions(self):
        """Monitor open positions, update PnL, handle SL/TP hits."""
        for ticket, pos in list(self.positions.items()):
            # Get current position from MT5
            mt5_pos = mt5.positions_get(ticket=ticket)
            if not mt5_pos:
                # Position closed - log and learn
                await self._handle_closed_position(ticket, pos)
                continue
            
            mt5_pos = mt5_pos[0]
            pos.current_pnl = mt5_pos.profit
            
            # Check if SL/TP hit (MT5 handles this, but we track for logging)
            # Could add trailing stop logic here
    
    async def _handle_closed_position(self, ticket: int, pos: Position):
        """Handle closed position - log, analyze, evolve."""
        # Get deal history for exact close price
        deals = mt5.history_deals_get(ticket=ticket)
        close_price = pos.entry_price
        close_reason = "unknown"
        
        if deals:
            for deal in deals:
                if deal.entry == mt5.DEAL_ENTRY_OUT:
                    close_price = deal.price
                    if deal.reason == mt5.DEAL_REASON_SL:
                        close_reason = "sl"
                    elif deal.reason == mt5.DEAL_REASON_TP:
                        close_reason = "tp"
                    elif deal.reason == mt5.DEAL_REASON_EXPERT:
                        close_reason = "manual"
        
        pnl = (close_price - pos.entry_price) * pos.lot_size * 100000  # Approx
        if pos.direction == "sell":
            pnl = -pnl
        
        self.trade_logger.log_close(pos, close_price, close_reason, pnl)
        self.logger.info(f"CLOSED: {pos.symbol} {pos.direction} PnL:{pnl:.2f} Reason:{close_reason}")
        
        # Evolution step
        await self.evolution.learn_from_trade(pos, close_price, close_reason, pnl)
        
        del self.positions[ticket]
    
    async def _wait_until(self, target_time: datetime):
        """Wait until target time."""
        while datetime.utcnow() < target_time:
            await asyncio.sleep(10)
    
    async def shutdown(self):
        """Graceful shutdown."""
        self.logger.info("Shutting down WILLAM...")
        self.running = False
        
        # Save evolution memory
        await self.evolution.save_memory()
        
        # Close MT5
        if MT5_AVAILABLE:
            mt5.shutdown()
        
        self.logger.info("WILLAM stopped")


async def main():
    bot = WillamBot()
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())