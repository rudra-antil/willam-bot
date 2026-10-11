"""
WILLAM - Forex/CFD Trading Bot — v2
Fresh implementation - no legacy dependencies

v2 upgrades:
- Non-blocking news handling: pausing for one pair's news never stops the others.
- Correct PnL from MT5 deal records (works for JPY, gold, indices — not just USD majors).
- Broker-adaptive order filling mode (IOC/FOK/RETURN) — no more 10030 rejections.
- JEV is a real LLM decision engine (nemotron -> free-llm-api -> rules fallback).
- News context flows signal -> position -> evolution so news trades learn properly.
"""

import asyncio
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml

# Local modules (signals.py holds shared TradeSignal/Position — no circular imports)
sys.path.append(str(Path(__file__).parent))
from signals import TradeSignal, Position
from news.checker import NewsChecker
from strategies.manager import StrategyManager
from risk.manager import RiskManager
from jev.engine import JEVEngine
from evolution.engine import EvolutionEngine
from loggers.trade_logger import TradeLogger

# Core imports
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None


class BotState(Enum):
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED_NEWS = "paused_news"          # kept for reporting; per-symbol windows now
    WAITING_CONFIRMATION = "waiting_confirmation"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class NewsWindow:
    """A live per-symbol news gate. Never blocks other symbols."""
    event: Any
    phase: str                            # "pre" (before release) | "post" (after release)
    symbol: str
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class WillamBot:
    """
    Main WILLAM trading bot orchestrator.

    Loop (every cycle, non-blocking):
    1. Manage existing positions (detect closes, learn).
    2. Update per-symbol news windows (pre-release block / post-release surprise logic).
    3. If a window is in 'post' and surprise known -> possibly generate news signal.
    4. For every symbol NOT gated by news -> run its validated strategy.
    5. JEV validates each candidate signal (LLM or fallback).
    6. Risk manager sizes; execute; log; evolution learns from every close.
    """

    MAGIC = 20240927  # WILLAM magic number

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
        self.news_windows: Dict[str, NewsWindow] = {}   # symbol -> active window
        self.symbol_strategies: Dict[str, str] = {}
        self.last_news_check = datetime.min.replace(tzinfo=timezone.utc)
        self.last_evolution_step_trade_count = -1
        self._filling_cache: Dict[str, int] = {}        # symbol -> working filling mode

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
                logging.StreamHandler(),
            ],
        )

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    async def initialize(self) -> bool:
        self.logger.info("=" * 60)
        self.logger.info(f"WILLAM v{self.config['bot']['version']} INITIALIZING")
        self.logger.info("=" * 60)

        if not await self._connect_mt5():
            self.state = BotState.ERROR
            return False

        await self._initialize_strategies()
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
        kwargs = {"timeout": cfg.get("timeout", 60000)}
        if cfg.get("path"):
            kwargs["path"] = cfg["path"]
        if cfg.get("login"):
            kwargs.update(login=cfg["login"], password=cfg.get("password", ""),
                          server=cfg.get("server", ""))

        if not mt5.initialize(**kwargs):
            self.logger.error(f"MT5 init failed: {mt5.last_error()}")
            return False

        acc = mt5.account_info()
        if acc:
            self.logger.info(f"MT5 Connected: {acc.login} | {acc.server} | Balance: {acc.balance:.2f} {acc.currency}")
            # Enable all symbols in Market Watch
            for sym in self.config["trading"]["symbols"]:
                mt5.symbol_select(sym, True)
            return True
        self.logger.error("MT5 connected but account_info() returned None")
        return False

    async def _initialize_strategies(self):
        self.logger.info("Initializing strategies for all symbols...")
        for symbol in self.config["trading"]["symbols"]:
            self.logger.info(f"  Backtesting {symbol}...")
            best = await self.strategy_manager.find_best_strategy(symbol)
            if best:
                self.symbol_strategies[symbol] = best
                self.logger.info(f"    -> Selected: {best}")
            else:
                self.logger.warning(f"    -> No valid strategy found for {symbol}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self):
        if not await self.initialize():
            return
        try:
            while self.running:
                await self._main_loop_iteration()
                await asyncio.sleep(30)
        except KeyboardInterrupt:
            self.logger.info("Shutdown signal received")
        except Exception as e:
            self.logger.exception(f"Fatal error: {e}")
            self.state = BotState.ERROR
        finally:
            await self.shutdown()

    async def _main_loop_iteration(self):
        now = datetime.now(timezone.utc)

        # 1. Manage positions (non-blocking) — detects closes, learns
        await self._manage_positions()

        # 2. Refresh news calendar (every 15 min)
        if (now - self.last_news_check).total_seconds() >= 900:
            await self._refresh_news_windows()
            self.last_news_check = now

        # 3. Tick news windows: expire, transition pre->post, fire surprise logic
        await self._tick_news_windows()

        # 4. Run strategies for all symbols not gated by news
        await self._run_strategies()

        # 5. Execute pending signals
        await self._process_pending_signals()

        # 6. Evolution step — only when new trades closed since last step
        closed_count = len(self.trade_logger.closed_trades)
        if closed_count > 0 and closed_count != self.last_evolution_step_trade_count and closed_count % 5 == 0:
            await self.evolution.step(self.trade_logger.closed_trades)
            self.last_evolution_step_trade_count = closed_count

        # 7. Persist JEV decisions each cycle
        self.jev.save_decisions()

    # ------------------------------------------------------------------
    # News windows — non-blocking per-symbol gates
    # ------------------------------------------------------------------

    async def _refresh_news_windows(self):
        """Create/refresh per-symbol news gates from the calendar."""
        try:
            upcoming = await self.news_checker.get_upcoming_news(
                hours_ahead=max(self.config["news"]["lookahead_hours"], 1)
            )
        except Exception as e:
            self.logger.warning(f"News refresh failed: {e}")
            return

        for event in upcoming:
            sym = event.symbol
            existing = self.news_windows.get(sym)
            if existing is None or existing.event.name != event.name:
                self.news_windows[sym] = NewsWindow(event=event, phase="pre", symbol=sym)
                self.logger.warning(f"NEWS GATE [{sym}]: '{event.name}' at {event.time:%H:%M} UTC — new entries paused")

    async def _tick_news_windows(self):
        """Advance windows; when a pre-window's event releases, move to post and analyze surprise."""
        now = datetime.now(timezone.utc)
        wait_after = timedelta(minutes=self.config["news"]["wait_after_minutes"])

        for sym, window in list(self.news_windows.items()):
            event = window.event

            if window.phase == "pre":
                if now >= event.time:
                    window.phase = "post"
                    window.started_at = now
                    self.logger.info(f"NEWS RELEASED [{sym}]: {event.name} — analyzing surprise")
                    asyncio.create_task(self._process_news_release(sym, window))
                # else: still pre-release; gate remains

            elif window.phase == "post":
                if now - window.started_at > wait_after:
                    del self.news_windows[sym]
                    self.logger.info(f"NEWS WINDOW CLOSED [{sym}] — normal trading resumed")

    async def _process_news_release(self, symbol: str, window: NewsWindow):
        """Fetch actual value, classify surprise, generate signal if surprising."""
        event = window.event
        actual = await self.news_checker.get_actual_value(event)
        forecast, previous = event.forecast, event.previous

        self.logger.info(f"News: {event.name} | Actual: {actual} | Forecast: {forecast} | Prev: {previous}")

        if actual is None or forecast is None:
            self.logger.info(f"No surprise data for {event.name} — resuming strategies after wait")
            return

        surprise = self._analyze_surprise(actual, forecast, previous)

        if surprise == "expected":
            # As per spec: follow the pre-existing trend, but only after confirmation
            self.state = BotState.WAITING_CONFIRMATION
            confirmed = await self._wait_for_trend_confirmation(symbol)
            if confirmed:
                await self._run_strategies()
        elif surprise == "positive":
            await self._generate_news_signal(symbol, "buy", "bullish_surprise", actual, forecast, event)
        elif surprise == "negative":
            await self._generate_news_signal(symbol, "sell", "bearish_surprise", actual, forecast, event)
        # "unknown" -> do nothing; window expiry resumes normal flow

    def _analyze_surprise(self, actual: float, forecast: float, previous: Optional[float]) -> str:
        if actual is None or forecast is None:
            return "unknown"
        diff = actual - forecast
        threshold = abs(forecast * 0.05) if forecast != 0 else abs(forecast) * 0.05 or 0.001
        if abs(diff) <= threshold:
            news_holiday_adjusted = "expected"
            return news_holiday_adjusted
        return "positive" if diff > 0 else "negative"

    async def _wait_for_trend_confirmation(self, symbol: str) -> bool:
        """Poll for trend confirmation (bounded; never blocks other symbols)."""
        wait_minutes = self.config["news"].get("confirmation_wait_minutes", 15)
        start = datetime.now(timezone.utc)
        while (datetime.now(timezone.utc) - start).total_seconds() < wait_minutes * 60:
            try:
                trend = await self.strategy_manager.get_current_trend(symbol)
                if trend.confirmed:
                    self.logger.info(f"Trend confirmed on {symbol}: {trend.direction}")
                    self.state = BotState.RUNNING
                    return True
            except Exception:
                pass
            await asyncio.sleep(60)
        self.logger.warning(f"No trend confirmation on {symbol} — skipping news-follow entry")
        self.state = BotState.RUNNING
        return False

    async def _generate_news_signal(self, symbol: str, direction: str, reason: str,
                                    actual: float, forecast: float, event: Any):
        try:
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                return
            entry = tick.ask if direction == "buy" else tick.bid
            sl, tp, lot = self.risk_manager.calculate_levels(
                symbol, direction, entry, self.config["trading"]["timeframe"])

            signal = TradeSignal(
                symbol=symbol, direction=direction, entry_price=entry, sl=sl, tp=tp,
                lot_size=lot, strategy_name=f"news_{reason}", confidence=0.85,
                reasoning=f"News surprise: actual={actual}, forecast={forecast}",
                news_context={"actual": actual, "forecast": forecast, "reason": reason,
                              "event": event.name, "previous": event.previous},
            )
            self.pending_signals.append(signal)
            self.logger.info(f"NEWS SIGNAL: {symbol} {direction} @ {entry:.5f} SL:{sl:.5f} TP:{tp:.5f} Lot:{lot:.2f}")
        except Exception as e:
            self.logger.error(f"News signal generation failed for {symbol}: {e}")

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------

    def _symbol_gated_by_news(self, symbol: str) -> bool:
        return symbol in self.news_windows

    async def _run_strategies(self):
        if self.state not in (BotState.RUNNING,):
            return
        for symbol in self.config["trading"]["symbols"]:
            if symbol not in self.symbol_strategies:
                continue
            if self._symbol_gated_by_news(symbol):
                continue  # this pair's news window is active — skip new entries
            strategy_name = self.symbol_strategies[symbol]
            try:
                signal = await self.strategy_manager.generate_signal(symbol, strategy_name)
            except Exception as e:
                self.logger.error(f"Signal generation failed {symbol}/{strategy_name}: {e}")
                continue
            if signal and await self.jev.validate(signal, self.positions):
                self.pending_signals.append(signal)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def _process_pending_signals(self):
        for signal in self.pending_signals[:]:
            if self.risk_manager.can_open_position(signal.symbol, self.positions):
                if await self._execute_signal(signal):
                    self.pending_signals.remove(signal)
            else:
                self.logger.warning(f"Risk check failed for {signal.symbol} — skipping")
                self.pending_signals.remove(signal)

    def _pick_filling_mode(self, symbol: str) -> int:
        """Use the symbol's known-good filling mode; probe IOC->FOK->RETURN once."""
        if symbol in self._filling_cache:
            return self._filling_cache[symbol]
        modes = [mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN]
        info = mt5.symbol_info(symbol)
        if info and getattr(info, "filling_mode", 0):
            allowed = info.filling_mode  # bitmask
            for m in modes:
                if allowed & m:
                    self._filling_cache[symbol] = m
                    return m
        self._filling_cache[symbol] = modes[0]
        return modes[0]

    async def _execute_signal(self, signal: TradeSignal) -> bool:
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": signal.symbol,
            "volume": signal.lot_size,
            "type": mt5.ORDER_TYPE_BUY if signal.direction == "buy" else mt5.ORDER_TYPE_SELL,
            "price": signal.entry_price,
            "sl": signal.sl,
            "tp": signal.tp,
            "deviation": 20,
            "magic": self.MAGIC,
            "comment": f"WILLAM_{signal.strategy_name}"[:31],  # MT5 comment cap
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._pick_filling_mode(signal.symbol),
        }

        result = mt5.order_send(request)

        if result is None:
            self.logger.error(f"Order send returned None for {signal.symbol} (last_error={mt5.last_error()})")
            return False

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            pos = Position(
                ticket=result.order, symbol=signal.symbol, direction=signal.direction,
                entry_price=signal.entry_price, sl=signal.sl, tp=signal.tp,
                lot_size=signal.lot_size, strategy_name=signal.strategy_name,
                open_time=datetime.now(timezone.utc),
                news_context=signal.news_context,  # flows to evolution
            )
            self.positions[result.order] = pos
            self.trade_logger.log_open(pos, signal)
            self.logger.info(f"EXECUTED: {signal.symbol} {signal.direction} Ticket:{result.order}")
            return True
        else:
            # 10030 = unsupported filling -> invalidate cache so next attempt probes again
            if result.retcode == 10030:
                self._filling_cache.pop(signal.symbol, None)
            self.logger.error(f"Order failed: {result.retcode} - {result.comment}")
            return False

    # ------------------------------------------------------------------
    # Position management + TRUE PnL
    # ------------------------------------------------------------------

    async def _manage_positions(self):
        for ticket, pos in list(self.positions.items()):
            try:
                mt5_positions = mt5.positions_get(ticket=ticket)
            except Exception:
                continue

            if not mt5_positions:
                await self._handle_closed_position(ticket, pos)
                continue

            mt5_pos = mt5_positions[0]
            pos.current_pnl = mt5_pos.profit

    async def _handle_closed_position(self, ticket: int, pos: Position):
        """Read the actual closing deal — correct PnL for ANY symbol class."""
        close_price = pos.entry_price
        close_reason = "unknown"
        pnl = 0.0

        try:
            from_day = datetime.now(timezone.utc) - timedelta(days=7)
            deals = mt5.history_deals_get(from_day, datetime.now(timezone.utc) + timedelta(minutes=5), ticket=ticket)
            if deals:
                for deal in deals:
                    if deal.entry == mt5.DEAL_ENTRY_OUT:
                        close_price = deal.price
                        pnl = deal.profit + deal.swap + deal.commission  # actual account-currency PnL
                        if deal.reason == mt5.DEAL_REASON_SL:
                            close_reason = "sl"
                        elif deal.reason == mt5.DEAL_REASON_TP:
                            close_reason = "tp"
                        else:
                            close_reason = "manual"
        except Exception as e:
            self.logger.error(f"Deal history read failed for {ticket}: {e}")
            # Fallback to tick-based estimate only if deal history unavailable
            try:
                tick = mt5.symbol_info_tick(pos.symbol)
                if tick:
                    close_price = tick.bid if pos.direction == "buy" else tick.ask
                    pnl = self.risk_manager.estimate_pnl(pos, close_price)
                    close_reason = "unknown"
            except Exception:
                pass

        self.trade_logger.log_close(pos, close_price, close_reason, pnl)
        self.logger.info(f"CLOSED: {pos.symbol} {pos.direction} PnL:{pnl:.2f} Reason:{close_reason}")

        try:
            await self.evolution.learn_from_trade(pos, close_price, close_reason, pnl)
        except Exception as e:
            self.logger.exception(f"Evolution learning failed for {ticket}: {e}")

        self.risk_manager.update_daily_pnl(pnl)
        self.positions.pop(ticket, None)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def shutdown(self):
        self.logger.info("Shutting down WILLAM...")
        self.running = False
        try:
            await self.evolution.save_memory()
        except Exception as e:
            self.logger.error(f"Evolution save failed: {e}")
        self.jev.save_decisions()
        if MT5_AVAILABLE:
            mt5.shutdown()
        self.logger.info("WILLAM stopped")


async def main():
    bot = WillamBot()
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
