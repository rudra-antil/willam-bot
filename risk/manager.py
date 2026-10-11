"""
Risk Manager Module for WILLAM — v2
Handles position sizing, stop loss, take profit, and risk limits.

v2 fixes:
- timedelta imported properly (was at file bottom).
- None-safe account reads (MT5 disconnect no longer crashes the loop).
- estimate_pnl() fallback for symbols where deal history is unavailable.
- Spread sanity check before allowing new entries.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import numpy as np

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None

logger = logging.getLogger("willam.risk")


@dataclass
class RiskParams:
    risk_percent: float
    max_risk_percent: float
    max_daily_drawdown: float
    max_positions_per_symbol: int
    max_total_positions: int
    risk_reward_ratio: float
    atr_multiplier_sl: float
    atr_multiplier_tp: float
    use_atr_stops: bool


class RiskManager:
    def __init__(self, config: Dict):
        self.config = config
        self.risk_config = config.get("risk", {})
        self.trading_config = config.get("trading", {})

        self.params = RiskParams(
            risk_percent=self.risk_config.get("default_risk_percent", 1.0),
            max_risk_percent=self.risk_config.get("max_risk_percent", 2.0),
            max_daily_drawdown=self.risk_config.get("max_daily_drawdown", 5.0),
            max_positions_per_symbol=self.risk_config.get("max_positions_per_symbol", 1),
            max_total_positions=self.risk_config.get("max_total_positions", 5),
            risk_reward_ratio=self.risk_config.get("risk_reward_ratio", 2.0),
            atr_multiplier_sl=self.risk_config.get("atr_multiplier_sl", 1.5),
            atr_multiplier_tp=self.risk_config.get("atr_multiplier_tp", 3.0),
            use_atr_stops=self.risk_config.get("use_atr_stops", True),
        )

        self.daily_pnl = 0.0
        self.daily_reset_time = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        self._halted_for_today = False

    # ------------------------------------------------------------------
    # SL / TP / lots
    # ------------------------------------------------------------------

    def calculate_levels(self, symbol: str, direction: str, entry_price: float, timeframe: str) -> tuple:
        """Returns (sl, tp, lot_size)."""
        if not MT5_AVAILABLE:
            raise RuntimeError("MT5 unavailable")
        info = mt5.symbol_info(symbol)
        if not info:
            raise ValueError(f"Symbol {symbol} not found")

        atr = self._get_atr(symbol, timeframe)

        if self.params.use_atr_stops and atr > 0:
            sl_distance = atr * self.params.atr_multiplier_sl
        else:
            sl_distance = info.point * 200

        min_dist = info.trade_stops_level * info.point
        if sl_distance < min_dist:
            sl_distance = max(min_dist * 1.5, info.point * 10)

        if direction == "buy":
            sl = entry_price - sl_distance
            tp = entry_price + sl_distance * self.params.risk_reward_ratio
        else:
            sl = entry_price + sl_distance
            tp = entry_price - sl_distance * self.params.risk_reward_ratio

        sl = round(sl, info.digits)
        tp = round(tp, info.digits)

        lot_size = self._calculate_lot_size(symbol, entry_price, sl)
        return sl, tp, lot_size

    def _get_atr(self, symbol: str, timeframe: str, period: int = 14) -> float:
        if not MT5_AVAILABLE:
            return 0.0
        try:
            timeframe_map = {
                "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
                "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1,
            }
            tf = timeframe_map.get(timeframe, mt5.TIMEFRAME_M15)
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, period + 10)
            if rates is None or len(rates) < period:
                return 0.0

            import pandas as pd
            df = pd.DataFrame(rates)
            high_low = df["high"] - df["low"]
            high_close = (df["high"] - df["close"].shift()).abs()
            low_close = (df["low"] - df["close"].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(period).mean().iloc[-1]
            return float(atr) if not np.isnan(atr) else 0.0
        except Exception:
            return 0.0

    def _calculate_lot_size(self, symbol: str, entry_price: float, sl: float) -> float:
        if not MT5_AVAILABLE:
            return 0.01
        account = mt5.account_info()
        if not account:
            logger.warning("account_info() None during lot sizing — defaulting to min lot")
            return 0.01

        balance = account.balance
        risk_amount = balance * (self.params.risk_percent / 100)

        info = mt5.symbol_info(symbol)
        if not info or info.trade_tick_value == 0 or info.trade_tick_size == 0 or info.point == 0:
            return 0.01

        points_diff = abs(entry_price - sl) / info.point
        value_per_lot = points_diff * info.trade_tick_value
        if value_per_lot <= 0:
            return 0.01

        lots = risk_amount / value_per_lot

        step = info.volume_step or 0.01
        lots = round(lots / step) * step

        lots = max(info.volume_min, min(lots, info.volume_max))

        # Cap at max risk percent
        max_risk_amount = balance * (self.params.max_risk_percent / 100)
        max_lots = max_risk_amount / value_per_lot
        lots = min(lots, max_lots)

        # Clamp again after capping (step alignment)
        lots = max(info.volume_min, min(lots, info.volume_max))
        lots = round(lots / step) * step
        return max(lots, info.volume_min)

    # ------------------------------------------------------------------
    # Permissions
    # ------------------------------------------------------------------

    def can_open_position(self, symbol: str, positions: Dict) -> bool:
        if not MT5_AVAILABLE:
            return False
        self._check_daily_reset()
        account = mt5.account_info()
        if not account:
            logger.warning("account_info() None — refusing new positions")
            return False

        # Daily drawdown halt (sticky for the rest of the UTC day)
        dd_limit = abs(self.params.max_daily_drawdown / 100 * account.balance)
        if self._halted_for_today or self.daily_pnl <= -dd_limit:
            if not self._halted_for_today:
                logger.warning(f"DAILY DRAWDOWN LIMIT HIT ({self.daily_pnl:.2f} <= -{dd_limit:.2f}) — halting for the day")
                self._halted_for_today = True
            return False

        # Per-symbol cap
        symbol_positions = sum(1 for p in positions.values() if p.symbol == symbol)
        if symbol_positions >= self.params.max_positions_per_symbol:
            return False

        # Total cap
        if len(positions) >= self.params.max_total_positions:
            return False

        # Spread sanity — refuse entries at blown-out spreads
        info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if info and tick and info.point and info.spread:
            spread = (tick.ask - tick.bid) / info.point
            if spread > max(info.spread * 3, 30):  # spread 3x average or >30 points
                logger.info(f"Spread too wide on {symbol}: {spread:.1f}")
                return False

        # Margin
        if account.margin_free < account.margin * 0.5:
            return False

        return True

    def _check_daily_reset(self):
        now = datetime.now(timezone.utc)
        if now >= self.daily_reset_time + timedelta(days=1):
            self.daily_pnl = 0.0
            self._halted_for_today = False
            self.daily_reset_time = now.replace(hour=0, minute=0, second=0, microsecond=0)

    def update_daily_pnl(self, pnl: float):
        self._check_daily_reset()
        self.daily_pnl += pnl

    def get_account_risk_status(self) -> Dict:
        if not MT5_AVAILABLE:
            return {}
        account = mt5.account_info()
        if not account:
            return {}
        return {
            "balance": account.balance,
            "equity": account.equity,
            "margin_used": account.margin,
            "margin_free": account.margin_free,
            "margin_level": account.margin_level,
            "daily_pnl": self.daily_pnl,
            "daily_pnl_percent": (self.daily_pnl / account.balance * 100) if account.balance > 0 else 0,
            "halted_for_today": self._halted_for_today,
            "max_daily_drawdown": self.params.max_daily_drawdown,
            "open_positions": len(mt5.positions_get() or []),
            "max_positions": self.params.max_total_positions,
        }

    # ------------------------------------------------------------------
    # PnL fallback + signal validation
    # ------------------------------------------------------------------

    def estimate_pnl(self, pos: Any, close_price: float) -> float:
        """
        Tick-value based PnL estimate, correct for ANY symbol class
        (JPY pairs, gold, indices) — used only when deal history is unavailable.
        """
        if not MT5_AVAILABLE:
            return 0.0
        info = mt5.symbol_info(pos.symbol)
        if not info or info.trade_tick_size == 0 or info.trade_tick_value == 0:
            return 0.0
        ticks = abs(close_price - pos.entry_price) / info.trade_tick_size
        gross = ticks * info.trade_tick_value * pos.lot_size
        return gross if pos.direction == "buy" else -gross

    def validate_signal(self, signal: Any, positions: Dict) -> bool:
        if not self.can_open_position(signal.symbol, positions):
            return False
        if signal.sl <= 0 or signal.tp <= 0:
            return False
        if signal.direction == "buy":
            risk = signal.entry_price - signal.sl
            reward = signal.tp - signal.entry_price
        else:
            risk = signal.sl - signal.entry_price
            reward = signal.entry_price - signal.tp
        if risk <= 0:
            return False
        if reward / risk < self.params.risk_reward_ratio * 0.8:
            return False
        return True
