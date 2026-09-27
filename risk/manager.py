"""
Risk Manager Module for WILLAM
Handles position sizing, stop loss, take profit, and risk limits
"""

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import MetaTrader5 as mt5
import numpy as np


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
            use_atr_stops=self.risk_config.get("use_atr_stops", True)
        )
        
        self.daily_pnl = 0.0
        self.daily_reset_time = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    
    def calculate_levels(self, symbol: str, direction: str, entry_price: float, timeframe: str) -> tuple:
        """
        Calculate stop loss, take profit, and lot size for a trade.
        Returns: (sl, tp, lot_size)
        """
        # Get symbol info
        info = mt5.symbol_info(symbol)
        if not info:
            raise ValueError(f"Symbol {symbol} not found")
        
        # Get ATR for volatility-based stops
        atr = self._get_atr(symbol, timeframe)
        
        # Calculate SL distance
        if self.params.use_atr_stops and atr > 0:
            sl_distance = atr * self.params.atr_multiplier_sl
        else:
            # Fallback: fixed pip distance
            sl_distance = info.point * 200  # 200 points default
        
        # Ensure minimum distance (broker stop level)
        min_dist = info.trade_stops_level * info.point
        if sl_distance < min_dist:
            sl_distance = min_dist * 1.5
        
        # Calculate SL and TP
        if direction == "buy":
            sl = entry_price - sl_distance
            tp_distance = sl_distance * self.params.risk_reward_ratio
            tp = entry_price + tp_distance
        else:
            sl = entry_price + sl_distance
            tp_distance = sl_distance * self.params.risk_reward_ratio
            tp = entry_price - tp_distance
        
        # Normalize to symbol digits
        sl = round(sl, info.digits)
        tp = round(tp, info.digits)
        
        # Calculate lot size based on risk
        lot_size = self._calculate_lot_size(symbol, entry_price, sl)
        
        return sl, tp, lot_size
    
    def _get_atr(self, symbol: str, timeframe: str, period: int = 14) -> float:
        """Get ATR value from MT5."""
        try:
            timeframe_map = {
                "M1": mt5.TIMEFRAME_M1,
                "M5": mt5.TIMEFRAME_M5,
                "M15": mt5.TIMEFRAME_M15,
                "H1": mt5.TIMEFRAME_H1,
                "H4": mt5.TIMEFRAME_H4,
                "D1": mt5.TIMEFRAME_D1,
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
        except:
            return 0.0
    
    def _calculate_lot_size(self, symbol: str, entry_price: float, sl: float) -> float:
        """Calculate lot size based on account risk percentage."""
        account = mt5.account_info()
        if not account:
            return 0.01
        
        balance = account.balance
        risk_amount = balance * (self.params.risk_percent / 100)
        
        # Get symbol info for pip value
        info = mt5.symbol_info(symbol)
        if not info:
            return 0.01
        
        # Calculate pip value per lot
        # For forex: 1 lot = 100,000 units, 1 pip = 10 units of quote currency
        # Need to convert to account currency
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return 0.01
        
        # Simplified: assume USD account, major pairs
        # Proper implementation would use mt5.symbol_info(symbol).trade_tick_value
        tick_value = info.trade_tick_value
        tick_size = info.trade_tick_size
        
        if tick_value == 0 or tick_size == 0:
            return 0.01
        
        # Points difference
        points_diff = abs(entry_price - sl) / info.point
        
        # Value per lot for this SL distance
        value_per_lot = points_diff * tick_value
        
        if value_per_lot <= 0:
            return 0.01
        
        # Calculate lots
        lots = risk_amount / value_per_lot
        
        # Round to step
        step = info.volume_step
        lots = round(lots / step) * step
        
        # Clamp to min/max
        lots = max(info.volume_min, min(lots, info.volume_max))
        
        # Additional safety: cap at max risk percent
        max_risk_amount = balance * (self.params.max_risk_percent / 100)
        max_lots = max_risk_amount / value_per_lot if value_per_lot > 0 else info.volume_min
        lots = min(lots, max_lots)
        
        return round(lots, 2)
    
    def can_open_position(self, symbol: str, positions: Dict) -> bool:
        """Check if we can open a new position based on risk limits."""
        # Check daily drawdown
        self._check_daily_reset()
        if self.daily_pnl <= -abs(self.params.max_daily_drawdown / 100 * mt5.account_info().balance):
            return False
        
        # Count positions for this symbol
        symbol_positions = sum(1 for p in positions.values() if p.symbol == symbol)
        if symbol_positions >= self.params.max_positions_per_symbol:
            return False
        
        # Check total positions
        if len(positions) >= self.params.max_total_positions:
            return False
        
        # Check margin
        account = mt5.account_info()
        if account and account.margin_free < account.margin * 0.5:  # Less than 50% free margin
            return False
        
        return True
    
    def _check_daily_reset(self):
        """Reset daily PnL at midnight UTC."""
        now = datetime.utcnow()
        if now >= self.daily_reset_time + timedelta(days=1):
            self.daily_pnl = 0.0
            self.daily_reset_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
    
    def update_daily_pnl(self, pnl: float):
        """Update daily PnL tracker."""
        self._check_daily_reset()
        self.daily_pnl += pnl
    
    def get_account_risk_status(self) -> Dict:
        """Get current risk status."""
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
            "max_daily_drawdown": self.params.max_daily_drawdown,
            "open_positions": len(mt5.positions_get() or []),
            "max_positions": self.params.max_total_positions
        }
    
    def validate_signal(self, signal: Any, positions: Dict) -> bool:
        """Validate a trade signal against risk rules."""
        # Check if we can open position
        if not self.can_open_position(signal.symbol, positions):
            return False
        
        # Check signal has valid SL/TP
        if signal.sl <= 0 or signal.tp <= 0:
            return False
        
        # Check risk/reward
        if signal.direction == "buy":
            risk = signal.entry_price - signal.sl
            reward = signal.tp - signal.entry_price
        else:
            risk = signal.sl - signal.entry_price
            reward = signal.entry_price - signal.tp
        
        if risk <= 0:
            return False
        
        actual_rr = reward / risk
        if actual_rr < self.params.risk_reward_ratio * 0.8:  # Allow 20% tolerance
            return False
        
        return True


from datetime import timedelta