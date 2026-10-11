"""
Strategy Manager Module for WILLAM
Handles backtesting, strategy selection, and signal generation
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import yaml
import pickle


@dataclass
class StrategyResult:
    name: str
    symbol: str
    timeframe: str
    winrate: float
    expectancy: float
    profit_factor: float
    max_drawdown: float
    total_trades: int
    avg_trade: float
    sharpe: float
    parameters: Dict
    equity_curve: List[float] = None


@dataclass
class TrendInfo:
    symbol: str
    direction: str  # "up", "down", "sideways"
    strength: float  # 0-1
    confirmed: bool
    timeframe: str
    indicators: Dict


class StrategyManager:
    def __init__(self, config: Dict):
        self.config = config
        self.strat_config = config.get("strategy", {})
        self.trading_config = config.get("trading", {})
        self.backtest_dir = Path("backtest")
        self.backtest_dir.mkdir(exist_ok=True)
        
        # Strategy registry
        self.strategies = {
            "trend_follow": self._strategy_trend_follow,
            "mean_reversion": self._strategy_mean_reversion,
            "breakout": self._strategy_breakout,
            "rsi_divergence": self._strategy_rsi_divergence,
            "macd_trend": self._strategy_macd_trend,
            "bollinger_bands": self._strategy_bollinger_bands,
            "ichimoku": self._strategy_ichimoku,
        }
        
        # Cache for best strategies per symbol
        self.best_strategies: Dict[str, StrategyResult] = {}
    
    async def find_best_strategy(self, symbol: str) -> Optional[str]:
        """Find highest winrate strategy for a symbol via backtesting."""
        # Load historical data
        data = await self._get_historical_data(symbol)
        if data is None or len(data) < 1000:
            return None
        
        best_result = None
        best_winrate = self.strat_config.get("min_winrate", 55.0)
        
        for strat_name, strat_func in self.strategies.items():
            try:
                result = await self._backtest_strategy(strat_name, symbol, data)
                if result and result.winrate >= best_winrate:
                    if best_result is None or result.winrate > best_result.winrate:
                        best_result = result
                        best_winrate = result.winrate
            except Exception as e:
                print(f"Backtest failed for {strat_name} on {symbol}: {e}")
        
        if best_result:
            # Save backtest results
            await self._save_backtest_result(best_result)
            self.best_strategies[symbol] = best_result
            return best_result.name
        
        return None
    
    async def _get_historical_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get historical data from MT5 or cache."""
        # Try cache first
        cache_file = self.backtest_dir / f"{symbol}_{self.trading_config['timeframe']}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except:
                pass
        
        # Fetch from MT5
        try:
            import MetaTrader5 as mt5
            
            timeframe_map = {
                "M1": mt5.TIMEFRAME_M1,
                "M5": mt5.TIMEFRAME_M5,
                "M15": mt5.TIMEFRAME_M15,
                "H1": mt5.TIMEFRAME_H1,
                "H4": mt5.TIMEFRAME_H4,
                "D1": mt5.TIMEFRAME_D1,
            }
            
            tf = timeframe_map.get(self.trading_config["timeframe"], mt5.TIMEFRAME_M15)
            lookback = self.strat_config.get("lookback_days", 365)
            
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, lookback * 96)  # ~96 bars/day for M15
            
            if rates is None or len(rates) == 0:
                return None
            
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.set_index("time", inplace=True)
            df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "tick_volume": "Volume"}, inplace=True)
            
            # Cache it
            with open(cache_file, "wb") as f:
                pickle.dump(df, f)
            
            return df
            
        except Exception as e:
            print(f"Failed to get data for {symbol}: {e}")
            return None
    
    async def _backtest_strategy(self, name: str, symbol: str, data: pd.DataFrame) -> Optional[StrategyResult]:
        """Run vectorized backtest for a strategy."""
        # Generate signals
        signals = self.strategies[name](data.copy())
        
        if signals is None or signals.sum() == 0:
            return None
        
        # Vectorized backtest
        returns = data["Close"].pct_change().shift(-1)  # Next bar return
        strategy_returns = signals * returns
        
        # Apply costs
        commission = self.config["backtest"].get("commission", 0.00007)
        slippage = self.config["backtest"].get("slippage", 0.00003)
        spread_mult = self.config["backtest"].get("spread_multiplier", 1.2)
        
        # Cost per trade (round trip)
        cost = (commission + slippage) * 2 * spread_mult
        trades = signals.diff().abs()  # 1 on entry, 1 on exit
        strategy_returns = strategy_returns - trades * cost
        
        # Metrics
        trades_mask = strategy_returns != 0
        trade_returns = strategy_returns[trades_mask]
        
        if len(trade_returns) < self.strat_config.get("min_trades_for_validation", 100):
            return None
        
        winrate = (trade_returns > 0).mean() * 100
        expectancy = trade_returns.mean()
        profit_factor = trade_returns[trade_returns > 0].sum() / abs(trade_returns[trade_returns < 0].sum()) if (trade_returns < 0).any() else 999
        
        equity = (1 + strategy_returns).cumprod()
        max_dd = (equity / equity.cummax() - 1).min() * 100
        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252 * 96) if strategy_returns.std() > 0 else 0
        
        return StrategyResult(
            name=name,
            symbol=symbol,
            timeframe=self.trading_config["timeframe"],
            winrate=winrate,
            expectancy=expectancy,
            profit_factor=profit_factor,
            max_drawdown=max_dd,
            total_trades=int(trades_mask.sum()),
            avg_trade=expectancy,
            sharpe=sharpe,
            parameters={},  # Would store optimized params
            equity_curve=equity.tolist()
        )
    
    async def _save_backtest_result(self, result: StrategyResult):
        """Save backtest result to file."""
        filename = f"{result.symbol}_{result.name}_{result.timeframe}.json"
        filepath = self.backtest_dir / filename
        
        data = {
            "name": result.name,
            "symbol": result.symbol,
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
            "equity_curve": result.equity_curve[-1000:] if result.equity_curve else []  # Last 1000 points
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
    
    async def generate_signal(self, symbol: str, strategy_name: str) -> Optional[Any]:
        """Generate live signal for a symbol using selected strategy."""
        # Get recent data
        data = await self._get_recent_data(symbol, bars=500)
        if data is None:
            return None
        
        # Generate signal
        signals = self.strategies[strategy_name](data)
        
        if signals is None:
            return None
        
        last_signal = signals.iloc[-1]
        prev_signal = signals.iloc[-2] if len(signals) > 1 else 0
        
        # Only signal on change (new entry)
        if last_signal == prev_signal:
            return None
        
        if last_signal == 0:
            return None  # Exit signal, not entry
        
        direction = "buy" if last_signal > 0 else "sell"
        
        # Get current price
        try:
            import MetaTrader5 as mt5
            tick = mt5.symbol_info_tick(symbol)
            entry = tick.ask if direction == "buy" else tick.bid
        except:
            entry = data["Close"].iloc[-1]
        
        # Create signal object (shared dataclass — no circular import)
        from signals import TradeSignal
        return TradeSignal(
            symbol=symbol,
            direction=direction,
            entry_price=entry,
            sl=0,  # Will be set by risk manager
            tp=0,
            lot_size=0,
            strategy_name=strategy_name,
            confidence=0.7,
            reasoning=f"{strategy_name} signal on {symbol}"
        )
    
    async def _get_recent_data(self, symbol: str, bars: int = 500) -> Optional[pd.DataFrame]:
        """Get recent data for live signals."""
        try:
            import MetaTrader5 as mt5
            
            timeframe_map = {
                "M1": mt5.TIMEFRAME_M1,
                "M5": mt5.TIMEFRAME_M5,
                "M15": mt5.TIMEFRAME_M15,
                "H1": mt5.TIMEFRAME_H1,
                "H4": mt5.TIMEFRAME_H4,
                "D1": mt5.TIMEFRAME_D1,
            }
            
            tf = timeframe_map.get(self.trading_config["timeframe"], mt5.TIMEFRAME_M15)
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
            
            if rates is None:
                return None
            
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.set_index("time", inplace=True)
            df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "tick_volume": "Volume"}, inplace=True)
            
            return df
        except:
            return None
    
    async def get_current_trend(self, symbol: str) -> TrendInfo:
        """Get current trend for confirmation logic."""
        data = await self._get_recent_data(symbol, bars=200)
        if data is None:
            return TrendInfo(symbol, "sideways", 0, False, self.trading_config["timeframe"], {})
        
        # Simple trend detection: EMA alignment
        data["ema_fast"] = data["Close"].ewm(span=20).mean()
        data["ema_slow"] = data["Close"].ewm(span=50).mean()
        data["ema_trend"] = data["Close"].ewm(span=200).mean()
        
        last = data.iloc[-1]
        
        if last["ema_fast"] > last["ema_slow"] > last["ema_trend"]:
            direction = "up"
            strength = min((last["ema_fast"] - last["ema_trend"]) / last["ema_trend"] * 100, 1)
        elif last["ema_fast"] < last["ema_slow"] < last["ema_trend"]:
            direction = "down"
            strength = min((last["ema_trend"] - last["ema_fast"]) / last["ema_trend"] * 100, 1)
        else:
            direction = "sideways"
            strength = 0
        
        # Confirmation: price above/below EMAs consistently
        recent = data.tail(10)
        if direction == "up":
            confirmed = (recent["Close"] > recent["ema_fast"]).all()
        elif direction == "down":
            confirmed = (recent["Close"] < recent["ema_fast"]).all()
        else:
            confirmed = False
        
        return TrendInfo(
            symbol=symbol,
            direction=direction,
            strength=strength,
            confirmed=confirmed,
            timeframe=self.trading_config["timeframe"],
            indicators={"ema_fast": last["ema_fast"], "ema_slow": last["ema_slow"], "ema_trend": last["ema_trend"]}
        )
    
    # ===== Strategy Implementations =====
    
    def _strategy_trend_follow(self, data: pd.DataFrame) -> pd.Series:
        """EMA crossover trend following."""
        data["ema_fast"] = data["Close"].ewm(span=20).mean()
        data["ema_slow"] = data["Close"].ewm(span=50).mean()
        data["ema_trend"] = data["Close"].ewm(span=200).mean()
        
        # Long when fast > slow > trend, Short when fast < slow < trend
        long = (data["ema_fast"] > data["ema_slow"]) & (data["ema_slow"] > data["ema_trend"])
        short = (data["ema_fast"] < data["ema_slow"]) & (data["ema_slow"] < data["ema_trend"])
        
        signals = pd.Series(0, index=data.index)
        signals[long] = 1
        signals[short] = -1
        
        # Forward fill to hold position
        signals = signals.replace(0, np.nan).ffill().fillna(0)
        return signals
    
    def _strategy_mean_reversion(self, data: pd.DataFrame) -> pd.Series:
        """Bollinger Bands mean reversion."""
        period = 20
        std_mult = 2
        
        data["bb_mid"] = data["Close"].rolling(period).mean()
        data["bb_std"] = data["Close"].rolling(period).std()
        data["bb_upper"] = data["bb_mid"] + std_mult * data["bb_std"]
        data["bb_lower"] = data["bb_mid"] - std_mult * data["bb_std"]
        
        # Long at lower band, short at upper band
        long = data["Close"] < data["bb_lower"]
        short = data["Close"] > data["bb_upper"]
        
        signals = pd.Series(0, index=data.index)
        signals[long] = 1
        signals[short] = -1
        
        # Exit at mid band
        exit_long = (data["Close"] > data["bb_mid"]) & (signals.shift(1) == 1)
        exit_short = (data["Close"] < data["bb_mid"]) & (signals.shift(1) == -1)
        signals[exit_long] = 0
        signals[exit_short] = 0
        
        signals = signals.replace(0, np.nan).ffill().fillna(0)
        return signals
    
    def _strategy_breakout(self, data: pd.DataFrame) -> pd.Series:
        """Donchian channel breakout."""
        period = 20
        
        data["highest"] = data["High"].rolling(period).max()
        data["lowest"] = data["Low"].rolling(period).min()
        
        long = data["Close"] > data["highest"].shift(1)
        short = data["Close"] < data["lowest"].shift(1)
        
        signals = pd.Series(0, index=data.index)
        signals[long] = 1
        signals[short] = -1
        
        # Exit on opposite breakout or 2*ATR trailing
        data["atr"] = self._calculate_atr(data, 14)
        exit_long = data["Close"] < (data["Close"].shift(1) - 2 * data["atr"])
        exit_short = data["Close"] > (data["Close"].shift(1) + 2 * data["atr"])
        
        signals[exit_long] = 0
        signals[exit_short] = 0
        
        signals = signals.replace(0, np.nan).ffill().fillna(0)
        return signals
    
    def _strategy_rsi_divergence(self, data: pd.DataFrame) -> pd.Series:
        """RSI divergence strategy."""
        period = 14
        
        delta = data["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        rs = gain / loss
        data["rsi"] = 100 - (100 / (1 + rs))
        
        # Simple RSI levels (divergence detection would need peak/trough detection)
        long = (data["rsi"] < 30) & (data["rsi"].shift(1) >= 30)
        short = (data["rsi"] > 70) & (data["rsi"].shift(1) <= 70)
        
        signals = pd.Series(0, index=data.index)
        signals[long] = 1
        signals[short] = -1
        
        # Exit at 50
        exit_long = (data["rsi"] > 50) & (signals.shift(1) == 1)
        exit_short = (data["rsi"] < 50) & (signals.shift(1) == -1)
        signals[exit_long] = 0
        signals[exit_short] = 0
        
        signals = signals.replace(0, np.nan).ffill().fillna(0)
        return signals
    
    def _strategy_macd_trend(self, data: pd.DataFrame) -> pd.Series:
        """MACD trend following."""
        data["ema_fast"] = data["Close"].ewm(span=12).mean()
        data["ema_slow"] = data["Close"].ewm(span=26).mean()
        data["macd"] = data["ema_fast"] - data["ema_slow"]
        data["signal"] = data["macd"].ewm(span=9).mean()
        data["histogram"] = data["macd"] - data["signal"]
        
        long = (data["macd"] > data["signal"]) & (data["macd"].shift(1) <= data["signal"].shift(1))
        short = (data["macd"] < data["signal"]) & (data["macd"].shift(1) >= data["signal"].shift(1))
        
        signals = pd.Series(0, index=data.index)
        signals[long] = 1
        signals[short] = -1
        
        # Exit on opposite cross
        exit_long = (data["macd"] < data["signal"]) & (signals.shift(1) == 1)
        exit_short = (data["macd"] > data["signal"]) & (signals.shift(1) == -1)
        signals[exit_long] = 0
        signals[exit_short] = 0
        
        signals = signals.replace(0, np.nan).ffill().fillna(0)
        return signals
    
    def _strategy_bollinger_bands(self, data: pd.DataFrame) -> pd.Series:
        """Bollinger Bands with trend filter."""
        period = 20
        std_mult = 2
        
        data["bb_mid"] = data["Close"].rolling(period).mean()
        data["bb_std"] = data["Close"].rolling(period).std()
        data["bb_upper"] = data["bb_mid"] + std_mult * data["bb_std"]
        data["bb_lower"] = data["bb_mid"] - std_mult * data["bb_std"]
        data["bb_width"] = (data["bb_upper"] - data["bb_lower"]) / data["bb_mid"]
        
        # Trend filter: only trade in direction of 200 EMA
        data["ema_trend"] = data["Close"].ewm(span=200).mean()
        
        long = (data["Close"] < data["bb_lower"]) & (data["Close"] > data["ema_trend"])
        short = (data["Close"] > data["bb_upper"]) & (data["Close"] < data["ema_trend"])
        
        signals = pd.Series(0, index=data.index)
        signals[long] = 1
        signals[short] = -1
        
        exit_long = (data["Close"] > data["bb_mid"]) & (signals.shift(1) == 1)
        exit_short = (data["Close"] < data["bb_mid"]) & (signals.shift(1) == -1)
        signals[exit_long] = 0
        signals[exit_short] = 0
        
        signals = signals.replace(0, np.nan).ffill().fillna(0)
        return signals
    
    def _strategy_ichimoku(self, data: pd.DataFrame) -> pd.Series:
        """Ichimoku Cloud strategy."""
        # Tenkan-sen (9 period)
        high_9 = data["High"].rolling(9).max()
        low_9 = data["Low"].rolling(9).min()
        data["tenkan"] = (high_9 + low_9) / 2
        
        # Kijun-sen (26 period)
        high_26 = data["High"].rolling(26).max()
        low_26 = data["Low"].rolling(26).min()
        data["kijun"] = (high_26 + low_26) / 2
        
        # Senkou Span A (Leading Span A)
        data["senkou_a"] = ((data["tenkan"] + data["kijun"]) / 2).shift(26)
        
        # Senkou Span B (Leading Span B) - 52 period
        high_52 = data["High"].rolling(52).max()
        low_52 = data["Low"].rolling(52).min()
        data["senkou_b"] = ((high_52 + low_52) / 2).shift(26)
        
        # Chikou Span (Lagging Span)
        data["chikou"] = data["Close"].shift(-26)
        
        # Cloud boundaries
        data["cloud_top"] = data[["senkou_a", "senkou_b"]].max(axis=1)
        data["cloud_bottom"] = data[["senkou_a", "senkou_b"]].min(axis=1)
        
        # Signals: Price above cloud + Tenkan > Kijun = Long
        long = (data["Close"] > data["cloud_top"]) & (data["tenkan"] > data["kijun"])
        short = (data["Close"] < data["cloud_bottom"]) & (data["tenkan"] < data["kijun"])
        
        signals = pd.Series(0, index=data.index)
        signals[long] = 1
        signals[short] = -1
        
        # Exit on Tenkan/Kijun cross opposite
        exit_long = (data["tenkan"] < data["kijun"]) & (signals.shift(1) == 1)
        exit_short = (data["tenkan"] > data["kijun"]) & (signals.shift(1) == -1)
        signals[exit_long] = 0
        signals[exit_short] = 0
        
        signals = signals.replace(0, np.nan).ffill().fillna(0)
        return signals
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = data["High"] - data["Low"]
        high_close = (data["High"] - data["Close"].shift()).abs()
        low_close = (data["Low"] - data["Close"].shift()).abs()
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()