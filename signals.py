"""
Shared signal/position dataclasses for WILLAM.
Lives at package root so every module can import it without cycles.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional


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
    news_context: Optional[Dict] = None  # carried from signal -> evolution learning
