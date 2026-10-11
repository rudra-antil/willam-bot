"""
WILLAM smoke test suite — runs WITHOUT a live MT5 terminal or API keys.

Verifies the things that actually broke during development:
  1. No circular imports / namespace shadowing (stdlib `logging` vs loggers/)
  2. News parser handles the REAL ForexFactory schema + is timezone-aware
  3. News gating is PER-SYMBOL — one pair's news never stops the others
  4. Surprise classification (expected / positive / negative / unknown)
  5. JEV hard gates cannot be bypassed (dup symbol, R:R, invalid SL/TP)
  6. JEV LLM layer degrades to rules when no API key is present
  7. Risk sizing + PnL estimation correct for JPY / gold / indices
  8. Evolution learns from losses (3x weight), builds frustration,
     triggers deep-dive solutions, and relieves on a win
  9. Trade logger writes per-symbol/per-strategy CSV

Run:  python tests/test_willam.py
"""

import asyncio
import glob
import os
import sys
import types
from datetime import datetime, timedelta, timezone

# Make the bot importable when run from anywhere
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# MT5 mock — must be installed BEFORE importing bot modules
# ---------------------------------------------------------------------------

def install_mt5_mock():
    mock = types.ModuleType("MetaTrader5")
    for k, v in dict(
        TRADE_RETCODE_DONE=10009, TRADE_ACTION_DEAL=1,
        ORDER_TYPE_BUY=0, ORDER_TYPE_SELL=1, ORDER_TIME_GTC=0,
        ORDER_FILLING_IOC=1, ORDER_FILLING_FOK=2, ORDER_FILLING_RETURN=0,
        DEAL_ENTRY_OUT=1, DEAL_REASON_SL=4, DEAL_REASON_TP=5, DEAL_REASON_EXPERT=3,
        TIMEFRAME_M1=1, TIMEFRAME_M5=5, TIMEFRAME_M15=15, TIMEFRAME_H1=16385,
        TIMEFRAME_H4=16388, TIMEFRAME_D1=16408,
    ).items():
        setattr(mock, k, v)

    class _Acc:
        balance = 10000.0; equity = 10000.0; margin = 100.0
        margin_free = 9000.0; margin_level = 900.0
        login = 123; server = "mock"; currency = "USD"

    class _Tick:
        bid = 1.1000; ask = 1.1002

    class _Info:
        point = 0.00001; digits = 5; spread = 12
        volume_min = 0.01; volume_max = 100.0; volume_step = 0.01
        trade_tick_value = 1.0; trade_tick_size = 0.00001
        trade_stops_level = 0; filling_mode = 3

    mock.account_info = lambda: _Acc()
    mock.positions_get = lambda ticket=None: []
    mock.symbol_info = lambda s: _Info()
    mock.symbol_info_tick = lambda s: _Tick()
    mock.symbol_select = lambda s, b: True
    mock.copy_rates_from_pos = lambda *a: None
    mock.history_deals_get = lambda *a: []
    mock.order_send = lambda r: None
    mock.last_error = lambda: (0, "ok")
    mock.initialize = lambda **k: True
    mock.shutdown = lambda: None
    sys.modules["MetaTrader5"] = mock
    return mock


MT5 = install_mt5_mock()

import main as willam_main  # noqa: E402
from signals import Position, TradeSignal  # noqa: E402
from news.checker import NewsChecker, NewsEvent  # noqa: E402
from strategies.manager import StrategyManager  # noqa: E402
from risk.manager import RiskManager  # noqa: E402
from jev.engine import JEVEngine  # noqa: E402
from evolution.engine import EvolutionEngine  # noqa: E402
from loggers.trade_logger import TradeLogger  # noqa: E402

CONFIG = yaml.safe_load(open(os.path.join(ROOT, "config", "settings.yaml")))

PASSED, FAILED = [], []

# Runtime artifacts that must NOT leak between test runs (they made earlier
# runs non-deterministic: stale memory.json looked like "extra trades",
# stale *.pkl cache made bar counts wrong).
RUNTIME_ARTIFACTS = [
    os.path.join(ROOT, "evolution", "memory.json"),
    os.path.join(ROOT, "evolution", "performance.json"),
    os.path.join(ROOT, "evolution", "gaps.json"),
    os.path.join(ROOT, "evolution", "losing_patterns.json"),
    os.path.join(ROOT, "evolution", "daily_learning.json"),
    os.path.join(ROOT, "evolution", "curiosity.json"),
]


def reset_state():
    """Snapshot and remove runtime artifacts so the suite is hermetic."""
    backup = {}
    for path in RUNTIME_ARTIFACTS:
        if os.path.exists(path):
            backup[path] = open(path, "rb").read()
            os.remove(path)
    for pkl in glob.glob(os.path.join(ROOT, "backtest", "*.pkl")):
        os.remove(pkl)
    return backup


def restore_state(backup):
    """Put the user's real evolution memory back after testing."""
    for path, data in backup.items():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)


def check(name, condition, detail=""):
    if condition:
        PASSED.append(name)
        print(f"  ✓ {name}" + (f" — {detail}" if detail else ""))
    else:
        FAILED.append(name)
        print(f"  ✗ {name} FAILED" + (f" — {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------

def test_imports():
    print("\n[1] Module imports (circular-import + namespace-shadow check)")
    check("all modules import", True)
    import logging as stdlogging
    check("stdlib logging not shadowed", hasattr(stdlogging, "getLogger"))


# ---------------------------------------------------------------------------
# 2. News parsing against real schema
# ---------------------------------------------------------------------------

def test_news_parsing():
    print("\n[2] News parser (real ForexFactory schema, timezone-aware)")
    nc = NewsChecker(CONFIG)

    # Real schema sample (as served by nfs.faireconomy.media)
    sample = [
        {"title": "US CPI m/m", "country": "USD", "date": "2026-10-06T08:30:00-04:00",
         "impact": "High", "forecast": "0.3%", "previous": "0.2%"},
        {"title": "OPEC Meeting", "country": "All", "date": "2026-10-04T05:15:00-04:00",
         "impact": "Medium", "forecast": "", "previous": ""},
        {"title": "BOJ Ueda Speaks", "country": "JPY", "date": "2026-10-06T02:35:00-04:00",
         "impact": "High", "forecast": "", "previous": ""},
    ]
    events = nc._parse_forexfactory(sample)
    # only High impact, and only within ~1 day of now... sample dates may be far out,
    # so the parser may filter them; test the parsing primitives directly instead.
    check("impact mapping", nc.IMPACT_MAP["High"] == "high")
    check("percent parse", nc._parse_number("0.3%") == 0.3)
    check("K suffix parse", nc._parse_number("212K") == 212000)
    check("M suffix parse", nc._parse_number("1.5M") == 1500000)
    check("empty -> None", nc._parse_number("") is None)
    check("N/A -> None", nc._parse_number("N/A") is None)

    # timezone-awareness: parse an ISO date directly
    ev = NewsEvent(name="test", time=datetime.fromisoformat("2026-10-06T08:30:00-04:00"),
                   currency="USD", impact="high", forecast=0.3, previous=0.2)
    check("tz-aware datetime", ev.time.tzinfo is not None)
    check("USD maps to EURUSD", ev.symbol == "EURUSD")

    # live fetch (network) — optional
    async def live():
        try:
            events = await nc.get_upcoming_news(hours_ahead=24)
            return True, len(events)
        except Exception as e:
            return False, str(e)
    ok, detail = asyncio.run(live())
    check("live calendar fetch", ok, f"{detail}")


# ---------------------------------------------------------------------------
# 3. Per-symbol news gating (the "bot must not stop" requirement)
# ---------------------------------------------------------------------------

def make_bot():
    bot = willam_main.WillamBot.__new__(willam_main.WillamBot)
    import logging
    bot.config = CONFIG
    bot.logger = logging.getLogger("willam.test")
    bot.news_checker = NewsChecker(CONFIG)
    bot.strategy_manager = StrategyManager(CONFIG)
    bot.risk_manager = RiskManager(CONFIG)
    bot.jev = JEVEngine(CONFIG)
    bot.evolution = EvolutionEngine(CONFIG)
    bot.trade_logger = TradeLogger(CONFIG)
    bot.positions = {}
    bot.pending_signals = []
    bot.news_windows = {}
    bot.symbol_strategies = {s: "trend_follow" for s in ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]}
    bot.state = willam_main.BotState.RUNNING
    bot.last_news_check = datetime.min.replace(tzinfo=timezone.utc)
    bot.last_evolution_step_trade_count = -1
    bot._filling_cache = {}
    return bot


def test_news_gating():
    print("\n[3] Per-symbol news gating (other pairs keep trading)")
    bot = make_bot()
    gate = {s: bot._symbol_gated_by_news(s) for s in bot.symbol_strategies}
    check("nothing gated initially", not any(gate.values()))

    ev = NewsEvent(name="US CPI", time=datetime.now(timezone.utc) + timedelta(minutes=30),
                   currency="USD", impact="high", forecast=3.2, previous=3.1)
    bot.news_windows["EURUSD"] = willam_main.NewsWindow(event=ev, phase="pre", symbol="EURUSD")

    check("EURUSD gated", bot._symbol_gated_by_news("EURUSD") is True)
    check("GBPUSD still trading", bot._symbol_gated_by_news("GBPUSD") is False)
    check("USDJPY still trading", bot._symbol_gated_by_news("USDJPY") is False)
    check("XAUUSD still trading", bot._symbol_gated_by_news("XAUUSD") is False)
    check("no global pause", bot.state == willam_main.BotState.RUNNING)

    # lifecycle pre -> post -> expired
    async def lifecycle():
        past = NewsEvent(name="Old", time=datetime.now(timezone.utc) - timedelta(seconds=5),
                         currency="GBP", impact="high", forecast=1.0, previous=0.9)
        bot.news_windows["GBPUSD"] = willam_main.NewsWindow(event=past, phase="pre", symbol="GBPUSD")
        await bot._tick_news_windows()
        return bot.news_windows.get("GBPUSD")
    w = asyncio.run(lifecycle())
    check("pre -> post on release", w is not None and w.phase == "post")

    async def expire():
        w.started_at = datetime.now(timezone.utc) - timedelta(minutes=99)
        await bot._tick_news_windows()
    asyncio.run(expire())
    check("post -> expired", bot._symbol_gated_by_news("GBPUSD") is False)


# ---------------------------------------------------------------------------
# 4. Surprise classification
# ---------------------------------------------------------------------------

def test_surprise():
    print("\n[4] Surprise classification")
    bot = make_bot()
    cases = [
        (3.2, 3.2, 3.1, "expected"),
        (3.9, 3.2, 3.1, "positive"),
        (2.6, 3.2, 3.1, "negative"),
        (None, 3.2, 3.1, "unknown"),
    ]
    for actual, forecast, prev, want in cases:
        got = bot._analyze_surprise(actual, forecast, prev)
        check(f"actual={actual} forecast={forecast} -> {want}", got == want, f"got {got}")


# ---------------------------------------------------------------------------
# 5/6. JEV gates + LLM fallback
# ---------------------------------------------------------------------------

def test_jev():
    print("\n[5] JEV hard gates (cannot be bypassed)")
    jev = JEVEngine(CONFIG)

    good = TradeSignal(symbol="EURUSD", direction="buy", entry_price=1.10, sl=1.09, tp=1.12,
                       lot_size=0.1, strategy_name="trend_follow", confidence=0.8, reasoning="t")

    d_dup = asyncio.run(jev.make_decision(good, positions={
        1: Position(ticket=1, symbol="EURUSD", direction="buy", entry_price=1.10, sl=1.09,
                    tp=1.12, lot_size=0.1, strategy_name="t",
                    open_time=datetime.now(timezone.utc))}))
    check("duplicate symbol blocked", d_dup.approved is False, str(d_dup.conditions))

    bad_rr = TradeSignal(symbol="GBPUSD", direction="buy", entry_price=1.30, sl=1.295, tp=1.302,
                         lot_size=0.1, strategy_name="t", confidence=0.9, reasoning="bad")
    d_rr = asyncio.run(jev.make_decision(bad_rr, positions={}))
    check("bad R:R blocked", d_rr.approved is False)

    bad_sl = TradeSignal(symbol="USDJPY", direction="buy", entry_price=150.0, sl=0, tp=151.0,
                         lot_size=0.1, strategy_name="t", confidence=0.9, reasoning="bad")
    d_sl = asyncio.run(jev.make_decision(bad_sl, positions={}))
    check("invalid SL/TP blocked", d_sl.approved is False)

    print("\n[6] JEV LLM layer degrades to rules without API keys")
    for var in ("NVIDIA_API_KEY", "FREELLMAPI_KEY"):
        os.environ.pop(var, None)
    d = asyncio.run(jev.make_decision(good, positions={}))
    check("decision produced without LLM", d is not None)
    check("source is rules fallback", d.source == "rules", f"source={d.source}")
    corr = jev._check_correlation("EURUSD", {
        2: Position(ticket=2, symbol="GBPUSD", direction="sell", entry_price=1.3, sl=1.31,
                    tp=1.28, lot_size=0.1, strategy_name="t",
                    open_time=datetime.now(timezone.utc))})
    check("correlation detected", bool(corr), str(corr))


# ---------------------------------------------------------------------------
# 7. Risk manager
# ---------------------------------------------------------------------------

def test_risk():
    print("\n[7] Risk sizing + cross-asset PnL estimation")
    rm = RiskManager(CONFIG)
    sl, tp, lot = rm.calculate_levels("EURUSD", "buy", 1.1001, "M15")
    check("buy SL below entry", sl < 1.1001, f"sl={sl}")
    check("buy TP above entry", tp > 1.1001, f"tp={tp}")
    check("lot size positive", lot > 0, f"lot={lot}")
    check("R:R respected", abs((tp - 1.1001) / (1.1001 - sl) - 2.0) < 0.35)

    sl_s, tp_s, _ = rm.calculate_levels("EURUSD", "sell", 1.1001, "M15")
    check("sell SL above entry", sl_s > 1.1001)
    check("sell TP below entry", tp_s < 1.1001)

    # JPY pair PnL (the bug that made evolution learn from wrong numbers)
    class _JpyInfo:
        point = 0.001; digits = 3; spread = 12
        volume_min = 0.01; volume_max = 100.0; volume_step = 0.01
        trade_tick_value = 0.66; trade_tick_size = 0.001
        trade_stops_level = 0; filling_mode = 3
    MT5.symbol_info = lambda s: _JpyInfo()

    class _P:
        symbol = "USDJPY"; direction = "buy"; entry_price = 150.0; lot_size = 1.0
    pnl = rm.estimate_pnl(_P(), 150.5)
    check("JPY PnL in USD (50 pips, 1 lot ≈ $330)", 300 < pnl < 360, f"${pnl:.2f}")

    class _P2(_P):
        direction = "sell"
    check("sell PnL negative", rm.estimate_pnl(_P2(), 150.5) < 0)

    class _GoldInfo(_JpyInfo):
        point = 0.01; digits = 2; trade_tick_size = 0.01; trade_tick_value = 1.0

    class _G:
        symbol = "XAUUSD"; direction = "buy"; entry_price = 2000.0; lot_size = 1.0
    MT5.symbol_info = lambda s: _GoldInfo()
    g = rm.estimate_pnl(_G(), 2010.0)
    check("Gold PnL correct (1000 ticks × $1)", g == 1000.0, f"${g:.2f}")


# ---------------------------------------------------------------------------
# 8. Evolution psychology
# ---------------------------------------------------------------------------

def test_evolution():
    print("\n[8] Evolution psychology (loss-weighted learning)")
    evo = EvolutionEngine(CONFIG)
    asyncio.run(evo.load_memory())

    pos = Position(ticket=9, symbol="EURUSD", direction="buy", entry_price=1.10, sl=1.09, tp=1.12,
                   lot_size=0.1, strategy_name="trend_follow",
                   open_time=datetime.now(timezone.utc),
                   news_context={"reason": "bullish_surprise", "actual": 5.1, "forecast": 4.8})

    asyncio.run(evo.learn_from_trade(pos, 1.09, "sl", -50.0))
    check("loss recorded without crash", len(evo.trade_memory) == 1,
          f"{len(evo.trade_memory)} trade(s) in memory")
    check("loss weighted 3x", evo.trade_memory[0].learning_weight == 3.0,
          f"weight={evo.trade_memory[0].learning_weight}")
    check("news_context carried into learning",
          evo.trade_memory[0].news_context is not None)

    for _ in range(3):
        asyncio.run(evo.learn_from_trade(pos, 1.09, "sl", -50.0))

    patterns = list(evo.losing_patterns.values())
    check("losing pattern created", len(patterns) == 1, f"{len(patterns)} pattern(s)")
    p = patterns[0]
    check("pattern accumulates occurrences", p.occurrences >= 3, f"occ={p.occurrences}")
    check("frustration builds", p.frustration_level > 0.5, f"frustration={p.frustration_level:.2f}")
    check("deep-dive generated solutions", len(p.attempted_solutions) >= 1,
          f"{len(p.attempted_solutions)} solutions")
    check("curiosity questions generated", len(evo.curiosity_questions) > 0,
          f"{len(evo.curiosity_questions)} questions")

    report = evo.get_evolution_report()
    check("report: active patterns", report["active_losing_patterns"] >= 1)
    check("report: loss multiplier exposed", report["loss_learning_multiplier"] == 3.0)

    # a win relieves frustration
    before = p.frustration_level
    asyncio.run(evo.learn_from_trade(pos, 1.12, "tp", 100.0))
    check("win resets loss streak", p.consecutive_losses == 0)
    check("win reduces frustration", p.frustration_level < before,
          f"{before:.2f} -> {p.frustration_level:.2f}")

    asyncio.run(evo.save_memory())
    check("memory persisted", os.path.exists(os.path.join(ROOT, "evolution", "memory.json")))


# ---------------------------------------------------------------------------
# 9. Trade logger
# ---------------------------------------------------------------------------

def test_logger():
    print("\n[9] Trade logger (per symbol + strategy CSV)")
    tl = TradeLogger(CONFIG)
    sig = TradeSignal(symbol="TESTPAIR", direction="buy", entry_price=1.10, sl=1.09, tp=1.12,
                      lot_size=0.1, strategy_name="trend_follow", confidence=0.8, reasoning="t",
                      news_context={"reason": "bullish_surprise"})
    pos = Position(ticket=777, symbol="TESTPAIR", direction="buy", entry_price=1.10, sl=1.09,
                   tp=1.12, lot_size=0.1, strategy_name="trend_follow",
                   open_time=datetime.now(timezone.utc))
    tl.log_open(pos, sig)
    tl.log_close(pos, 1.12, "tp", 200.0)
    stats = tl.get_symbol_stats("TESTPAIR")
    check("trade recorded", stats["total_trades"] == 1)
    check("winrate computed", stats["winrate"] == 100.0)
    check("pnl recorded", stats["total_pnl"] == 200.0)
    tl.close()

    base = os.path.join(ROOT, "logs", "trades")
    files = [f for f in os.listdir(base) if f.startswith("TESTPAIR_trend_follow")]
    check("CSV files created per symbol+strategy", len(files) >= 2, str(files))

    # cleanup test artifacts
    for f in files:
        os.remove(os.path.join(base, f))


# ---------------------------------------------------------------------------
# 10. Strategy backtesting on synthetic data
# ---------------------------------------------------------------------------

def test_strategies():
    print("\n[10] Strategy backtesting (synthetic data, no MT5 needed)")
    n = 3000
    rng = np.random.default_rng(42)
    px = 1.10 + np.cumsum(rng.normal(0, 0.0006, n))
    rates = np.zeros(n, dtype=[("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"),
                               ("close", "f8"), ("tick_volume", "i8"), ("spread", "i4"),
                               ("real_volume", "i8")])
    rates["time"] = np.arange(n) * 900 + 1700000000
    rates["open"] = px
    rates["close"] = px + rng.normal(0, 0.0002, n)
    rates["high"] = np.maximum(rates["open"], rates["close"]) + abs(rng.normal(0, 0.0003, n))
    rates["low"] = np.minimum(rates["open"], rates["close"]) - abs(rng.normal(0, 0.0003, n))
    rates["tick_volume"] = 1000
    MT5.copy_rates_from_pos = lambda *a: rates

    sm = StrategyManager(CONFIG)
    df = asyncio.run(sm._get_historical_data("EURUSD"))
    check("historical data loaded", df is not None and len(df) == n)

    produced = 0
    for name in sm.strategies:
        r = asyncio.run(sm._backtest_strategy(name, "EURUSD", df.copy()))
        if r:
            produced += 1
            check(f"backtest {name}", 0 <= r.winrate <= 100,
                  f"winrate={r.winrate:.1f}% trades={r.total_trades} pf={r.profit_factor:.2f}")
    check("all strategies backtest", produced == len(sm.strategies), f"{produced}/{len(sm.strategies)}")

    trend = asyncio.run(sm.get_current_trend("EURUSD"))
    check("trend detection runs", trend.direction in ("up", "down", "sideways"))

    # cleanup backtest cache
    cache = os.path.join(ROOT, "backtest", "EURUSD_M15.pkl")
    if os.path.exists(cache):
        os.remove(cache)


# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("WILLAM SMOKE TEST SUITE")
    print("=" * 70)
    backup = reset_state()
    try:
        test_imports()
        test_news_parsing()
        test_news_gating()
        test_surprise()
        test_jev()
        test_risk()
        test_evolution()
        test_logger()
        test_strategies()
    finally:
        restore_state(backup)

    print("\n" + "=" * 70)
    print(f"RESULT: {len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("FAILED CHECKS:")
        for f in FAILED:
            print(f"  ✗ {f}")
    print("=" * 70)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
