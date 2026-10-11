"""
News Checker Module for WILLAM — v2
Fixed: real ForexFactory JSON schema (title/country/ISO-8601 date),
proper timezone handling with DST via zoneinfo, actual-value polling.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any
from zoneinfo import ZoneInfo
from dataclasses import dataclass

import aiohttp

logger = logging.getLogger("willam.news")


@dataclass
class NewsEvent:
    name: str
    time: datetime  # timezone-aware UTC
    currency: str
    impact: str
    forecast: Optional[float]
    previous: Optional[float]
    actual: Optional[float] = None
    symbol: str = ""  # related trading symbol

    def __post_init__(self):
        currency_map = {
            "USD": "EURUSD",
            "EUR": "EURUSD",
            "GBP": "GBPUSD",
            "JPY": "USDJPY",
            "AUD": "AUDUSD",
            "CAD": "USDCAD",
            "NZD": "NZDUSD",
            "CHF": "USDCHF",
        }
        self.symbol = currency_map.get(self.currency, "EURUSD")


from dataclasses import dataclass


class NewsChecker:
    FF_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    IMPACT_MAP = {"Low": "low", "Medium": "medium", "High": "high", "Holiday": "low"}

    def __init__(self, config: Dict):
        self.config = config
        self.news_config = config.get("news", {})
        self.cache: List[NewsEvent] = []
        self.last_fetch = datetime.min
        self.fetch_failed_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_upcoming_news(self, hours_ahead: int = 1) -> List[NewsEvent]:
        """High-impact events in the next N hours (UTC)."""
        now = datetime.now(timezone.utc)

        # Refresh cache every 15 min
        if (now - self.last_fetch.replace(tzinfo=timezone.utc)).total_seconds() > 900:
            await self._fetch_calendar()

        upcoming = [
            e for e in self.cache
            if e.impact == "high" and now <= e.time <= now + timedelta(hours=hours_ahead)
        ]
        return upcoming

    async def get_news_window_status(self) -> Dict[str, Any]:
        """
        Non-blocking window check used by the main loop every cycle:
        - 'blocked': high-impact event due within pause_before_minutes -> no NEW entries
        - 'just_released': event released within wait_after_minutes -> verify surprise, no blind entries
        - 'clear': safe to trade strategies normally
        """
        now = datetime.now(timezone.utc)
        pause_before = self.news_config.get("pause_before_minutes", 60)
        wait_after = self.news_config.get("wait_after_minutes", 30)

        # refresh cache if stale
        try:
            if (now - self.last_fetch.replace(tzinfo=timezone.utc)).total_seconds() > 900:
                await self._fetch_calendar()
        except Exception:
            pass

        high = [e for e in self.cache if e.impact == "high"]

        for e in high:
            mins_until = (e.time - now).total_seconds() / 60
            mins_since = (now - e.time).total_seconds() / 60
            if 0 <= mins_until <= pause_before:
                return {"status": "blocked", "event": e, "minutes_until": round(mins_until, 1)}
            if 0 <= mins_since <= wait_after:
                return {"status": "just_released", "event": e, "minutes_since": round(mins_since, 1)}

        return {"status": "clear"}

    async def get_actual_value(self, event: NewsEvent, max_wait_minutes: int = 12, poll_seconds: int = 45) -> Optional[float]:
        """Poll until the actual value appears (ForexFactory updates within minutes)."""
        deadline = datetime.now(timezone.utc) + timedelta(minutes=max_wait_minutes)
        last_err = None
        while datetime.now(timezone.utc) < deadline:
            try:
                await self._fetch_calendar(force=True)
                for e in self.cache:
                    if e.name == event.name and abs((e.time - event.time).total_seconds()) < 3600:
                        if e.actual is not None:
                            return e.actual
            except Exception as ex:
                last_err = ex
            await asyncio.sleep(poll_seconds)
        if last_err:
            logger.warning(f"get_actual_value failed: {last_err}")
        return None

    # ------------------------------------------------------------------
    # Fetch + parse (real ForexFactory schema)
    # ------------------------------------------------------------------

    async def _fetch_calendar(self, force: bool = False):
        self.last_fetch = datetime.now(timezone.utc)
        events: List[NewsEvent] = []

        try:
            events = await self._fetch_forexfactory()
            self.fetch_failed_count = 0
        except Exception as e:
            self.fetch_failed_count += 1
            logger.warning(f"ForexFactory fetch failed ({self.fetch_failed_count}): {e}")

        if events:
            # Deduplicate by name+time
            seen = set()
            unique = []
            for e in events:
                key = (e.name, e.time)
                if key not in seen:
                    seen.add(key)
                    unique.append(e)
            self.cache = unique
            logger.info(f"Fetched {len(unique)} calendar events")
        elif force:
            logger.warning("Calendar fetch returned no events on forced refresh")

    async def _fetch_forexfactory(self) -> List[NewsEvent]:
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(self.FF_URL) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"FF calendar HTTP {resp.status}")
                data = await resp.json()
        return self._parse_forexfactory(data)

    def _parse_forexfactory(self, data: List[Dict]) -> List[NewsEvent]:
        """Real schema: title, country, date (ISO-8601 with offset), impact, forecast, previous."""
        events = []
        now_utc = datetime.now(timezone.utc)

        for item in data:
            try:
                impact = self.IMPACT_MAP.get(str(item.get("impact", "")).strip(), "low")

                date_raw = str(item.get("date", "")).strip()
                if not date_raw:
                    continue
                # ISO-8601 with offset e.g. "2026-10-06T02:35:00-04:00" -> aware datetime
                event_time = datetime.fromisoformat(date_raw)
                if event_time.tzinfo is None:
                    # ForexFactory publishes in US/Eastern — attach it
                    event_time = event_time.replace(tzinfo=ZoneInfo("America/New_York"))
                event_time = event_time.astimezone(timezone.utc)

                # Skip events more than a day old
                if (now_utc - event_time).total_seconds() > 86400:
                    continue

                currency = str(item.get("country", "")).strip() or "USD"
                if currency == "All":
                    currency = "USD"

                events.append(NewsEvent(
                    name=str(item.get("title", "Unknown")).strip(),
                    time=event_time,
                    currency=currency,
                    impact=impact,
                    forecast=self._parse_number(item.get("forecast")),
                    previous=self._parse_number(item.get("previous")),
                ))
            except Exception as e:
                logger.debug(f"Skipping calendar item: {e}")
                continue

        return events

    def _parse_number(self, val: Any) -> Optional[float]:
        """Parse '5.2%', '1.2K', '3.5M', '-0.3' -> float (raw, no % scaling — keep units)."""
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)

        s = str(val).strip().replace(",", "")
        if not s or s.lower() in ("n/a", "na", "-"):
            return None

        pct = s.endswith("%")
        if pct:
            s = s[:-1]

        mult = 1
        if s and s[-1].upper() in "KMB":
            mult = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}[s[-1].upper()]
            s = s[:-1]

        try:
            num = float(s) * mult
            # Keep percentages in percent units (e.g. CPI 3.2% -> 3.2, not 0.032)
            return num
        except ValueError:
            return None
