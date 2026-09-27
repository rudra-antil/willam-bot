"""
News Checker Module for WILLAM
Checks economic calendar for high-impact events
"""

import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import json


class ImpactLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class NewsEvent:
    name: str
    time: datetime
    currency: str
    impact: str
    forecast: Optional[float]
    previous: Optional[float]
    actual: Optional[float] = None
    symbol: str = ""  # Related trading symbol
    
    def __post_init__(self):
        # Map currency to symbol
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


class NewsChecker:
    def __init__(self, config: Dict):
        self.config = config
        self.news_config = config.get("news", {})
        self.cache: List[NewsEvent] = []
        self.last_fetch = datetime.min
        
    async def get_upcoming_news(self, hours_ahead: int = 1) -> List[NewsEvent]:
        """Get high-impact news events in the next N hours."""
        now = datetime.utcnow()
        cutoff = now + timedelta(hours=hours_ahead)
        
        # Refresh cache if older than 15 minutes
        if (now - self.last_fetch).total_seconds() > 900:
            await self._fetch_calendar()
        
        # Filter to high-impact within window
        upcoming = [
            e for e in self.cache
            if e.impact == "high" and now <= e.time <= cutoff
        ]
        
        return upcoming
    
    async def _fetch_calendar(self):
        """Fetch economic calendar from API."""
        self.last_fetch = datetime.utcnow()
        
        # Try multiple sources
        events = []
        
        # Source 1: ForexFactory (via API or scraping)
        try:
            ff_events = await self._fetch_forexfactory()
            events.extend(ff_events)
        except Exception as e:
            print(f"ForexFactory fetch failed: {e}")
        
        # Source 2: Investing.com (if API available)
        try:
            inv_events = await self._fetch_investing()
            events.extend(inv_events)
        except Exception as e:
            print(f"Investing.com fetch failed: {e}")
        
        # Deduplicate by name+time
        seen = set()
        unique = []
        for e in events:
            key = (e.name, e.time)
            if key not in seen:
                seen.add(key)
                unique.append(e)
        
        self.cache = unique
        print(f"Fetched {len(unique)} news events")
    
    async def _fetch_forexfactory(self) -> List[NewsEvent]:
        """Fetch from ForexFactory calendar."""
        # This is a simplified version - in production, use their API or scrape
        # ForexFactory has a JSON endpoint: https://nfs.faireconomy.media/ff_calendar_thisweek.json
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return self._parse_forexfactory(data)
        return []
    
    def _parse_forexfactory(self, data: List[Dict]) -> List[NewsEvent]:
        """Parse ForexFactory JSON format."""
        events = []
        impact_map = {"Low": "low", "Medium": "medium", "High": "high"}
        
        for item in data:
            try:
                # Parse date/time
                date_str = item.get("date", "")
                time_str = item.get("time", "")
                
                if not date_str or not time_str:
                    continue
                
                # ForexFactory format: "Jan 15" + "08:30am"
                dt_str = f"{date_str} {time_str}"
                event_time = datetime.strptime(dt_str, "%b %d %I:%M%p")
                event_time = event_time.replace(year=datetime.utcnow().year)
                
                # Adjust for UTC (FF is usually ET)
                # This is simplified - proper implementation needs timezone handling
                event_time = event_time + timedelta(hours=4)  # ET to UTC (approx)
                
                impact = impact_map.get(item.get("impact", "Low"), "low")
                
                # Only keep high impact
                if impact != "high":
                    continue
                
                # Parse forecast/previous
                forecast = self._parse_number(item.get("forecast"))
                previous = self._parse_number(item.get("previous"))
                
                event = NewsEvent(
                    name=item.get("event", "Unknown"),
                    time=event_time,
                    currency=item.get("currency", "USD"),
                    impact=impact,
                    forecast=forecast,
                    previous=previous
                )
                events.append(event)
                
            except Exception as e:
                continue
        
        return events
    
    async def _fetch_investing(self) -> List[NewsEvent]:
        """Fetch from Investing.com (placeholder - requires API key)."""
        # Investing.com requires paid API for calendar
        # This is a placeholder for future implementation
        return []
    
    async def get_actual_value(self, event: NewsEvent) -> Optional[float]:
        """Get actual released value for an event."""
        # Re-fetch calendar to get updated actual values
        await self._fetch_calendar()
        
        for e in self.cache:
            if e.name == event.name and abs((e.time - event.time).total_seconds()) < 300:
                return e.actual
        
        return None
    
    def _parse_number(self, val: Any) -> Optional[float]:
        """Parse number from string like '5.2%', '1.2K', '3.5M'."""
        if val is None or val == "":
            return None
        if isinstance(val, (int, float)):
            return float(val)
        
        s = str(val).strip().replace(",", "")
        
        # Handle percentage
        if s.endswith("%"):
            return float(s[:-1]) / 100
        
        # Handle K/M/B
        mult = 1
        if s.endswith("K"):
            mult = 1000
            s = s[:-1]
        elif s.endswith("M"):
            mult = 1000000
            s = s[:-1]
        elif s.endswith("B"):
            mult = 1000000000
            s = s[:-1]
        
        try:
            return float(s) * mult
        except ValueError:
            return None