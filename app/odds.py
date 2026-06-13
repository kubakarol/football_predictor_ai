from __future__ import annotations

import os
from typing import Any, Dict, List

import requests

THE_ODDS_API_BASE = "https://api.the-odds-api.com/v4"


def get_soccer_sports() -> List[Dict[str, Any]]:
    api_key = os.getenv("ODDS_API_KEY")

    if not api_key:
        return [{"error": "Set ODDS_API_KEY in .env to use The Odds API."}]

    r = requests.get(
        f"{THE_ODDS_API_BASE}/sports",
        params={"apiKey": api_key},
        timeout=20,
    )
    r.raise_for_status()

    sports = r.json()

    return [
        s for s in sports
        if "soccer" in s.get("key", "").lower()
        or "soccer" in s.get("group", "").lower()
    ]


def get_odds(sport_key: str) -> List[Dict[str, Any]]:
    api_key = os.getenv("ODDS_API_KEY")

    if not api_key:
        return [{"error": "Set ODDS_API_KEY in .env to use The Odds API."}]

    regions = os.getenv("ODDS_REGIONS", "eu,uk")

    r = requests.get(
        f"{THE_ODDS_API_BASE}/sports/{sport_key}/odds",
        params={
            "apiKey": api_key,
            "regions": regions,
            "markets": "h2h",
            "oddsFormat": "decimal",
        },
        timeout=20,
    )
    r.raise_for_status()

    return r.json()


def _norm_team(name: str) -> str:
    return str(name or "").strip().lower()


def find_odds_for_match(home_team: str, away_team: str, sport_key: str):
    events = get_odds(sport_key)

    searched_home = _norm_team(home_team)
    searched_away = _norm_team(away_team)

    for event in events:
        event_home = event.get("home_team")
        event_away = event.get("away_team")

        event_home_norm = _norm_team(event_home)
        event_away_norm = _norm_team(event_away)

        direct_match = (
            event_home_norm == searched_home
            and event_away_norm == searched_away
        )

        reversed_match = (
            event_home_norm == searched_away
            and event_away_norm == searched_home
        )

        if direct_match or reversed_match:
            outcomes = []

            for bookmaker in event.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    if market.get("key") != "h2h":
                        continue

                    for outcome in market.get("outcomes", []):
                        outcomes.append({
                            "bookmaker": bookmaker.get("title"),
                            "name": outcome.get("name"),
                            "price": outcome.get("price"),
                            "last_update": market.get("last_update"),
                        })

            return {
                "found": True,
                "matched": "direct" if direct_match else "reversed",
                "event": {
                    "commence_time": event.get("commence_time"),
                    "home_team": event_home,
                    "away_team": event_away,
                },
                "outcomes": outcomes,
            }

    sample_events = [
        {
            "commence_time": event.get("commence_time"),
            "home_team": event.get("home_team"),
            "away_team": event.get("away_team"),
            "teams": event.get("teams", []),
        }
        for event in events[:72]
    ]

    return {
        "found": False,
        "events_checked": len(events),
        "searched_home": home_team,
        "searched_away": away_team,
        "available_events": sample_events,
    }