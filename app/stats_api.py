from __future__ import annotations

import os
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import requests

API_FOOTBALL_BASE = "https://v3.football.api-sports.io"


def has_api_football_key() -> bool:
    return bool(os.getenv("API_FOOTBALL_KEY"))


def _norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name or ""))
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def _num(value: Any) -> float:
    if value is None:
        return 0.0

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        value = value.replace("%", "").strip()
        if not value:
            return 0.0

        try:
            return float(value)
        except ValueError:
            return 0.0

    return 0.0


def _api_get(path: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    api_key = os.getenv("API_FOOTBALL_KEY")

    if not api_key:
        raise RuntimeError("Set API_FOOTBALL_KEY in .env to use API-Football stats.")

    r = requests.get(
        f"{API_FOOTBALL_BASE}{path}",
        headers={"x-apisports-key": api_key},
        params=params or {},
        timeout=25,
    )
    r.raise_for_status()

    data = r.json()

    if data.get("errors"):
        raise RuntimeError(str(data.get("errors")))

    return data.get("response", [])


def _teams_match(api_home: str, api_away: str, home: str, away: str) -> Tuple[bool, bool]:
    ah, aa = _norm(api_home), _norm(api_away)
    h, a = _norm(home), _norm(away)

    direct = ah == h and aa == a
    reversed_ = ah == a and aa == h

    return direct, reversed_


def find_fixture(home_team: str, away_team: str, match_date: str) -> Optional[Dict[str, Any]]:
    fixtures = _api_get("/fixtures", {"date": match_date})
    fallback: Optional[Dict[str, Any]] = None

    for fixture in fixtures:
        api_home = fixture.get("teams", {}).get("home", {}).get("name", "")
        api_away = fixture.get("teams", {}).get("away", {}).get("name", "")

        direct, reversed_ = _teams_match(api_home, api_away, home_team, away_team)

        if direct or reversed_:
            fixture["_matched"] = "direct" if direct else "reversed"
            return fixture

        names = {_norm(api_home), _norm(api_away)}
        wanted = {_norm(home_team), _norm(away_team)}

        if wanted.issubset(names):
            fallback = fixture

    return fallback


def _stat_value(stats: List[Dict[str, Any]], stat_type: str) -> float:
    wanted = stat_type.lower()

    for stat in stats:
        if str(stat.get("type", "")).lower() == wanted:
            return _num(stat.get("value"))

    return 0.0


def _fixture_stats(fixture_id: int) -> List[Dict[str, Any]]:
    return _api_get("/fixtures/statistics", {"fixture": fixture_id})


def recent_team_profile(
    team_id: int,
    team_name: str,
    last: int = 5,
    before_date: str = "",
) -> Dict[str, Any]:
    from datetime import datetime, timedelta

    last = max(1, min(int(last or 5), 10))

    if not team_id:
        return {
            "team_id": team_id,
            "team_name": team_name,
            "matches_used": 0,
            "averages": {},
            "sample_matches": [],
            "note": "Brak team_id z API-Football.",
        }

    if before_date:
        to_dt = datetime.fromisoformat(before_date[:10])
    else:
        to_dt = datetime.utcnow()

    from_dt = to_dt - timedelta(days=540)

    fixtures: List[Dict[str, Any]] = []

    seasons_to_try = [
        to_dt.year,
        to_dt.year - 1,
    ]

    for season in seasons_to_try:
        try:
            season_fixtures = _api_get(
                "/fixtures",
                {
                    "team": team_id,
                    "season": season,
                    "from": from_dt.date().isoformat(),
                    "to": to_dt.date().isoformat(),
                },
            )

            fixtures.extend(season_fixtures)

        except Exception:
            continue

    finished_statuses = {"FT", "AET", "PEN"}

    finished = []

    for fixture in fixtures:
        status = fixture.get("fixture", {}).get("status", {}).get("short")
        fx_date = fixture.get("fixture", {}).get("date", "")

        if status in finished_statuses and fx_date:
            finished.append(fixture)

    finished = sorted(
        finished,
        key=lambda f: f.get("fixture", {}).get("date", ""),
        reverse=True,
    )[:last]

    sums = {
        "corners_for": 0.0,
        "corners_against": 0.0,
        "shots_on_goal_for": 0.0,
        "shots_on_goal_against": 0.0,
        "total_shots_for": 0.0,
        "total_shots_against": 0.0,
        "yellow_for": 0.0,
        "yellow_against": 0.0,
        "red_for": 0.0,
        "red_against": 0.0,
    }

    used = 0
    sample = []

    for fixture in finished:
        fixture_id = fixture.get("fixture", {}).get("id")

        if not fixture_id:
            continue

        try:
            stats = _fixture_stats(fixture_id)
        except Exception:
            continue

        if len(stats) < 2:
            continue

        team_stats = None
        opponent_stats = None

        for item in stats:
            tid = item.get("team", {}).get("id")

            if tid == team_id:
                team_stats = item.get("statistics", [])
            else:
                opponent_stats = item.get("statistics", [])

        if team_stats is None or opponent_stats is None:
            continue

        sums["corners_for"] += _stat_value(team_stats, "Corner Kicks")
        sums["corners_against"] += _stat_value(opponent_stats, "Corner Kicks")

        sums["shots_on_goal_for"] += _stat_value(team_stats, "Shots on Goal")
        sums["shots_on_goal_against"] += _stat_value(opponent_stats, "Shots on Goal")

        sums["total_shots_for"] += _stat_value(team_stats, "Total Shots")
        sums["total_shots_against"] += _stat_value(opponent_stats, "Total Shots")

        sums["yellow_for"] += _stat_value(team_stats, "Yellow Cards")
        sums["yellow_against"] += _stat_value(opponent_stats, "Yellow Cards")

        sums["red_for"] += _stat_value(team_stats, "Red Cards")
        sums["red_against"] += _stat_value(opponent_stats, "Red Cards")

        used += 1

        sample.append({
            "date": fixture.get("fixture", {}).get("date"),
            "home": fixture.get("teams", {}).get("home", {}).get("name"),
            "away": fixture.get("teams", {}).get("away", {}).get("name"),
            "score": fixture.get("goals"),
        })

    if used == 0:
        return {
            "team_id": team_id,
            "team_name": team_name,
            "matches_used": 0,
            "averages": {},
            "sample_matches": sample,
            "note": "Nie udało się pobrać statystyk z ostatnich meczów. API może nie mieć danych statystycznych dla tych spotkań albo darmowy plan je ogranicza.",
        }

    avg = {k: round(v / used, 2) for k, v in sums.items()}

    avg["cards_for"] = round(avg["yellow_for"] + avg["red_for"], 2)
    avg["cards_against"] = round(avg["yellow_against"] + avg["red_against"], 2)
    avg["match_cards_total"] = round(avg["cards_for"] + avg["cards_against"], 2)
    avg["match_corners_total"] = round(avg["corners_for"] + avg["corners_against"], 2)

    return {
        "team_id": team_id,
        "team_name": team_name,
        "matches_used": used,
        "averages": avg,
        "sample_matches": sample,
    }


def match_stats_context(
    home_team: str,
    away_team: str,
    match_date: str,
    last: int = 5,
) -> Dict[str, Any]:
    if not has_api_football_key():
        return {
            "available": False,
            "source": "API-Football",
            "reason": "Brak API_FOOTBALL_KEY w .env. Sugestie będą tylko z modelu i kursów.",
        }

    if not match_date:
        return {
            "available": False,
            "source": "API-Football",
            "reason": "Podaj datę meczu, żeby znaleźć fixture w API-Football.",
        }

    fixture = find_fixture(home_team, away_team, match_date)

    if not fixture:
        return {
            "available": False,
            "source": "API-Football",
            "reason": "Nie znaleziono meczu w fixtures API-Football dla tej daty i nazw drużyn.",
        }

    fx = fixture.get("fixture", {})
    teams = fixture.get("teams", {})

    api_home = teams.get("home", {})
    api_away = teams.get("away", {})

    home_profile = recent_team_profile(
        api_home.get("id"),
        api_home.get("name", home_team),
        last=last,
        before_date=match_date,
    )

    away_profile = recent_team_profile(
        api_away.get("id"),
        api_away.get("name", away_team),
        last=last,
        before_date=match_date,
    )

    return {
        "available": True,
        "source": "API-Football",
        "fixture": {
            "id": fx.get("id"),
            "date": fx.get("date"),
            "referee": fx.get("referee"),
            "venue": fx.get("venue"),
            "status": fx.get("status"),
            "home_team": api_home.get("name"),
            "away_team": api_away.get("name"),
        },
        "last_matches_requested": max(1, min(int(last or 5), 10)),
        "home_profile": home_profile,
        "away_profile": away_profile,
    }