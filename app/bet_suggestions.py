from __future__ import annotations

from typing import Any, Dict, List, Optional

from .stats_api import match_stats_context


def _level(score: float) -> str:
    if score >= 0.75:
        return "green"
    if score >= 0.45:
        return "yellow"
    return "red"


def _round(x: Optional[float], n: int = 2) -> Optional[float]:
    try:
        return round(float(x), n)
    except Exception:
        return None


def _value_suggestions(prediction: Dict[str, Any]) -> List[Dict[str, Any]]:
    match = prediction.get("match", {})

    home = match.get("home_team", "gospodarz")
    away = match.get("away_team", "gość")

    names = {
        "home_win": home,
        "draw": "Remis",
        "away_win": away,
    }

    suggestions: List[Dict[str, Any]] = []

    for key, flag in prediction.get("value_flags", {}).items():
        if not isinstance(flag, dict) or not flag.get("has_odds"):
            continue

        edge = float(flag.get("edge") or 0)
        model_prob = float(flag.get("model_probability") or 0)
        odds = flag.get("bookmaker_odds")

        if edge > 0.03:
            suggestions.append({
                "level": _level(min(1.0, 0.45 + edge * 6 + model_prob * 0.2)),
                "market": "1X2 / wynik meczu",
                "pick": names.get(key, key),
                "odds": odds,
                "model_probability": round(model_prob, 4),
                "edge": round(edge, 4),
                "reason": f"Model daje większe prawdopodobieństwo niż wynika z kursu. Edge około {edge:.1%}.",
            })

        elif edge < -0.04:
            suggestions.append({
                "level": "red",
                "market": "1X2 / wynik meczu",
                "pick": f"raczej odpuść: {names.get(key, key)}",
                "odds": odds,
                "model_probability": round(model_prob, 4),
                "edge": round(edge, 4),
                "reason": "Kurs wygląda za niski względem prawdopodobieństwa modelu.",
            })

    has_actionable = any(s["level"] in ("green", "yellow") for s in suggestions)

    if not has_actionable:
        probs = prediction.get("probabilities", {})
        best_key = max(probs, key=probs.get) if probs else None

        if best_key:
            suggestions.append({
                "level": "yellow",
                "market": "bezpieczniejszy kierunek",
                "pick": names.get(best_key, best_key),
                "model_probability": probs.get(best_key),
                "reason": "To najbardziej prawdopodobny wynik według modelu, ale bez dodatniego value z kursu traktuj to ostrożnie.",
            })

    return suggestions


def _stats_suggestions(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not ctx.get("available"):
        return [{
            "level": "yellow",
            "market": "statystyki dodatkowe",
            "pick": "brak pełnej analizy rożnych/kartek/strzałów",
            "reason": ctx.get("reason", "Brak danych z API-Football."),
        }]

    home = ctx.get("fixture", {}).get("home_team", "Gospodarz")
    away = ctx.get("fixture", {}).get("away_team", "Gość")

    h = ctx.get("home_profile", {}).get("averages", {}) or {}
    a = ctx.get("away_profile", {}).get("averages", {}) or {}

    suggestions: List[Dict[str, Any]] = []

    home_corners = ((h.get("corners_for", 0) or 0) + (a.get("corners_against", 0) or 0)) / 2
    away_corners = ((a.get("corners_for", 0) or 0) + (h.get("corners_against", 0) or 0)) / 2
    total_corners = home_corners + away_corners

    if home_corners - away_corners >= 1.2:
        suggestions.append({
            "level": "green" if home_corners - away_corners >= 2.0 else "yellow",
            "market": "rożne",
            "pick": f"{home} więcej rzutów rożnych",
            "projection": {
                home: _round(home_corners),
                away: _round(away_corners),
            },
            "reason": "Projekcja rożnych jest po stronie gospodarza na podstawie ostatnich meczów obu drużyn.",
        })

    elif away_corners - home_corners >= 1.2:
        suggestions.append({
            "level": "green" if away_corners - home_corners >= 2.0 else "yellow",
            "market": "rożne",
            "pick": f"{away} więcej rzutów rożnych",
            "projection": {
                home: _round(home_corners),
                away: _round(away_corners),
            },
            "reason": "Projekcja rożnych jest po stronie gościa na podstawie ostatnich meczów obu drużyn.",
        })

    if total_corners >= 9.6:
        suggestions.append({
            "level": "yellow",
            "market": "suma rożnych",
            "pick": "Over 8.5 rożnych",
            "projection": _round(total_corners),
            "reason": "Łączna projekcja rożnych jest wysoka. Sprawdź jeszcze kurs i składy.",
        })

    elif total_corners <= 8.0:
        suggestions.append({
            "level": "yellow",
            "market": "suma rożnych",
            "pick": "Under 10.5 rożnych",
            "projection": _round(total_corners),
            "reason": "Łączna projekcja rożnych jest raczej niska.",
        })

    home_sog = ((h.get("shots_on_goal_for", 0) or 0) + (a.get("shots_on_goal_against", 0) or 0)) / 2
    away_sog = ((a.get("shots_on_goal_for", 0) or 0) + (h.get("shots_on_goal_against", 0) or 0)) / 2

    if home_sog - away_sog >= 1.0:
        suggestions.append({
            "level": "yellow",
            "market": "strzały celne",
            "pick": f"{home} więcej strzałów celnych",
            "projection": {
                home: _round(home_sog),
                away: _round(away_sog),
            },
            "reason": "Statystyki sugerują przewagę gospodarza w strzałach celnych.",
        })

    elif away_sog - home_sog >= 1.0:
        suggestions.append({
            "level": "yellow",
            "market": "strzały celne",
            "pick": f"{away} więcej strzałów celnych",
            "projection": {
                home: _round(home_sog),
                away: _round(away_sog),
            },
            "reason": "Statystyki sugerują przewagę gościa w strzałach celnych.",
        })

    home_match_cards = h.get("match_cards_total", 0) or 0
    away_match_cards = a.get("match_cards_total", 0) or 0
    projected_cards = (home_match_cards + away_match_cards) / 2

    if projected_cards <= 4.6:
        suggestions.append({
            "level": "yellow",
            "market": "kartki",
            "pick": "Under 5.5 kartek",
            "projection": _round(projected_cards),
            "reason": "Średnia kartek z ostatnich meczów obu drużyn jest poniżej linii 5.5. Ryzyko: presja meczu i styl sędziego.",
        })

    elif projected_cards >= 5.1:
        suggestions.append({
            "level": "yellow",
            "market": "kartki",
            "pick": "Over 3.5 / ostrożnie Over 4.5 kartek",
            "projection": _round(projected_cards),
            "reason": "Profil ostatnich spotkań sugeruje mecz kartkowy. Sprawdź sędziego i stawkę meczu.",
        })

    referee = ctx.get("fixture", {}).get("referee")

    if referee:
        suggestions.append({
            "level": "yellow",
            "market": "sędzia",
            "pick": f"Sędzia: {referee}",
            "reason": "API znalazło sędziego. W tej wersji pokazuję nazwisko, ale nie liczę jeszcze jego historycznej średniej kartek.",
        })

    if not suggestions:
        suggestions.append({
            "level": "yellow",
            "market": "statystyki",
            "pick": "brak mocnego sygnału",
            "reason": "Rożne, kartki i strzały nie dały wyraźnej przewagi żadnej stronie.",
        })

    return suggestions


def build_bet_suggestions(
    prediction: Dict[str, Any],
    include_external_stats: bool = False,
    stats_last_matches: int = 5,
) -> Dict[str, Any]:
    suggestions = _value_suggestions(prediction)

    stats_context: Dict[str, Any] = {
        "available": False,
        "reason": "Analiza statystyk zewnętrznych jest wyłączona. Zaznacz opcję w UI i ustaw API_FOOTBALL_KEY.",
    }

    if include_external_stats:
        match = prediction.get("match", {})

        try:
            stats_context = match_stats_context(
                home_team=match.get("home_team", ""),
                away_team=match.get("away_team", ""),
                match_date=match.get("date", ""),
                last=stats_last_matches,
            )

        except Exception as e:
            stats_context = {
                "available": False,
                "source": "API-Football",
                "reason": str(e),
            }

        suggestions.extend(_stats_suggestions(stats_context))

    order = {
        "green": 0,
        "yellow": 1,
        "red": 2,
    }

    suggestions = sorted(
        suggestions,
        key=lambda s: order.get(s.get("level"), 9),
    )

    return {
        "note": "To są sugestie analityczne, nie pewniaki. Stawiaj tylko wtedy, gdy kurs realnie daje value i masz własny limit ryzyka.",
        "suggestions": suggestions[:8],
        "stats_context": stats_context,
    }