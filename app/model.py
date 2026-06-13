from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
import json
import math

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
STATE_PATH = ARTIFACT_DIR / "team_state.json"
DATA_PATH = BASE_DIR / "data" / "matches.csv"

RESULT_CLASS_ORDER = ["home_win", "draw", "away_win"]


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def standardize_match_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Accepts Kaggle international-results style CSV or football-data.co.uk style CSV.

    Required canonical columns after this function:
    date, home_team, away_team, home_score, away_score, tournament, city, country, neutral,
    odds_home, odds_draw, odds_away.
    """
    raw = df.copy()
    raw.columns = [c.strip() for c in raw.columns]
    lower_map = {c.lower(): c for c in raw.columns}

    # Kaggle / martj42 international-results format
    if {"date", "home_team", "away_team", "home_score", "away_score"}.issubset(lower_map):
        out = pd.DataFrame()
        out["date"] = raw[lower_map["date"]]
        out["home_team"] = raw[lower_map["home_team"]]
        out["away_team"] = raw[lower_map["away_team"]]
        out["home_score"] = raw[lower_map["home_score"]]
        out["away_score"] = raw[lower_map["away_score"]]
        out["tournament"] = raw[lower_map.get("tournament", raw.columns[0])] if "tournament" in lower_map else "Unknown"
        out["city"] = raw[lower_map.get("city", raw.columns[0])] if "city" in lower_map else ""
        out["country"] = raw[lower_map.get("country", raw.columns[0])] if "country" in lower_map else ""
        if "neutral" in lower_map:
            out["neutral"] = raw[lower_map["neutral"]].astype(str).str.lower().isin(["true", "1", "yes", "y"])
        else:
            out["neutral"] = False
        out["odds_home"] = np.nan
        out["odds_draw"] = np.nan
        out["odds_away"] = np.nan
        return clean_matches(out)

    # football-data.co.uk format, e.g. Date, HomeTeam, AwayTeam, FTHG, FTAG, B365H/B365D/B365A
    variants = {
        "date": ["date"],
        "home_team": ["hometeam", "home"],
        "away_team": ["awayteam", "away"],
        "home_score": ["fthg", "hg", "homescore"],
        "away_score": ["ftag", "ag", "awayscore"],
    }

    selected = {}
    for canonical, possible in variants.items():
        for p in possible:
            if p in lower_map:
                selected[canonical] = lower_map[p]
                break
    if len(selected) == 5:
        out = pd.DataFrame()
        for k, source_col in selected.items():
            out[k] = raw[source_col]
        out["tournament"] = raw[lower_map.get("div", raw.columns[0])] if "div" in lower_map else "League"
        out["city"] = ""
        out["country"] = ""
        out["neutral"] = False

        def first_col(names: List[str]) -> Optional[str]:
            for name in names:
                if name.lower() in lower_map:
                    return lower_map[name.lower()]
            return None

        h = first_col(["B365H", "AvgH", "MaxH", "PSH"])
        d = first_col(["B365D", "AvgD", "MaxD", "PSD"])
        a = first_col(["B365A", "AvgA", "MaxA", "PSA"])
        out["odds_home"] = raw[h] if h else np.nan
        out["odds_draw"] = raw[d] if d else np.nan
        out["odds_away"] = raw[a] if a else np.nan
        return clean_matches(out)

    raise ValueError(
        "Unknown CSV format. Expected either international-results columns "
        "date, home_team, away_team, home_score, away_score or football-data.co.uk columns "
        "Date, HomeTeam, AwayTeam, FTHG, FTAG."
    )


def clean_matches(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    parsed_dates = pd.to_datetime(out["date"], errors="coerce")
    missing_dates = parsed_dates.isna()
    if missing_dates.any():
        parsed_dates.loc[missing_dates] = pd.to_datetime(out.loc[missing_dates, "date"], errors="coerce", dayfirst=True)
    out["date"] = parsed_dates
    out = out.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    out["home_team"] = out["home_team"].astype(str).str.strip()
    out["away_team"] = out["away_team"].astype(str).str.strip()
    out["home_score"] = pd.to_numeric(out["home_score"], errors="coerce")
    out["away_score"] = pd.to_numeric(out["away_score"], errors="coerce")
    out = out.dropna(subset=["home_score", "away_score"])
    out["home_score"] = out["home_score"].astype(int)
    out["away_score"] = out["away_score"].astype(int)
    for c in ["tournament", "city", "country"]:
        if c not in out:
            out[c] = ""
        out[c] = out[c].fillna("").astype(str)
    if "neutral" not in out:
        out["neutral"] = False
    out["neutral"] = out["neutral"].astype(str).str.lower().isin(["true", "1", "yes", "y"])
    for c in ["odds_home", "odds_draw", "odds_away"]:
        if c not in out:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.sort_values("date").reset_index(drop=True)
    return out


@dataclass
class RollingTeamStats:
    elo: Dict[str, float] = field(default_factory=lambda: defaultdict(lambda: 1500.0))
    last_matches: Dict[str, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=20)))
    last_date: Dict[str, pd.Timestamp] = field(default_factory=dict)

    def snapshot(self, team: str, date: pd.Timestamp) -> Dict[str, float]:
        matches = list(self.last_matches[team])
        last5 = matches[-5:]
        last10 = matches[-10:]

        def avg(items, key, default=0.0):
            if not items:
                return default
            return float(np.mean([m[key] for m in items]))

        days_rest = np.nan
        if team in self.last_date:
            days_rest = float((date - self.last_date[team]).days)
            # Cap because long breaks between national-team windows distort ML too much.
            days_rest = min(days_rest, 120.0)

        return {
            "elo": float(self.elo[team]),
            "points_last5": avg(last5, "points"),
            "goals_for_last5": avg(last5, "gf"),
            "goals_against_last5": avg(last5, "ga"),
            "goal_diff_last5": avg(last5, "gd"),
            "opponent_elo_last5": avg(last5, "opp_elo", 1500.0),
            "points_last10": avg(last10, "points"),
            "goal_diff_last10": avg(last10, "gd"),
            "days_rest": days_rest,
            "matches_known": float(len(matches)),
        }

    def update(self, home: str, away: str, hs: int, as_: int, date: pd.Timestamp, neutral: bool) -> None:
        home_elo_before = self.elo[home]
        away_elo_before = self.elo[away]
        if hs > as_:
            home_score, away_score = 1.0, 0.0
            hp, ap = 3, 0
        elif hs < as_:
            home_score, away_score = 0.0, 1.0
            hp, ap = 0, 3
        else:
            home_score, away_score = 0.5, 0.5
            hp, ap = 1, 1

        home_adv = 0.0 if neutral else 55.0
        expected_home = 1 / (1 + 10 ** (-(home_elo_before + home_adv - away_elo_before) / 400))
        expected_away = 1 - expected_home
        margin = abs(hs - as_)
        k = 22 if margin <= 1 else 28 if margin == 2 else 34
        self.elo[home] = home_elo_before + k * (home_score - expected_home)
        self.elo[away] = away_elo_before + k * (away_score - expected_away)

        self.last_matches[home].append({
            "points": hp, "gf": hs, "ga": as_, "gd": hs - as_, "opp_elo": away_elo_before
        })
        self.last_matches[away].append({
            "points": ap, "gf": as_, "ga": hs, "gd": as_ - hs, "opp_elo": home_elo_before
        })
        self.last_date[home] = date
        self.last_date[away] = date

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "elo": dict(self.elo),
            "last_matches": {k: list(v) for k, v in self.last_matches.items()},
            "last_date": {k: str(v.date()) for k, v in self.last_date.items()},
        }

    @classmethod
    def from_jsonable(cls, data: Dict[str, Any]) -> "RollingTeamStats":
        obj = cls()
        obj.elo = defaultdict(lambda: 1500.0, {k: float(v) for k, v in data.get("elo", {}).items()})
        obj.last_matches = defaultdict(lambda: deque(maxlen=20))
        for k, items in data.get("last_matches", {}).items():
            obj.last_matches[k] = deque(items, maxlen=20)
        obj.last_date = {k: pd.to_datetime(v) for k, v in data.get("last_date", {}).items()}
        return obj


FEATURE_COLUMNS = [
    "home_elo", "away_elo", "elo_diff",
    "home_points_last5", "away_points_last5", "points_last5_diff",
    "home_gf_last5", "away_gf_last5", "home_ga_last5", "away_ga_last5",
    "home_gd_last5", "away_gd_last5", "gd_last5_diff",
    "home_opp_elo_last5", "away_opp_elo_last5", "opp_elo_last5_diff",
    "home_points_last10", "away_points_last10", "points_last10_diff",
    "home_gd_last10", "away_gd_last10", "gd_last10_diff",
    "home_days_rest", "away_days_rest", "rest_diff",
    "home_matches_known", "away_matches_known",
    "is_neutral", "host_home", "is_world_cup", "is_friendly",
    "odds_imp_home", "odds_imp_draw", "odds_imp_away",
    "market_margin",
]


def odds_to_implied(oh: Any, od: Any, oa: Any) -> Tuple[float, float, float, float]:
    oh, od, oa = _safe_float(oh), _safe_float(od), _safe_float(oa)
    if np.isnan(oh) or np.isnan(od) or np.isnan(oa) or min(oh, od, oa) <= 1.0:
        return np.nan, np.nan, np.nan, np.nan
    inv = np.array([1 / oh, 1 / od, 1 / oa], dtype=float)
    margin = float(inv.sum() - 1)
    normalized = inv / inv.sum()
    return float(normalized[0]), float(normalized[1]), float(normalized[2]), margin


def make_feature_row(
    stats: RollingTeamStats,
    date: pd.Timestamp,
    home: str,
    away: str,
    country: str = "",
    tournament: str = "",
    neutral: bool = False,
    odds_home: Any = np.nan,
    odds_draw: Any = np.nan,
    odds_away: Any = np.nan,
) -> Dict[str, float]:
    h = stats.snapshot(home, date)
    a = stats.snapshot(away, date)
    oh, od, oa, margin = odds_to_implied(odds_home, odds_draw, odds_away)
    tournament_l = str(tournament).lower()
    return {
        "home_elo": h["elo"],
        "away_elo": a["elo"],
        "elo_diff": h["elo"] - a["elo"],
        "home_points_last5": h["points_last5"],
        "away_points_last5": a["points_last5"],
        "points_last5_diff": h["points_last5"] - a["points_last5"],
        "home_gf_last5": h["goals_for_last5"],
        "away_gf_last5": a["goals_for_last5"],
        "home_ga_last5": h["goals_against_last5"],
        "away_ga_last5": a["goals_against_last5"],
        "home_gd_last5": h["goal_diff_last5"],
        "away_gd_last5": a["goal_diff_last5"],
        "gd_last5_diff": h["goal_diff_last5"] - a["goal_diff_last5"],
        "home_opp_elo_last5": h["opponent_elo_last5"],
        "away_opp_elo_last5": a["opponent_elo_last5"],
        "opp_elo_last5_diff": h["opponent_elo_last5"] - a["opponent_elo_last5"],
        "home_points_last10": h["points_last10"],
        "away_points_last10": a["points_last10"],
        "points_last10_diff": h["points_last10"] - a["points_last10"],
        "home_gd_last10": h["goal_diff_last10"],
        "away_gd_last10": a["goal_diff_last10"],
        "gd_last10_diff": h["goal_diff_last10"] - a["goal_diff_last10"],
        "home_days_rest": h["days_rest"],
        "away_days_rest": a["days_rest"],
        "rest_diff": (h["days_rest"] if not np.isnan(h["days_rest"]) else np.nan) - (a["days_rest"] if not np.isnan(a["days_rest"]) else np.nan),
        "home_matches_known": h["matches_known"],
        "away_matches_known": a["matches_known"],
        "is_neutral": 1.0 if neutral else 0.0,
        "host_home": 1.0 if (not neutral and country and str(country).strip().lower() == home.strip().lower()) else 0.0,
        "is_world_cup": 1.0 if "world cup" in tournament_l or "world" in tournament_l else 0.0,
        "is_friendly": 1.0 if "friendly" in tournament_l else 0.0,
        "odds_imp_home": oh,
        "odds_imp_draw": od,
        "odds_imp_away": oa,
        "market_margin": margin,
    }


def build_training_matrix(matches: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, RollingTeamStats]:
    stats = RollingTeamStats()
    features: List[Dict[str, float]] = []
    labels: List[str] = []

    for _, row in matches.sort_values("date").iterrows():
        date = pd.to_datetime(row["date"])
        home = str(row["home_team"])
        away = str(row["away_team"])
        feature = make_feature_row(
            stats=stats,
            date=date,
            home=home,
            away=away,
            country=str(row.get("country", "")),
            tournament=str(row.get("tournament", "")),
            neutral=bool(row.get("neutral", False)),
            odds_home=row.get("odds_home", np.nan),
            odds_draw=row.get("odds_draw", np.nan),
            odds_away=row.get("odds_away", np.nan),
        )
        hs, aas = int(row["home_score"]), int(row["away_score"])
        if hs > aas:
            label = "home_win"
        elif hs < aas:
            label = "away_win"
        else:
            label = "draw"
        features.append(feature)
        labels.append(label)
        stats.update(home, away, hs, aas, date, bool(row.get("neutral", False)))

    X = pd.DataFrame(features)[FEATURE_COLUMNS]
    y = pd.Series(labels)
    return X, y, stats


def train_from_csv(csv_path: Path = DATA_PATH) -> Dict[str, Any]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    matches = standardize_match_csv(df)
    if len(matches) < 200:
        raise ValueError("Need at least ~200 historical matches for a useful first model.")

    X, y, stats = build_training_matrix(matches)
    # Keep chronological-ish split by using shuffle=False. Recent data becomes validation set.
    split_index = int(len(X) * 0.82)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    labels = list(model.classes_)
    result = {
        "rows_used": int(len(matches)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "log_loss": float(log_loss(y_test, proba, labels=labels)),
        "classes": labels,
        "model_path": str(MODEL_PATH),
        "note": "Accuracy is only a rough metric. For betting, track calibration and closing-line value too.",
    }
    joblib.dump({"model": model, "features": FEATURE_COLUMNS, "classes": labels}, MODEL_PATH)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(stats.to_jsonable(), f, ensure_ascii=False, indent=2)
    matches.to_csv(DATA_PATH, index=False)
    return result


def load_model_and_state() -> Tuple[Any, RollingTeamStats]:
    if not MODEL_PATH.exists() or not STATE_PATH.exists():
        raise FileNotFoundError("Model is not trained yet. Upload CSV and run /train first.")
    bundle = joblib.load(MODEL_PATH)
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        stats = RollingTeamStats.from_jsonable(json.load(f))
    return bundle, stats


def predict_match(
    home_team: str,
    away_team: str,
    match_date: Optional[str] = None,
    country: str = "",
    city: str = "",
    neutral: bool = False,
    tournament: str = "",
    odds_home: Optional[float] = None,
    odds_draw: Optional[float] = None,
    odds_away: Optional[float] = None,
) -> Dict[str, Any]:
    bundle, stats = load_model_and_state()
    model = bundle["model"]
    date = pd.to_datetime(match_date) if match_date else pd.Timestamp.today()

    def predict_single(
        first_team: str,
        second_team: str,
        first_odds: Optional[float],
        draw_odds: Optional[float],
        second_odds: Optional[float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        feature = make_feature_row(
            stats=stats,
            date=date,
            home=first_team,
            away=second_team,
            country=country,
            tournament=tournament,
            neutral=neutral,
            odds_home=first_odds,
            odds_draw=draw_odds,
            odds_away=second_odds,
        )

        X = pd.DataFrame([feature])[FEATURE_COLUMNS]
        proba = model.predict_proba(X)[0]
        classes = list(model.classes_)

        probs = {c: float(proba[i]) for i, c in enumerate(classes)}
        for c in RESULT_CLASS_ORDER:
            probs.setdefault(c, 0.0)

        probs = {c: probs[c] for c in RESULT_CLASS_ORDER}
        return probs, feature

    normal_probs, feature = predict_single(
        home_team,
        away_team,
        odds_home,
        odds_draw,
        odds_away,
    )

    if neutral:
        reversed_probs, _ = predict_single(
            away_team,
            home_team,
            odds_away,
            odds_draw,
            odds_home,
        )

        probs = {
            "home_win": (normal_probs["home_win"] + reversed_probs["away_win"]) / 2,
            "draw": (normal_probs["draw"] + reversed_probs["draw"]) / 2,
            "away_win": (normal_probs["away_win"] + reversed_probs["home_win"]) / 2,
        }

        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}
    else:
        probs = normal_probs

    suggested = max(probs, key=probs.get)
    fair = {c: (round(1 / p, 2) if p > 0 else None) for c, p in probs.items()}
    value_flags = value_detection(probs, odds_home, odds_draw, odds_away)
    explanation = explain_prediction(home_team, away_team, feature, probs, value_flags)

    if neutral:
        explanation += " Mecz oznaczono jako neutralny, więc predykcja została policzona symetrycznie w obu ustawieniach drużyn i uśredniona."

    return {
        "match": {
            "home_team": home_team,
            "away_team": away_team,
            "date": str(date.date()),
            "city": city,
            "country": country,
            "neutral": neutral,
            "tournament": tournament,
        },
        "probabilities": {k: round(v, 4) for k, v in probs.items()},
        "suggested_result": suggested,
        "fair_odds": fair,
        "value_flags": value_flags,
        "feature_snapshot": {k: (None if pd.isna(v) else round(float(v), 4)) for k, v in feature.items()},
        "explanation": explanation,
    }


def value_detection(probs: Dict[str, float], odds_home: Any, odds_draw: Any, odds_away: Any) -> Dict[str, Any]:
    offered = {"home_win": odds_home, "draw": odds_draw, "away_win": odds_away}
    flags = {}
    for key, odd in offered.items():
        odd_f = _safe_float(odd)
        if np.isnan(odd_f) or odd_f <= 1:
            flags[key] = {"has_odds": False}
            continue
        model_prob = probs[key]
        implied = 1 / odd_f
        edge = model_prob - implied
        flags[key] = {
            "has_odds": True,
            "bookmaker_odds": odd_f,
            "bookmaker_implied_probability_raw": round(implied, 4),
            "model_probability": round(model_prob, 4),
            "edge": round(edge, 4),
            "is_potential_value": bool(edge > 0.03),
        }
    return flags


def explain_prediction(home: str, away: str, f: Dict[str, float], probs: Dict[str, float], value_flags: Dict[str, Any]) -> str:
    parts = []
    elo_diff = f["elo_diff"]
    gd_diff = f["gd_last5_diff"]
    opp_diff = f["opp_elo_last5_diff"]
    parts.append(f"Model daje: {home} {probs['home_win']:.1%}, remis {probs['draw']:.1%}, {away} {probs['away_win']:.1%}.")
    if abs(elo_diff) > 60:
        leader = home if elo_diff > 0 else away
        parts.append(f"Największy sygnał to przewaga Elo po stronie {leader} ({elo_diff:+.0f}).")
    if abs(gd_diff) > 0.35:
        leader = home if gd_diff > 0 else away
        parts.append(f"Forma z ostatnich 5 meczów bramkowo wygląda lepiej dla {leader} ({gd_diff:+.2f} różnicy średniego bilansu).")
    if abs(opp_diff) > 50:
        leader = home if opp_diff > 0 else away
        parts.append(f"Ostatni rywale byli trudniejsi po stronie {leader} ({opp_diff:+.0f} Elo).")
    if f["host_home"] == 1:
        parts.append(f"Dochodzi realny atut gospodarza: {home} gra u siebie.")
    potentials = [k for k, v in value_flags.items() if isinstance(v, dict) and v.get("is_potential_value")]
    if potentials:
        names = {"home_win": home, "draw": "remis", "away_win": away}
        parts.append("Według kursów potencjalne value widzę przy: " + ", ".join(names[p] for p in potentials) + ".")
    return " ".join(parts)


def append_result_and_refresh_state(result: Dict[str, Any]) -> Dict[str, Any]:
    # Appends one new match to data/matches.csv and refreshes rolling state. Model is not retrained automatically.
    row = pd.DataFrame([result])
    row = standardize_match_csv(row)
    if DATA_PATH.exists():
        existing = pd.read_csv(DATA_PATH)
        existing = standardize_match_csv(existing)
        combined = pd.concat([existing, row], ignore_index=True).drop_duplicates(
            subset=["date", "home_team", "away_team", "home_score", "away_score"], keep="last"
        )
    else:
        combined = row
    combined = combined.sort_values("date")
    combined.to_csv(DATA_PATH, index=False)
    # Rebuild state, but do not retrain every match unless user asks.
    _, _, stats = build_training_matrix(combined)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(stats.to_jsonable(), f, ensure_ascii=False, indent=2)
    return {"rows_now": int(len(combined)), "state_refreshed": True, "retrain_recommended": len(combined) % 25 == 0}


def team_form(team: str, limit: int = 10) -> Dict[str, Any]:
    if not DATA_PATH.exists():
        raise FileNotFoundError("No match data yet.")
    df = standardize_match_csv(pd.read_csv(DATA_PATH))
    mask = (df["home_team"].str.lower() == team.lower()) | (df["away_team"].str.lower() == team.lower())
    team_df = df.loc[mask].sort_values("date").tail(limit)
    matches = []
    for _, r in team_df.iterrows():
        is_home = r["home_team"].lower() == team.lower()
        gf = int(r["home_score"] if is_home else r["away_score"])
        ga = int(r["away_score"] if is_home else r["home_score"])
        opponent = r["away_team"] if is_home else r["home_team"]
        result = "W" if gf > ga else "L" if gf < ga else "D"
        matches.append({
            "date": str(pd.to_datetime(r["date"]).date()),
            "opponent": opponent,
            "score": f"{gf}:{ga}",
            "result": result,
            "tournament": r.get("tournament", ""),
            "country": r.get("country", ""),
            "neutral": bool(r.get("neutral", False)),
        })
    return {"team": team, "last_matches": matches}
