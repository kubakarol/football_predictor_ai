from __future__ import annotations

from pathlib import Path
from typing import Optional
import shutil

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .schemas import MatchPredictionRequest, MatchResultIn, TrainResponse, PredictionResponse
from .model import DATA_PATH, train_from_csv, predict_match, append_result_and_refresh_state, team_form
from .odds import find_odds_for_match, get_soccer_sports
from .bet_suggestions import build_bet_suggestions

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
app = FastAPI(title="Football AI Predictor", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.post("/upload")
def upload_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a CSV file.")
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DATA_PATH.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"saved_to": str(DATA_PATH), "next_step": "POST /train"}


@app.post("/train", response_model=TrainResponse)
def train():
    try:
        return train_from_csv(DATA_PATH)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
def predict(req: MatchPredictionRequest):
    try:
        payload = req.model_dump()

        include_external_stats = bool(payload.pop("include_external_stats", False))
        stats_last_matches = int(payload.pop("stats_last_matches", 5) or 5)

        result = predict_match(**payload)

        result["bet_suggestions"] = build_bet_suggestions(
            prediction=result,
            include_external_stats=include_external_stats,
            stats_last_matches=stats_last_matches,
        )

        return result

    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/result")
def add_result(result: MatchResultIn):
    try:
        return append_result_and_refresh_state(result.model_dump())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/team/{team}/form")
def form(team: str, limit: int = 10):
    try:
        return team_form(team, limit)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/odds/sports")
def odds_sports():
    return get_soccer_sports()


@app.get("/odds/match")
def odds_match(home_team: str, away_team: str, sport_key: str = "soccer_fifa_world_cup"):
    try:
        return find_odds_for_match(home_team, away_team, sport_key)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
