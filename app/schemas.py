from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class MatchPredictionRequest(BaseModel):
    home_team: str = Field(..., examples=["Mexico"])
    away_team: str = Field(..., examples=["South Africa"])
    match_date: Optional[str] = Field(None, description="YYYY-MM-DD. If omitted, today is used.")
    country: Optional[str] = Field(None, description="Host country, e.g. Mexico")
    city: Optional[str] = None
    neutral: bool = False
    tournament: Optional[str] = Field(None, examples=["FIFA World Cup"])
    odds_home: Optional[float] = Field(None, gt=1.0)
    odds_draw: Optional[float] = Field(None, gt=1.0)
    odds_away: Optional[float] = Field(None, gt=1.0)


class MatchResultIn(BaseModel):
    date: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    tournament: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    neutral: bool = False


class TrainResponse(BaseModel):
    rows_used: int
    accuracy: float
    log_loss: float
    classes: List[str]
    model_path: str
    note: str


class PredictionResponse(BaseModel):
    match: Dict[str, Any]
    probabilities: Dict[str, float]
    suggested_result: str
    fair_odds: Dict[str, float]
    value_flags: Dict[str, Any]
    feature_snapshot: Dict[str, Any]
    explanation: str
