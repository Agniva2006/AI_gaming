from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Any

class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=30)
    email: EmailStr
    password: str = Field(..., min_length=4)
    full_name: Optional[str] = ""
    role: Optional[str] = "Player / Coach"

class UserLogin(BaseModel):
    username: str
    password: str

class FormationCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    formation_type: str = Field(default="Custom")
    coordinates: List[List[float]] = Field(..., description="11 (x, y) normalized coordinates [0.0 - 1.0]")

class MatchRecord(BaseModel):
    match_type: Optional[str] = "HUMAN_VS_AI"
    human_score: int
    ai_score: int
    human_formation: Optional[str] = "4-3-3"
    ai_formation: Optional[str] = "4-4-2"
    possession_human: Optional[float] = 50.0
    possession_ai: Optional[float] = 50.0
    shots_human: Optional[int] = 0
    shots_ai: Optional[int] = 0
    passes_human: Optional[int] = 0
    passes_ai: Optional[int] = 0
    result: Optional[str] = "DRAW"
    ai_reward: Optional[float] = 0.0
    ai_loss: Optional[float] = 0.0
    duration_seconds: Optional[float] = 180.0
    xg_human: Optional[float] = 0.0
    xg_ai: Optional[float] = 0.0
    tactical_style: Optional[str] = "Balanced"
    shots_data: Optional[List[Dict[str, Any]]] = None
    ai_adaptation: Optional[Dict[str, Any]] = None

class TrainStepRequest(BaseModel):
    episodes: Optional[int] = 1

class BCTrainRequest(BaseModel):
    episodes: Optional[int] = 8
    epochs: Optional[int] = 3
    batch_size: Optional[int] = 64
    lr: Optional[float] = 1e-3

