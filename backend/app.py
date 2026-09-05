import sys
import os
import json
from datetime import datetime
from typing import Optional, List, Dict, Any

# Ensure AI_gaming root directory is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fastapi import FastAPI, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.database import db
from backend.models import UserRegister, UserLogin, FormationCreate, MatchRecord, TrainStepRequest, BCTrainRequest
from backend.auth import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user,
    get_optional_user
)

app = FastAPI(
    title="RL Football 2D Game Suite",
    description="Backend API and live match telemetry gateway for Human vs RL Learning AI Football.",
    version="6.0.0"
)

# CORS configurations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------
# LIVE TELEMETRY WEBSOCKET BROADCASTER
# -------------------------------------------------------------
class LiveMatchBroadcaster:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        dead_connections = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                dead_connections.append(connection)
        for conn in dead_connections:
            if conn in self.active_connections:
                self.active_connections.remove(conn)

broadcaster = LiveMatchBroadcaster()

@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    await broadcaster.connect(websocket)
    try:
        while True:
            data_text = await websocket.receive_text()
            try:
                data = json.loads(data_text)
                # If message is from Pygame engine, rebroadcast to all web clients
                if data.get("type") in ["match_tick", "match_event", "goal", "full_time"]:
                    await broadcaster.broadcast(data)
            except Exception:
                pass
    except WebSocketDisconnect:
        broadcaster.disconnect(websocket)

@app.post("/api/live/broadcast")
async def broadcast_tick(payload: Dict[str, Any]):
    """Allow internal engine to send live updates via HTTP if WebSocket is not used."""
    await broadcaster.broadcast(payload)
    return {"status": "ok"}

# -------------------------------------------------------------
# ROOT STATUS
# -------------------------------------------------------------
@app.get("/")
def read_root():
    ai_stats = db.get_ai_stats()
    match_summary = db.get_match_summary()
    return {
        "status": "online",
        "service": "2D Football RL Engine API",
        "version": "6.0.0",
        "ai_model": "PPO Actor-Critic + Spatial GNN",
        "total_matches_recorded": match_summary.get("total_matches", 0),
        "ai_episodes_trained": ai_stats.get("episodes_trained", 0)
    }

# -------------------------------------------------------------
# AUTHENTICATION ENDPOINTS
# -------------------------------------------------------------
@app.post("/api/auth/register")
async def register(req: UserRegister):
    existing = db.get_user_by_username(req.username)
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists. Please pick another.")

    pwd_hash = hash_password(req.password)
    try:
        user_id = db.create_user(
            username=req.username,
            email=req.email,
            password_hash=pwd_hash,
            full_name=req.full_name or "",
            role=req.role or "Player / Coach"
        )
        return {"success": True, "message": "Account created successfully.", "user_id": user_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Registration failed: {str(e)}")

@app.post("/api/auth/login")
async def login(req: UserLogin):
    user = db.get_user_by_username(req.username)
    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    token = create_access_token({"sub": user["username"]})
    user_sanitized = {k: v for k, v in user.items() if k != "password_hash"}
    return {
        "success": True,
        "access_token": token,
        "token_type": "bearer",
        "user": user_sanitized
    }

@app.get("/api/auth/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    user_sanitized = {k: v for k, v in current_user.items() if k != "password_hash"}
    return {"success": True, "user": user_sanitized}

# -------------------------------------------------------------
# FORMATIONS ENDPOINTS
# -------------------------------------------------------------
@app.get("/api/formations")
async def list_formations(current_user: Optional[dict] = Depends(get_optional_user)):
    user_id = current_user["id"] if current_user else None
    formations = db.get_formations(user_id=user_id)
    return {"success": True, "formations": formations}

@app.post("/api/formations")
async def save_formation(req: FormationCreate, current_user: Optional[dict] = Depends(get_optional_user)):
    user_id = current_user["id"] if current_user else None
    fid = db.save_formation(
        name=req.name,
        formation_type=req.formation_type,
        coordinates=req.coordinates,
        user_id=user_id
    )
    return {"success": True, "message": f"Formation '{req.name}' saved.", "formation_id": fid}

@app.delete("/api/formations/{formation_id}")
async def delete_formation(formation_id: int, current_user: Optional[dict] = Depends(get_optional_user)):
    user_id = current_user["id"] if current_user else None
    deleted = db.delete_formation(formation_id, user_id=user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Formation not found or cannot delete default presets.")
    return {"success": True, "message": "Formation deleted."}

# -------------------------------------------------------------
# MATCHES ENDPOINTS
# -------------------------------------------------------------
@app.post("/api/matches/record")
async def record_match(req: MatchRecord, current_user: Optional[dict] = Depends(get_optional_user)):
    user_id = current_user["id"] if current_user else None
    match_id = db.record_match(
        match_type=req.match_type,
        human_score=req.human_score,
        ai_score=req.ai_score,
        human_formation=req.human_formation,
        ai_formation=req.ai_formation,
        possession_human=req.possession_human,
        possession_ai=req.possession_ai,
        shots_human=req.shots_human,
        shots_ai=req.shots_ai,
        passes_human=req.passes_human,
        passes_ai=req.passes_ai,
        result=req.result,
        ai_reward=req.ai_reward,
        ai_loss=req.ai_loss,
        duration_seconds=req.duration_seconds,
        user_id=user_id,
        xg_human=req.xg_human or 0.0,
        xg_ai=req.xg_ai or 0.0,
        tactical_style=req.tactical_style or "Balanced",
        shots_data=req.shots_data,
        ai_adaptation=req.ai_adaptation
    )
    
    # Broadcast full-time result to web clients
    await broadcaster.broadcast({
        "type": "match_recorded",
        "match_id": match_id,
        "human_score": req.human_score,
        "ai_score": req.ai_score,
        "result": req.result,
        "xg_human": req.xg_human,
        "xg_ai": req.xg_ai,
        "tactical_style": req.tactical_style
    })

    return {"success": True, "message": "Match result saved to database.", "match_id": match_id}

@app.get("/api/matches/history")
async def get_match_history(limit: int = 20):
    matches = db.get_match_history(limit=limit)
    return {"success": True, "matches": matches}

@app.get("/api/matches/summary")
async def get_match_summary():
    summary = db.get_match_summary()
    return {"success": True, "summary": summary}

# -------------------------------------------------------------
# AI & RL EVOLUTION ENDPOINTS
# -------------------------------------------------------------
@app.get("/api/ai/stats")
async def get_ai_stats():
    stats = db.get_ai_stats()
    return {"success": True, "stats": stats}

@app.get("/api/ai/history")
async def get_ai_history(limit: int = 50):
    history = db.get_ai_training_history(limit=limit)
    return {"success": True, "history": history}

@app.post("/api/ai/train-step")
async def trigger_train_step(req: TrainStepRequest):
    """Execute background PPO training iteration on the RL agent."""
    from rl_env.trainer import PPOTrainer
    trainer = PPOTrainer()
    res = trainer.train_step(num_episodes=req.episodes or 1)
    
    # Record in SQLite database
    db.record_ai_training(
        episode=res.get("episode", 1),
        reward=res.get("reward", 0.0),
        actor_loss=res.get("actor_loss", 0.0),
        critic_loss=res.get("critic_loss", 0.0),
        source="BACKGROUND"
    )
    
    # Broadcast training telemetry
    await broadcaster.broadcast({
        "type": "telemetry",
        "reward": res.get("reward", 0.0),
        "loss": res.get("loss", 0.0),
        "episode": res.get("episode", 1)
    })
    
    return {"success": True, "result": res}
 
@app.post("/api/ai/pretrain-bc")
async def trigger_pretrain_bc(req: Optional[BCTrainRequest] = None):
    """Execute offline Behavioral Cloning from expert demonstrations to warm-start policy."""
    from rl_env.behavioral_cloning import generate_expert_dataset, BCTrainer
    episodes = req.episodes if req and req.episodes else 6
    epochs = req.epochs if req and req.epochs else 3
    batch_size = req.batch_size if req and req.batch_size else 64
    lr = req.lr if req and req.lr else 1e-3

    obs, acts, vals = generate_expert_dataset(num_episodes=episodes, max_steps=180)
    trainer = BCTrainer()
    res = trainer.train(obs, acts, vals, epochs=epochs, batch_size=batch_size, lr=lr)

    # Record in SQLite database
    db.record_ai_training(
        episode=res.get("epochs", 1),
        reward=1.0,
        actor_loss=res.get("final_loss", 0.0),
        critic_loss=0.0,
        source="BEHAVIORAL_CLONING"
    )

    # Broadcast training telemetry
    await broadcaster.broadcast({
        "type": "bc_completed",
        "samples": res.get("samples", 0),
        "accuracy": res.get("accuracy", 0.0),
        "loss": res.get("final_loss", 0.0)
    })

    return {"success": True, "result": res}

@app.post("/api/ai/reset")
async def reset_ai():
    """Reset AI training statistics and re-initialize rookie policy weights."""
    db.reset_ai_stats()
    ckpt_path = os.path.join(BASE_DIR, "rl_env", "checkpoints", "ppo_gnn_model.pt")
    if os.path.exists(ckpt_path):
        try:
            os.remove(ckpt_path)
        except Exception:
            pass
    from rl_env.nn_brain import FootballActorCritic, TORCH_AVAILABLE
    if TORCH_AVAILABLE:
        rookie_brain = FootballActorCritic()
        rookie_brain.save_weights(ckpt_path)
    return {"success": True, "message": "AI neural policy reset to baseline rookie weights."}

@app.get("/api/ai/download-model")
async def download_model():
    ckpt_path = os.path.join(BASE_DIR, "rl_env", "checkpoints", "ppo_gnn_model.pt")
    if not os.path.exists(ckpt_path):
        from rl_env.nn_brain import FootballActorCritic, TORCH_AVAILABLE
        if TORCH_AVAILABLE:
            rookie_brain = FootballActorCritic()
            rookie_brain.save_weights(ckpt_path)
        else:
            raise HTTPException(status_code=404, detail="Model weights not found.")
    return FileResponse(
        path=ckpt_path,
        filename="ppo_gnn_model.pt",
        media_type="application/octet-stream"
    )

# -------------------------------------------------------------
# STATIC FILES SERVING (FRONTEND)
# -------------------------------------------------------------
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/dashboard", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
