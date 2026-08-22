import sys
import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

# Ensure parent directory is in sys.path to resolve backend package imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Depends, HTTPException, status, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr
import asyncio
from rl_env.trainer import PPOTrainer

from backend.auth import (
    load_users,
    save_users,
    hash_password,
    verify_password,
    create_access_token,
    get_current_user,
    check_formation_quota,
    check_download_quota,
    PLANS
)
from backend.payment import router as payment_router

app = FastAPI(
    title="RL Train Football API Suite",
    description="SaaS Backend API wrapping the Dynamic GNN-PPO multi-agent reinforcement learning football engine.",
    version="5.0.0"
)

# CORS configurations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models schemas
class UserRegister(BaseModel):
    username: str
    email: EmailStr
    full_name: Optional[str] = ""
    role: Optional[str] = "student"
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class ProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None

class FormationSave(BaseModel):
    name: str
    coordinates: List[List[float]] = Field(..., description="Coordinates as 11 (x, y) tuples")

# Mount stripe billing router
app.include_router(payment_router)

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "RL Train Football Deep API Engine",
        "version": "5.0.0",
        "rl_policy": "Graph Attention Network (GAT) + PPO",
        "plans": list(PLANS.keys())
    }

# -------------------------------------------------------------
# TELEMETRY WEBSOCKET ENDPOINT
# -------------------------------------------------------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
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

manager = ConnectionManager()
trainer_instance = None
simulation_task = None

async def simulation_loop():
    global trainer_instance
    try:
        # Initialize trainer lazily
        if trainer_instance is None:
            trainer_instance = PPOTrainer()
            
        while True:
            if not manager.active_connections:
                await asyncio.sleep(1)
                continue
            
            # Run one train step (in thread to avoid blocking asyncio loop)
            # PPOTrainer.train_step is synchronous, so we run it in executor
            loop = asyncio.get_running_loop()
            step_info = await loop.run_in_executor(None, trainer_instance.train_step, 1)
            
            await manager.broadcast({
                "type": "telemetry",
                "reward": step_info["reward"],
                "loss": step_info["loss"],
                "realism": step_info["realism"]
            })
            # Delay to simulate 1 second per step or so
            await asyncio.sleep(1.0)
    except Exception as e:
        print("Simulation loop error:", e)

@app.on_event("startup")
async def startup_event():
    global simulation_task
    simulation_task = asyncio.create_task(simulation_loop())

@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming messages if needed
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# -------------------------------------------------------------
# AUTHENTICATION ENDPOINTS
# -------------------------------------------------------------
@app.post("/auth/register")
async def register(req: UserRegister):
    users = load_users()
    username = req.username.strip().lower()
    
    if username in users:
        raise HTTPException(status_code=400, detail="Username is already taken.")
    
    for u in users.values():
        if u.get("email") == req.email:
            raise HTTPException(status_code=400, detail="Email is already registered.")

    users[username] = {
        "username": username,
        "email": req.email,
        "password_hash": hash_password(req.password),
        "full_name": req.full_name,
        "role": req.role,
        "plan": "free",
        "stripe_customer_id": "",
        "stripe_subscription_id": "",
        "subscription_status": "",
        "saved_formations": {},
        "activity": [
            {
                "endpoint": "User account created",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        ]
    }
    save_users(users)
    return {"success": True, "message": "User registered successfully."}

@app.post("/auth/login")
async def login(req: UserLogin):
    users = load_users()
    username = req.username.strip().lower()
    
    if username not in users or not verify_password(req.password, users[username]["password_hash"]):
        raise HTTPException(status_code=400, detail="Invalid username or password.")
        
    u = users[username]
    token = create_access_token({"sub": username})
    
    # Log session connection
    activity = u.setdefault("activity", [])
    activity.append({
        "endpoint": "API JWT Token Generated (Sign In)",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })
    save_users(users)

    # Sanitize password hash from return payload
    user_data = u.copy()
    del user_data["password_hash"]

    return {
        "success": True,
        "access_token": token,
        "token_type": "bearer",
        "user": user_data
    }

@app.get("/auth/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    user_data = current_user.copy()
    if "password_hash" in user_data:
        del user_data["password_hash"]
    return {"success": True, "user": user_data}

@app.patch("/auth/profile/update")
async def update_profile(req: ProfileUpdate, current_user: dict = Depends(get_current_user)):
    users = load_users()
    username = current_user["username"]
    u = users[username]
    
    if req.full_name is not None:
        u["full_name"] = req.full_name
    if req.email is not None:
        u["email"] = req.email
        
    activity = u.setdefault("activity", [])
    activity.append({
        "endpoint": "Account settings profile updated",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })
    save_users(users)
    
    user_data = u.copy()
    del user_data["password_hash"]
    return {"success": True, "user": user_data}

@app.get("/auth/activity")
async def get_activity(current_user: dict = Depends(get_current_user)):
    return {"success": True, "activity": current_user.get("activity", [])[-20:]}

# -------------------------------------------------------------
# TACTICAL FORMATIONS ENDPOINTS
# -------------------------------------------------------------
@app.post("/tactics/formations")
async def save_formation(req: FormationSave, current_user: dict = Depends(get_current_user)):
    if not check_formation_quota(current_user):
        raise HTTPException(
            status_code=403,
            detail=f"You have reached the maximum custom formations limit for your {current_user.get('plan')} plan. Upgrade your tier to unlock more!"
        )
        
    users = load_users()
    username = current_user["username"]
    u = users[username]
    
    formations = u.setdefault("saved_formations", {})
    formations[req.name] = req.coordinates
    
    activity = u.setdefault("activity", [])
    activity.append({
        "endpoint": f"Saved custom tactical formation: {req.name}",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })
    save_users(users)
    return {"success": True, "message": f"Formation '{req.name}' successfully persisted."}

@app.get("/tactics/formations")
async def list_formations(current_user: dict = Depends(get_current_user)):
    return {"success": True, "formations": current_user.get("saved_formations", {})}

# -------------------------------------------------------------
# MODEL PERSISTENCE & DOWNLOAD ENDPOINTS
# -------------------------------------------------------------
from fastapi.responses import FileResponse

@app.get("/ai/models/download")
async def download_model(current_user: dict = Depends(get_current_user)):
    if not check_download_quota(current_user):
        raise HTTPException(
            status_code=403,
            detail="PPO Actor-Critic weights download requires a Pro Trainer or Developer subscription."
        )
        
    users = load_users()
    username = current_user["username"]
    u = users[username]
    
    activity = u.setdefault("activity", [])
    activity.append({
        "endpoint": "Downloaded pre-trained PPO model checkpoint weights (.pt)",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })
    save_users(users)

    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rl_env", "checkpoints", "ppo_gnn_model.pt")
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model weights not found. Ensure training has started to generate the initial checkpoint.")
        
    return FileResponse(
        path=model_path,
        filename="ppo_gnn_model.pt",
        media_type="application/octet-stream"
    )
