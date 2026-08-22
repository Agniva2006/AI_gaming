import os
import json
from datetime import datetime, timedelta
from jose import jwt, JWTError
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from typing import Dict, Any, Optional

# Security configurations
SECRET_KEY = os.environ.get("SIGN0_SECRET_KEY", "7e4c2780e9063d3fb1505c87ab42f49463c2c1bb8bf21d4d87be33f5d506ad5e")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "users_db.json")

# Plans definition
PLANS = {
    "free": {"max_saved_formations": 3, "allow_model_downloads": False},
    "pro": {"max_saved_formations": 50, "allow_model_downloads": True},
    "developer": {"max_saved_formations": 9999, "allow_model_downloads": True}
}

from threading import Lock

_db_lock = Lock()

def load_users() -> dict:
    with _db_lock:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        if not os.path.exists(DB_PATH):
            default_db = {
                "trainer1": {
                    "username": "trainer1",
                    "email": "trainer1@stadium.ai",
                    "password_hash": pwd_context.hash("password123"),
                    "full_name": "Pro Coach",
                    "role": "educator",
                    "plan": "free",
                    "stripe_customer_id": "",
                    "stripe_subscription_id": "",
                    "subscription_status": "",
                    "saved_formations": {},
                    "activity": []
                }
            }
            with open(DB_PATH, "w") as f:
                json.dump(default_db, f, indent=2)
            return default_db
        try:
            with open(DB_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return {}

def save_users(users: dict):
    with _db_lock:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        with open(DB_PATH, "w") as f:
            json.dump(users, f, indent=2)

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Access token is missing. Please sign in.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not token:
        raise credentials_exception
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
        
    users = load_users()
    if username not in users:
        raise credentials_exception
    return users[username]

async def get_optional_user(token: str = Depends(oauth2_scheme)) -> Optional[dict]:
    if not token:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username:
            users = load_users()
            return users.get(username)
    except JWTError:
        pass
    return None

def check_formation_quota(user: dict) -> bool:
    plan = user.get("plan", "free")
    plan_meta = PLANS.get(plan, PLANS["free"])
    saved_count = len(user.get("saved_formations", {}))
    return saved_count < plan_meta["max_saved_formations"]

def check_download_quota(user: dict) -> bool:
    plan = user.get("plan", "free")
    plan_meta = PLANS.get(plan, PLANS["free"])
    return plan_meta["allow_model_downloads"]
