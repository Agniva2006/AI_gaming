import sqlite3
import json
import os
import threading
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "football.db")

class Database:
    """Thread-safe SQLite Database Manager for RL Train Football."""
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Users table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT DEFAULT '',
                role TEXT DEFAULT 'Player / Coach',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # 2. Formations table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS formations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NULL,
                name TEXT NOT NULL,
                formation_type TEXT NOT NULL,
                coordinates_json TEXT NOT NULL,
                is_default BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # 3. Matches table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NULL,
                match_type TEXT DEFAULT 'HUMAN_VS_AI',
                human_score INTEGER DEFAULT 0,
                ai_score INTEGER DEFAULT 0,
                human_formation TEXT DEFAULT '4-3-3',
                ai_formation TEXT DEFAULT '4-4-2',
                possession_human REAL DEFAULT 50.0,
                possession_ai REAL DEFAULT 50.0,
                shots_human INTEGER DEFAULT 0,
                shots_ai INTEGER DEFAULT 0,
                passes_human INTEGER DEFAULT 0,
                passes_ai INTEGER DEFAULT 0,
                result TEXT DEFAULT 'DRAW',
                ai_reward REAL DEFAULT 0.0,
                ai_loss REAL DEFAULT 0.0,
                duration_seconds REAL DEFAULT 180.0,
                xg_human REAL DEFAULT 0.0,
                xg_ai REAL DEFAULT 0.0,
                tactical_style TEXT DEFAULT 'Balanced',
                shots_data_json TEXT DEFAULT '[]',
                ai_adaptation_json TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # Safely ensure new columns exist if table was previously created
            for col, col_type in [
                ("xg_human", "REAL DEFAULT 0.0"),
                ("xg_ai", "REAL DEFAULT 0.0"),
                ("tactical_style", "TEXT DEFAULT 'Balanced'"),
                ("shots_data_json", "TEXT DEFAULT '[]'"),
                ("ai_adaptation_json", "TEXT DEFAULT '{}'")
            ]:
                try:
                    cursor.execute(f"ALTER TABLE matches ADD COLUMN {col} {col_type}")
                except Exception:
                    pass

            # 4. AI model global stats
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_model_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT UNIQUE NOT NULL,
                episodes_trained INTEGER DEFAULT 0,
                total_matches INTEGER DEFAULT 0,
                ai_wins INTEGER DEFAULT 0,
                human_wins INTEGER DEFAULT 0,
                draws INTEGER DEFAULT 0,
                current_reward REAL DEFAULT 0.0,
                average_loss REAL DEFAULT 0.0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # 5. AI training history / telemetry
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_training_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode INTEGER NOT NULL,
                reward REAL NOT NULL,
                actor_loss REAL DEFAULT 0.0,
                critic_loss REAL DEFAULT 0.0,
                source TEXT DEFAULT 'MATCH',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # Seed default AI stats row if empty
            cursor.execute("SELECT COUNT(*) FROM ai_model_stats WHERE model_name = 'PPO_GNN_FOOTBALL'")
            if cursor.fetchone()[0] == 0:
                cursor.execute("""
                INSERT INTO ai_model_stats (model_name, episodes_trained, total_matches, ai_wins, human_wins, draws, current_reward, average_loss)
                VALUES ('PPO_GNN_FOOTBALL', 10, 0, 0, 0, 0, 0.25, 0.05)
                """)

            # Seed standard default tactical formations if empty
            cursor.execute("SELECT COUNT(*) FROM formations WHERE is_default = 1")
            if cursor.fetchone()[0] == 0:
                self._seed_default_formations(cursor)

            conn.commit()

    def _seed_default_formations(self, cursor):
        presets = {
            "4-3-3 Standard": {
                "type": "4-3-3",
                "coords": [
                    [0.05, 0.5],   # GK
                    [0.20, 0.15],  # LB
                    [0.15, 0.35],  # CB
                    [0.15, 0.65],  # CB
                    [0.20, 0.85],  # RB
                    [0.32, 0.50],  # CDM
                    [0.45, 0.30],  # LCM
                    [0.45, 0.70],  # RCM
                    [0.70, 0.15],  # LW
                    [0.75, 0.50],  # ST
                    [0.70, 0.85]   # RW
                ]
            },
            "4-4-2 Classic": {
                "type": "4-4-2",
                "coords": [
                    [0.05, 0.5],   # GK
                    [0.20, 0.15],  # LB
                    [0.15, 0.35],  # CB
                    [0.15, 0.65],  # CB
                    [0.20, 0.85],  # RB
                    [0.40, 0.15],  # LM
                    [0.38, 0.38],  # CM
                    [0.38, 0.62],  # CM
                    [0.40, 0.85],  # RM
                    [0.72, 0.40],  # ST
                    [0.72, 0.60]   # ST
                ]
            },
            "3-5-2 Attacking": {
                "type": "3-5-2",
                "coords": [
                    [0.05, 0.5],   # GK
                    [0.18, 0.25],  # LCB
                    [0.14, 0.50],  # CB
                    [0.18, 0.75],  # RCB
                    [0.38, 0.12],  # LWB
                    [0.35, 0.36],  # CM
                    [0.30, 0.50],  # CDM
                    [0.35, 0.64],  # CM
                    [0.38, 0.88],  # RWB
                    [0.73, 0.40],  # ST
                    [0.73, 0.60]   # ST
                ]
            },
            "5-3-2 Solid Wall": {
                "type": "5-3-2",
                "coords": [
                    [0.05, 0.5],   # GK
                    [0.20, 0.12],  # LWB
                    [0.16, 0.30],  # CB
                    [0.14, 0.50],  # CB
                    [0.16, 0.70],  # CB
                    [0.20, 0.88],  # RWB
                    [0.42, 0.30],  # LCM
                    [0.38, 0.50],  # CM
                    [0.42, 0.70],  # RCM
                    [0.72, 0.42],  # ST
                    [0.72, 0.58]   # ST
                ]
            },
            "4-2-3-1 Modern": {
                "type": "4-2-3-1",
                "coords": [
                    [0.05, 0.5],   # GK
                    [0.20, 0.15],  # LB
                    [0.15, 0.35],  # CB
                    [0.15, 0.65],  # CB
                    [0.20, 0.85],  # RB
                    [0.32, 0.38],  # LDM
                    [0.32, 0.62],  # RDM
                    [0.55, 0.20],  # LAM
                    [0.52, 0.50],  # CAM
                    [0.55, 0.80],  # RAM
                    [0.75, 0.50]   # ST
                ]
            }
        }
        for name, data in presets.items():
            cursor.execute("""
            INSERT INTO formations (user_id, name, formation_type, coordinates_json, is_default)
            VALUES (NULL, ?, ?, ?, 1)
            """, (name, data["type"], json.dumps(data["coords"])))

    # --- User operations ---
    def create_user(self, username, email, password_hash, full_name="", role="Player / Coach"):
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO users (username, email, password_hash, full_name, role)
            VALUES (?, ?, ?, ?, ?)
            """, (username.strip().lower(), email.strip().lower(), password_hash, full_name, role))
            conn.commit()
            return cursor.lastrowid

    def get_user_by_username(self, username):
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (username.strip().lower(),))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_user_by_id(self, user_id):
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    # --- Formation operations ---
    def save_formation(self, name, formation_type, coordinates, user_id=None):
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            coords_str = json.dumps(coordinates) if isinstance(coordinates, list) else coordinates
            cursor.execute("""
            INSERT INTO formations (user_id, name, formation_type, coordinates_json, is_default)
            VALUES (?, ?, ?, ?, 0)
            """, (user_id, name, formation_type, coords_str))
            conn.commit()
            return cursor.lastrowid

    def get_formations(self, user_id=None):
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            if user_id is not None:
                cursor.execute("""
                SELECT * FROM formations WHERE is_default = 1 OR user_id = ? ORDER BY id ASC
                """, (user_id,))
            else:
                cursor.execute("SELECT * FROM formations ORDER BY id ASC")
            rows = cursor.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                item["coordinates"] = json.loads(item["coordinates_json"])
                results.append(item)
            return results

    def delete_formation(self, formation_id, user_id=None):
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            if user_id is not None:
                cursor.execute("DELETE FROM formations WHERE id = ? AND user_id = ? AND is_default = 0", (formation_id, user_id))
            else:
                cursor.execute("DELETE FROM formations WHERE id = ? AND is_default = 0", (formation_id,))
            conn.commit()
            return cursor.rowcount > 0

    # --- Match operations ---
    def record_match(self, match_type, human_score, ai_score, human_formation, ai_formation,
                     possession_human, possession_ai, shots_human, shots_ai, passes_human, passes_ai,
                     result, ai_reward=0.0, ai_loss=0.0, duration_seconds=180.0, user_id=None,
                     xg_human=0.0, xg_ai=0.0, tactical_style="Balanced",
                     shots_data=None, ai_adaptation=None):
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            shots_json = json.dumps(shots_data if shots_data is not None else [])
            adapt_json = json.dumps(ai_adaptation if ai_adaptation is not None else {})
            cursor.execute("""
            INSERT INTO matches (
                user_id, match_type, human_score, ai_score, human_formation, ai_formation,
                possession_human, possession_ai, shots_human, shots_ai, passes_human, passes_ai,
                result, ai_reward, ai_loss, duration_seconds, xg_human, xg_ai, tactical_style,
                shots_data_json, ai_adaptation_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id, match_type, human_score, ai_score, human_formation, ai_formation,
                possession_human, possession_ai, shots_human, shots_ai, passes_human, passes_ai,
                result, ai_reward, ai_loss, duration_seconds, xg_human, xg_ai, tactical_style,
                shots_json, adapt_json
            ))
            match_id = cursor.lastrowid

            # Also update AI aggregate stats
            cursor.execute("SELECT * FROM ai_model_stats WHERE model_name = 'PPO_GNN_FOOTBALL'")
            row = cursor.fetchone()
            if row:
                ai_wins = row["ai_wins"] + (1 if result == "LOSS" else 0)
                human_wins = row["human_wins"] + (1 if result == "WIN" else 0)
                draws = row["draws"] + (1 if result == "DRAW" else 0)
                total_matches = row["total_matches"] + 1
                episodes = row["episodes_trained"] + 1
                # Exponential moving average for reward & loss
                curr_reward = row["current_reward"] * 0.9 + ai_reward * 0.1
                avg_loss = row["average_loss"] * 0.9 + ai_loss * 0.1

                cursor.execute("""
                UPDATE ai_model_stats SET
                    episodes_trained = ?,
                    total_matches = ?,
                    ai_wins = ?,
                    human_wins = ?,
                    draws = ?,
                    current_reward = ?,
                    average_loss = ?,
                    last_updated = CURRENT_TIMESTAMP
                WHERE model_name = 'PPO_GNN_FOOTBALL'
                """, (episodes, total_matches, ai_wins, human_wins, draws, curr_reward, avg_loss))

            # Record training entry
            cursor.execute("""
            INSERT INTO ai_training_history (episode, reward, actor_loss, critic_loss, source)
            VALUES (?, ?, ?, ?, 'MATCH')
            """, (episodes if row else 1, ai_reward, ai_loss * 0.5, ai_loss * 0.5))

            conn.commit()
            return match_id

    def get_match_history(self, limit=20):
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM matches ORDER BY id DESC LIMIT ?", (limit,))
            rows = [dict(r) for r in cursor.fetchall()]
            for r in rows:
                try:
                    r["shots_data"] = json.loads(r.get("shots_data_json") or "[]")
                except Exception:
                    r["shots_data"] = []
                try:
                    r["ai_adaptation"] = json.loads(r.get("ai_adaptation_json") or "{}")
                except Exception:
                    r["ai_adaptation"] = {}
            return rows

    def get_match_summary(self):
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT 
                COUNT(*) as total_matches,
                SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as human_wins,
                SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END) as ai_wins,
                SUM(CASE WHEN result = 'DRAW' THEN 1 ELSE 0 END) as draws,
                SUM(human_score) as total_human_goals,
                SUM(ai_score) as total_ai_goals,
                AVG(possession_human) as avg_possession_human
            FROM matches
            """)
            row = cursor.fetchone()
            return dict(row) if row else {}

    # --- AI telemetry operations ---
    def get_ai_stats(self):
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM ai_model_stats WHERE model_name = 'PPO_GNN_FOOTBALL'")
            row = cursor.fetchone()
            if row:
                d = dict(row)
                tot = max(1, d["total_matches"])
                d["ai_win_rate"] = round((d["ai_wins"] / tot) * 100, 1)
                d["human_win_rate"] = round((d["human_wins"] / tot) * 100, 1)
                return d
            return {}

    def record_ai_training(self, episode, reward, actor_loss, critic_loss, source="BACKGROUND"):
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO ai_training_history (episode, reward, actor_loss, critic_loss, source)
            VALUES (?, ?, ?, ?, ?)
            """, (episode, reward, actor_loss, critic_loss, source))
            
            # Update aggregate stats
            cursor.execute("""
            UPDATE ai_model_stats SET
                episodes_trained = episodes_trained + 1,
                current_reward = current_reward * 0.9 + ? * 0.1,
                average_loss = average_loss * 0.9 + (? + ?) * 0.05,
                last_updated = CURRENT_TIMESTAMP
            WHERE model_name = 'PPO_GNN_FOOTBALL'
            """, (reward, actor_loss, critic_loss))
            conn.commit()

    def get_ai_training_history(self, limit=50):
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM ai_training_history ORDER BY id DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            return [dict(r) for r in reversed(rows)]

    def reset_ai_stats(self):
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            UPDATE ai_model_stats SET
                episodes_trained = 0,
                total_matches = 0,
                ai_wins = 0,
                human_wins = 0,
                draws = 0,
                current_reward = 0.0,
                average_loss = 0.0,
                last_updated = CURRENT_TIMESTAMP
            WHERE model_name = 'PPO_GNN_FOOTBALL'
            """)
            cursor.execute("DELETE FROM ai_training_history")
            conn.commit()

# Global database singleton
db = Database()

