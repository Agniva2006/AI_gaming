# ⚽ StadiumAI: 2D Football (Human vs Learning RL AI)

An interactive, physics-driven 2D football game built with Python (Pygame), PyTorch, FastAPI, SQLite, and a modern web dashboard. Play directly as Team Blue against a deep reinforcement learning (RL) AI opponent with dynamic formations, where the AI continuously collects match experience and adapts to your playstyle with every game.

---

## 🌟 Key Features

- **🎮 Human vs Learning RL AI**: Take active control of your team on the pitch! Control player movement, sprint with stamina drain, pass directionally to teammates, shoot on target, execute lofted chips, and slide-tackle opponents.
- **🛡️ Multiple Tactical Formations**: Choose and switch between tactical formations for both your team and the AI opponent:
  - `4-4-2 Classic`
  - `4-3-3 Attacking Wing Play`
  - `3-5-2 Midfield Dominance`
  - `5-3-2 Solid Wall Defense`
  - `4-2-3-1 Modern Tactical Pivot`
  - Custom formations designed and saved in the web dashboard!
- **🧠 Continuous Match Learning**: Unlike static AI bots, the AI opponent collects state-action-reward transitions throughout every match. Upon full-time, the policy network executes an online PPO (Proximal Policy Optimization) update, persisting checkpoints so the AI opponent gets visibly smarter the more games you play.
- **⚽ Realistic 2D/3D Football Physics**: Smooth rolling turf friction, 3D ball altitude ($z$-axis) with height-scaled ground shadows, Magnus effect curving shots, goal net ripples, and intelligent Goalkeeper AI with shot trajectory prediction and diving saves.
- **🗄️ Thread-Safe SQLite Database**: Automatically persists users, tactical formations, match histories (scores, possession %, shots, passes), and AI training logs to `backend/football.db`.
- **🌐 Connected Modern Web Dashboard**: Live browser dashboard featuring Match Center statistics, an interactive drag-and-drop 2D pitch formation builder, real-time Chart.js AI reward & loss curves, and complete match histories.

---

## 🕹️ In-Game Controls

| Key | Action | Football Context & Mechanics |
|-----|--------|------------------------------|
| **W / A / S / D** or **Arrow Keys** | Move active player | Smooth 2D directional locomotion with turning inertia |
| **Left / Right Shift** | Sprint | High-speed sprint; sprinting triggers heavier first-touches |
| **Hold J / Z** | **Shot Power Meter** | Hold to charge power meter (Green -> Yellow -> Red); release to strike! Sweet spot (40%-88%) unleashes rockets; overcharge (>88%) skies wild |
| **K** or **X** | **Pass / Tackle** | Crisp ground pass to teammate / Slide tackle when defending |
| **Hold C** or **Left Alt** | **Defensive Jockey** | Jockey stance when defending: squares hips to ball, containment speed, expands tackle radius by 40% with clean traps |
| **E** | **Knock-Ahead Burst** | Knock ball 40px into open space when possessing ball to outpace defenders |
| **L** | **Through-Ball** | Penetrating forward through-ball into running channels |
| **SPACE** or **TAB** | Switch Player | Intelligently cycles control to the player nearest the ball |
| **1** | **Balanced** Tactic | Default balanced structure, structured position holding |
| **2** | **Tiki-Taka** Tactic | High compactness, short passing triangles, patient buildup |
| **3** | **Gegenpress** Tactic | Aggressive high defensive line, suffocating turnover press |
| **4** | **Counter-Attack** | Deep defensive block, rapid direct vertical transitions |
| **5** | **Park The Bus** | Ultra-compact penalty box wall defending a narrow lead |
| **T** | **VAR & Passing Overlay** | Live passing safety rays (Green/Yellow/Red) & VAR offside line |
| **F3** | Debug Overlay | Real-time FPS, velocity vectors, and coordinate telemetry |
| **ESC** | Pause / Main Menu | Return to main menu to adjust match setup |

---

## 🔬 Realistic Football Physics & AI Intelligence

1. **Shot Power Meter & Tiered Ballistics**:
   - Holding `J`/`Z` displays a 3-tier overhead charging bar. Low charge produces low driven ground shots, the sweet spot (40%–88%) unleashes rocket strikes, and overcharging (>88%) imparts high vertical lift with wide shot dispersion.
2. **Defensive Jockeying & Dynamic First-Touches**:
   - Holding `C` or `Alt` on defense squares hips to the ball, containment movement, and increases tackle radius by 40% with clean interception trapping. First touches dynamically react to movement: sprinting causes heavy touches into space, while walking or jockeying cleanly deadens ball velocity.
3. **Knock-Ahead Skill Burst**:
   - Pressing `E` while in possession knocks the ball 40px forward into open pitch space, letting rapid wingers burn defenders in footraces.
4. **Real-Time AI Opponent Tendency Profiling**:
   - Tracks your tactical habits in real-time: flank overload bias (Left Wing, Center, Right Wing), vertical through-ball vs ground pass frequency, and shot distances.
5. **Dynamic AI Counter-Tactics**:
   - The AI dynamically shifts fullbacks and midfielders to overload and choke your favorite flank, drops its defensive line 20% deeper to erase space behind against through-balls, and modulates pressing intensity.
6. **Live Passing Lane Safety Rays**:
   - Toggle overlay with `T` to see dynamic raycasts to all teammates: Green (>85% clear), Yellow (contested lane), Red (blocked by opponent).
7. **Opta-Style 2D Shot Map & AI Opponent Mind Debrief**:
   - Web dashboard features an interactive 2D pitch shot map with $xG$-proportional colored markers and a full AI tactical debrief card revealing exactly how the AI scouted and countered your playstyle.
8. **Physical Player Collisions & Woodwork Rebounds**:
   - Full shoulder-barge mass deflections and circular collision physics for all 4 goalposts with coefficient of restitution.
9. **Online PPO Adaptation**:
   - PyTorch Actor-Critic neural network collects match experience during gameplay and performs online policy updates upon full-time, persisting weights so the AI learns continuously.

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- `pip`

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Project
```bash
python main.py
```
This simultaneously:
1. Starts the **FastAPI Backend & WebSocket Gateway** at `http://127.0.0.1:8000`.
2. Serves the **Web Dashboard** at `http://127.0.0.1:8000/dashboard`.
3. Launches the **Pygame 2D Football Engine**.

---

## 🏛️ Project Architecture

```
AI_gaming/
├── main.py                     # Dual launcher (FastAPI server + Pygame engine)
├── engine/
│   ├── game.py                 # Game state machine, human player loop, match learning
│   ├── match.py                # Match clock, scoring, goal ripples, kickoff resets
│   └── settings.py             # Pitch dimensions, physics, and gameplay tuning
├── entities/
│   ├── players.py              # Interactive human controls, sprint stamina, kicking
│   ├── ball.py                 # 3D ball physics, Magnus effect, rolling friction
│   └── team.py                 # Team roster (11 players), formation positioning
├── ai/
│   ├── ai_controller.py        # RL policy inference + tactical FSM support
│   └── goalkeeper.py           # Dedicated GK with shot trajectory prediction & diving
├── tactics/
│   ├── formations.py           # 5 tactical formation presets + custom coordinate loader
│   └── manager.py              # Score/time dynamic tactical shifting
├── rendering/
│   ├── renderer.py             # 2.5D rendering, dynamic camera, radar minimap, HUD
│   ├── camera.py               # Viewport smoothing and ball tracking
│   ├── particles.py            # Turf dust, ball trails, and goal spark effects
│   └── net_physics.py          # Goal net grid spring-mass displacement
├── rl_env/
│   ├── nn_brain.py             # Spatial GNN + PPO Actor-Critic neural network
│   ├── gnn_encoder.py          # Spatial Graph Attention Network (GAT)
│   ├── behavioral_cloning.py   # Expert demonstration generator & supervised BC trainer
│   ├── football_env.py         # Headless gym-like environment
│   ├── trainer.py              # PPO policy optimization from match trajectories
│   └── checkpoints/            # Persisted .pt neural weights
├── train_bc.py                 # Standalone Behavioral Cloning CLI runner
├── backend/
│   ├── app.py                  # FastAPI REST API & WebSocket broadcast gateway
│   ├── database.py             # Thread-safe SQLite repository (users, matches, formations, ai stats)
│   ├── auth.py                 # JWT authentication & bcrypt password hashing
│   ├── models.py               # Pydantic request/response validation schemas
│   └── football.db             # Real SQLite database
└── frontend/
    ├── index.html              # Modern glassmorphic dashboard
    ├── style.css               # Responsive dark theme styling
    └── app.js                  # Frontend client (WebSocket + REST + Chart.js + 2D pitch editor)
```

---

## 🧠 Offline Behavioral Cloning Pre-Training

To warm-start the GNN Actor-Critic policy so the AI begins with immediate passing, shooting, and positioning competence on Match #1 (rather than random exploration), execute the imitation learning pipeline:

```powershell
python train_bc.py --episodes 15 --epochs 5
```
Or click **⚡ Pre-Train AI (Imitation)** in the Web Dashboard under **📈 AI Learning & Evolution**.

---

## 📡 REST API & WebSocket Endpoints

- `GET /` — API & AI model status
- `POST /api/auth/register` & `POST /api/auth/login` — Account authentication
- `GET /api/formations` & `POST /api/formations` — Tactical formation CRUD
- `POST /api/matches/record` — Records completed matches from game loop
- `GET /api/matches/history` & `GET /api/matches/summary` — Match statistics and head-to-head records
- `GET /api/ai/stats` & `GET /api/ai/history` — Real-time AI learning telemetry
- `POST /api/ai/pretrain-bc` — Trigger offline Behavioral Cloning pre-training
- `POST /api/ai/train-step` — Trigger background PPO training step
- `POST /api/ai/reset` — Reset AI policy to rookie baseline
- `GET /api/ai/download-model` — Download latest PyTorch policy weights (`.pt`)
- `WS /ws/live` — Live WebSocket stream of matches and training telemetry
- `WS /ws/live` — Live WebSocket stream of matches and training telemetry
