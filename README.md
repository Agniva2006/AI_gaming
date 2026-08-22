# ⚽ RL Train Football

A multi-agent autonomous football simulation powered by Graph Neural Networks and Proximal Policy Optimization (PPO), with a full-stack SaaS backend and web dashboard.

## What It Does

- **Autonomous Matches**: Two teams of 11 AI-controlled players play full 3-minute football matches with zero human input
- **Neural Network AI**: A Graph Attention Network (GAT) encodes spatial player relationships, feeding into a PPO Actor-Critic policy for tactical decision-making
- **Real-Time Visualization**: A Pygame 2.5D engine renders the match with dynamic camera tracking, particle effects, and physical goal net ripples
- **SaaS Backend**: A FastAPI REST API provides user authentication, tactical formation storage, model weight downloads, and Stripe billing integration
- **Web Dashboard**: A glassmorphic browser-based dashboard with a drag-and-drop formation editor, live telemetry charts, and subscription management

## Quick Start

### Prerequisites
- Python 3.10+
- pip

### Install & Run

```bash
cd AI_gaming
pip install -r requirements.txt
python main.py
```

This starts both the FastAPI backend (port 8000) and the Pygame visualization simultaneously.

### Controls

| Key | Action |
|-----|--------|
| **↑/↓ or W/S** | Navigate menu |
| **Enter/Space** | Select menu option |
| **F3** | Toggle debug overlay (velocity vectors, stamina) |
| **ESC** | Return to menu |

### Web Dashboard
Open `frontend/index.html` in your browser while `main.py` is running. The dashboard connects to the API at `localhost:8000`.

## Architecture

```
main.py
├── Process 1: FastAPI Backend (uvicorn, port 8000)
│   ├── JWT Authentication (/auth/*)
│   ├── Formation CRUD (/tactics/*)
│   ├── Model Download (/ai/models/download)
│   ├── Stripe Billing (/payment/*)
│   └── WebSocket Telemetry (/ws/telemetry)
│
└── Process 2: Pygame Engine (main thread)
    ├── Game State Machine (Menu → Gameplay → Training)
    ├── Entity System (22 Players + Ball, 3D physics)
    ├── AI Controllers (GNN inference + FSM fallback)
    ├── Physics (collisions, gravity, Magnus spin)
    └── 2.5D Renderer (camera, particles, net physics)
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Game Engine | Pygame 2.x |
| Neural Networks | PyTorch (GAT + PPO Actor-Critic) |
| Backend API | FastAPI + Uvicorn |
| Authentication | JWT (PyJWT) + bcrypt |
| Payments | Stripe API (sandbox mode) |
| Frontend | Vanilla HTML/CSS/JS + Chart.js |
| Data | JSON flat-file database |

## Key Technical Decisions

- **Dual-process architecture**: Pygame and FastAPI run in separate OS processes via `multiprocessing` to avoid GIL contention and Pygame's single-threaded event loop blocking the API
- **GNN for spatial reasoning**: A Graph Attention Network dynamically constructs adjacency graphs based on player proximity (350px threshold), allowing the policy to reason about passing lanes and marking pressure
- **FSM + Neural hybrid**: The AI uses neural network inference when a trained checkpoint exists, with a deterministic FSM fallback for reliable behavior without trained weights
- **Thread-safe database**: The JSON user database uses `threading.Lock` to prevent corruption during concurrent API requests

## Project Structure

| Directory | Files | Purpose |
|-----------|-------|---------|
| `engine/` | 3 | Game loop, match rules, configuration (116 constants) |
| `entities/` | 4 | 3D entity base class, player physics, ball physics, team factory |
| `ai/` | 2 | Outfield AI controller, dedicated goalkeeper AI with shot prediction |
| `rl_env/` | 7 | Neural network, GNN encoder, RL environment, PPO trainer, self-play |
| `tactics/` | 2 | Formation coordinates (4-3-3, 4-4-2), dynamic tactical manager |
| `rendering/` | 4 | 2.5D renderer, dynamic camera, particles, spring-mass goal nets |
| `backend/` | 4 | FastAPI app, JWT auth, Stripe payments, user database |
| `frontend/` | 3 | SPA dashboard, CSS theme, auth client |
| `physics/` | 1 | Collision detection and resolution |
| `analytics/` | 2 | Spatial analysis (offside line, compactness), post-match reports |
| `stats/` | 1 | Match statistics tracking (possession, shots, tackles) |

## Sample Match Output

```
========================================
          FULL TIME MATCH REPORT
========================================
Score: BLUE (4-3-3)  2 - 1  RED (4-4-2)
----------------------------------------
Possession:         52.3%  |  47.7%
Passes Attempted:   34     |  28
Shots:              12     |  9
Tackles/Recoveries: 18     |  22
----------------------------------------
TACTICAL SUMMARY:
> A tightly contested match. Both tactical systems neutralized each
> other, resulting in a fierce battle for the middle third.
========================================
```

## Known Limitations

- The GNN-PPO model requires pre-training via `rl_env/trainer.py` or `rl_env/self_play.py` before neural inference produces intelligent behavior. Without a trained checkpoint, the AI falls back to a deterministic FSM
- The `TacticalDiffusionGenerator` and `TrajectoryDiscriminator` modules are implemented but require training on match trajectory data before activation
- Stripe billing runs in sandbox mode by default (no real charges). Set `STRIPE_API_KEY` environment variable for production
- The web telemetry chart displays a simulated preview; real-time training data requires connecting via the WebSocket endpoint

## Future Enhancements

- [ ] Train the diffusion model on collected match trajectories for diverse scenario generation
- [ ] Integrate stable-baselines3 for large-scale distributed training (scaffold in `self_play.py`)
- [ ] Add a Voronoi-based spatial pressure heatmap to the web dashboard
- [ ] Support additional formations (3-4-3, 4-1-4-1) in the tactical manager

## Requirements

```
pygame>=2.1.0
torch>=2.0.0
fastapi>=0.100.0
uvicorn>=0.23.0
pyjwt>=2.8.0
bcrypt>=4.0.0
pydantic[email]>=2.0.0
stripe>=5.0.0
numpy>=1.24.0
```
