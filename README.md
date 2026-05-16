# ⚽ Adaptive Football Simulation

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Pygame](https://img.shields.io/badge/Pygame-2.6.1-green.svg)
![Architecture](https://img.shields.io/badge/Architecture-ECS_Inspired-orange.svg)
![RL](https://img.shields.io/badge/Reinforcement_Learning-Gymnasium-purple.svg)

**Adaptive Football** is a modular, high-performance 2D tactical football simulator. Built from scratch in Python using Pygame, it moves beyond simple arcade mechanics to deliver a deeply tactical simulation featuring dynamic formations, player attributes, realistic match flow, and a fully integrated Reinforcement Learning environment.

---

## ✨ Core Features

### 🧠 Advanced Tactical AI
- **Dynamic Formations:** Watch a fluid **4-3-3** (Team Blue) clash against a compact **4-4-2** (Team Red). Support players intelligently shift into zones based on their specific roles (e.g., Wingers stay wide, CDMs protect the backline).
- **Context-Aware Decision Making:** AI evaluates risk, checks passing lanes, and will recycle possession backward if forward options are too heavily marked.
- **Goalkeeper AI:** GKs don't just chase the ball—they use linear projection to predict shot intercept points and execute high-speed diving saves.

### 🏃‍♂️ RPG-Style Attributes & Stamina
Every player is generated with unique attributes based on their tactical role:
- **Attributes:** Pace, Stamina, Passing, Vision, Shooting, Composure.
- **Fatigue System:** Sprinting drains stamina. As players tire, their maximum speed dynamically drops (up to a 30% penalty).
- **Organic Errors:** Passes and shots are not perfectly laser-guided; their accuracy deviates based on a player's `Passing` and `Shooting` attributes.

### 📊 Data Analytics & Reporting
- **Real-Time Stats Tracking:** Measures possession percentages, tracks attempted/completed passes, and logs tackles/recoveries.
- **Post-Match Report:** At the end of a 3-minute match, the system generates a natural-language tactical report analyzing the flow of the game directly in your terminal.

### 🤖 Reinforcement Learning Ready
- **Gymnasium Environment:** Includes a headless, fast-forwarding `FootballEnv` with a 95-dimensional observation space and 12 discrete actions.
- **Self-Play Scaffold:** Built-in support for multi-agent training where two RL models can learn by playing against each other.

---

## 🎮 Controls & Debugging

Take control of the Blue Team (4-3-3) while the AI manages the rest of your squad and the opponent.

| Key | Action |
| :--- | :--- |
| **W, A, S, D** | Move controlled player |
| **Shift (Hold)** | Sprint (Drains stamina quickly!) |
| **Space** | Pass ball in facing direction |
| **J** | Shoot toward target goal |
| **K** | Switch control to teammate nearest the ball |
| **F3** | Toggle Visual Tactical Debug Overlay |

*Note: Pressing **F3** reveals player stamina, tactical roles (e.g., LW, CB), and their intended velocity target lines.*

---

## 🏗️ Architecture

The project is built using an Entity-Component-System (ECS) inspired pattern to maintain strict separation of concerns, ensuring maximum performance and modularity.

```text
adaptive_football/
├── ai/                 # Tactical FSM controllers and Goalkeeper logic
├── analytics/          # Post-match NLP report generator
├── attributes/         # Player profiles and fatigue mechanics
├── debug/              # F3 visual overlays
├── engine/             # Core game loop, delta-time management, match state
├── entities/           # Data models (Ball, Player, Team)
├── physics/            # Circle-circle intersection and collision resolution
├── rendering/          # Pure rendering logic (decoupled for headless RL)
├── rl_env/             # Gymnasium wrappers and self-play scripts
├── stats/              # Real-time possession and passing trackers
├── tactics/            # Dynamic formation zoning and role definitions
└── main.py             # Entry point
```

---

## 🚀 Getting Started

### Prerequisites
Make sure you have Python 3.11+ and the required packages installed:
```bash
pip install pygame numpy
```
*(If you plan to use the RL scaffold, you will also need `gymnasium` and `stable-baselines3`)*

### Run the Simulation
```bash
python main.py
```

### Run the RL Self-Play Test (Random Agents)
```bash
python -m rl_env.self_play
```

---

*“Football is played with the head. Your feet are just the tools.”* — **Adaptive Football Engine**
