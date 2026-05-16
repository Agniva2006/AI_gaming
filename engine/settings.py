
# Screen Size
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

# Game Constants
FIELD_COLOR = (34, 139, 34)
FIELD_COLOR_DARK = (30, 120, 30)    # alternating grass stripes
LINE_COLOR = (255, 255, 255)

TITLE = "Adaptive Football"
FPS = 60

# Ball constants
BALL_RADIUS = 8
BALL_COLOR = (255, 255, 255)
BALL_FRICTION = 300
BALL_STOP_THRESHOLD = 5

# Kick constants
KICK_RANGE = 30
PASS_POWER = 400
SHOOT_POWER = 700

# Collision / Dribbling
DRIBBLE_PUSH_SPEED = 150

# --- Player constants ---
PLAYER_RADIUS = 15
PLAYER_SPEED = 200
SPRINT_SPEED = 340

# --- Team colors ---
TEAM_A_COLOR = (30, 144, 255)       # Dodger blue (human team)
TEAM_B_COLOR = (220, 50, 50)        # Red (AI team)
CONTROLLED_HIGHLIGHT = (255, 255, 0) # Yellow ring around controlled player

# --- Goal dimensions ---
GOAL_WIDTH = 30                     # depth of the goal (drawn as rectangle)
GOAL_HEIGHT = 200                   # how tall the goal opening is
GOAL_TOP = SCREEN_HEIGHT // 2 - GOAL_HEIGHT // 2    # 260
GOAL_BOTTOM = SCREEN_HEIGHT // 2 + GOAL_HEIGHT // 2  # 460

GOAL_LEFT_CENTER = (0, SCREEN_HEIGHT // 2)
GOAL_RIGHT_CENTER = (SCREEN_WIDTH, SCREEN_HEIGHT // 2)

# --- Pitch markings ---
CENTER_CIRCLE_RADIUS = 70
PENALTY_AREA_WIDTH = 150
PENALTY_AREA_HEIGHT = 350
PENALTY_AREA_TOP = SCREEN_HEIGHT // 2 - PENALTY_AREA_HEIGHT // 2

# --- 4-4-2 Formation positions ---
# Defined as (x, y) tuples. Index 0 = GK, 1-4 = DEF, 5-8 = MID, 9-10 = FWD
# Team A attacks RIGHT (human)
FORMATION_442_A = [
    (80, 360),          # GK
    (220, 130),         # LB
    (220, 290),         # CB
    (220, 430),         # CB
    (220, 590),         # RB
    (450, 130),         # LM
    (450, 290),         # CM
    (450, 430),         # CM
    (450, 590),         # RM
    (620, 290),         # ST
    (620, 430),         # ST
]

# Team B attacks LEFT (AI) — mirrored
FORMATION_442_B = [
    (1200, 360),        # GK
    (1060, 590),        # LB
    (1060, 430),        # CB
    (1060, 290),        # CB
    (1060, 130),        # RB
    (830, 590),         # LM
    (830, 430),         # CM
    (830, 290),         # CM
    (830, 130),         # RM
    (660, 430),         # ST
    (660, 290),         # ST
]

# Player roles (by index in formation)
ROLE_GK = 0
ROLE_DEF = (1, 2, 3, 4)
ROLE_MID = (5, 6, 7, 8)
ROLE_FWD = (9, 10)

# --- AI constants ---
AI_SPEED = 180                      # slightly slower than human for fairness
AI_CHASE_SPEED = 200
AI_PRESS_DISTANCE = 250             # how close before AI presses ball carrier
AI_SHOOT_DISTANCE = 300             # how close to goal before AI shoots
AI_PASS_COOLDOWN = 0.5              # seconds between AI passes
AI_BALL_SHIFT_FACTOR = 0.25         # how much formation shifts toward ball (0-1)

# --- Match constants ---
KICKOFF_PAUSE = 1.0                 # seconds pause after goal before play resumes

# --- Goalkeeper AI (Phase 9) ---
GK_DIVE_SPEED = 400                 # speed when diving for a shot
GK_NORMAL_SPEED = 200               # speed when positioning
GK_COME_OUT_DISTANCE = 200          # how far GK will come off line for 1v1
SHOT_SPEED_THRESHOLD = 250          # ball speed that triggers a dive
GK_RECOVERY_TIME = 0.5             # seconds to recover after a dive

# --- RL Environment (Phase 10-11) ---
RL_MAX_STEPS = 3000                 # max steps per episode
RL_REWARD_GOAL = 1.0                # reward for scoring
RL_REWARD_CONCEDE = -1.0            # penalty for conceding
RL_REWARD_POSSESSION = 0.001        # small reward per step with possession
RL_REWARD_BALL_PROGRESS = 0.0005    # reward for moving ball toward opponent goal
NUM_ACTIONS = 12                    # number of discrete actions for RL agent