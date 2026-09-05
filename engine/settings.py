# Screen Size & Window
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
TITLE = "⚽ RL Train Football — Human vs Learning RL AI"
FPS = 60

# Game & Pitch Appearance
FIELD_COLOR = (34, 139, 34)
FIELD_COLOR_DARK = (30, 120, 30)      # alternating grass stripes
LINE_COLOR = (255, 255, 255)

# Ball Constants
BALL_RADIUS = 8
BALL_COLOR = (255, 255, 255)
BALL_FRICTION = 280.0                  # ground rolling friction
BALL_STOP_THRESHOLD = 5.0

# Kick & Shot Powers
KICK_RANGE = 32.0                      # reach distance to kick ball
PASS_POWER = 420.0                     # crisp pass speed
SHOOT_POWER = 680.0                    # shot speed
CHIP_LIFT = 300.0                      # lofted pass upward lift
TACKLE_RANGE = 35.0

# Player Constants
PLAYER_RADIUS = 15
PLAYER_SPEED = 210.0
SPRINT_SPEED = 330.0

# Team Colors
TEAM_A_COLOR = (30, 144, 255)          # Dodger blue (Human Team)
TEAM_B_COLOR = (220, 50, 50)           # Crimson Red (RL AI Team)
CONTROLLED_HIGHLIGHT = (255, 220, 0)   # Vibrant neon yellow marker for human player

# Goal Dimensions
GOAL_WIDTH = 32
GOAL_HEIGHT = 200
GOAL_TOP = SCREEN_HEIGHT // 2 - GOAL_HEIGHT // 2       # 260
GOAL_BOTTOM = SCREEN_HEIGHT // 2 + GOAL_HEIGHT // 2     # 460
GOAL_LEFT_CENTER = (0, SCREEN_HEIGHT // 2)
GOAL_RIGHT_CENTER = (SCREEN_WIDTH, SCREEN_HEIGHT // 2)

# Pitch Markings
CENTER_CIRCLE_RADIUS = 75
PENALTY_AREA_WIDTH = 150
PENALTY_AREA_HEIGHT = 350
PENALTY_AREA_TOP = SCREEN_HEIGHT // 2 - PENALTY_AREA_HEIGHT // 2

# AI Tuning
AI_SPEED = 195.0
AI_CHASE_SPEED = 215.0
AI_PRESS_DISTANCE = 240.0
AI_SHOOT_DISTANCE = 220.0
AI_PASS_COOLDOWN = 0.35
AI_SHOOT_COOLDOWN = 1.0

# Match Rules
MATCH_DURATION = 120.0                 # 2 minutes per match (fast, dynamic, fun)
KICKOFF_PAUSE = 1.2                    # pause after goal before kickoff

# Goalkeeper AI
GK_DIVE_SPEED = 420.0
GK_NORMAL_SPEED = 200.0
GK_COME_OUT_DISTANCE = 180.0
SHOT_SPEED_THRESHOLD = 240.0
GK_RECOVERY_TIME = 0.5

# RL Brain & Training Parameters
RL_MAX_STEPS = 600
RL_REWARD_GOAL = 2.0
RL_REWARD_CONCEDE = -2.0
RL_REWARD_SHOT = 0.4
RL_REWARD_PASS = 0.15
RL_REWARD_TACKLE = 0.25
RL_REWARD_POSSESSION = 0.002
RL_REWARD_PROGRESS = 0.001
RL_REWARD_BALL_PROGRESS = 0.001
NUM_ACTIONS = 12