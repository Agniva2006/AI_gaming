import pygame
from engine import settings

# Standard tactical roles for 11 players
FORMATION_ROLES = {
    "4-3-3": ["GK", "LB", "CB", "CB", "RB", "CDM", "LCM", "RCM", "LW", "ST", "RW"],
    "4-4-2": ["GK", "LB", "CB", "CB", "RB", "LM", "CM", "CM", "RM", "ST", "ST"],
    "3-5-2": ["GK", "LCB", "CB", "RCB", "LWB", "CM", "CDM", "CM", "RWB", "ST", "ST"],
    "5-3-2": ["GK", "LWB", "CB", "CB", "CB", "RWB", "LCM", "CM", "RCM", "ST", "ST"],
    "4-2-3-1": ["GK", "LB", "CB", "CB", "RB", "LDM", "RDM", "LAM", "CAM", "RAM", "ST"]
}

# Normalized formation layouts (0.0 to 1.0)
FORMATION_PRESETS = {
    "4-3-3": [
        (0.05, 0.50),  # GK
        (0.20, 0.15),  # LB
        (0.15, 0.35),  # CB
        (0.15, 0.65),  # CB
        (0.20, 0.85),  # RB
        (0.32, 0.50),  # CDM
        (0.45, 0.30),  # LCM
        (0.45, 0.70),  # RCM
        (0.70, 0.15),  # LW
        (0.75, 0.50),  # ST
        (0.70, 0.85),  # RW
    ],
    "4-4-2": [
        (0.05, 0.50),  # GK
        (0.20, 0.15),  # LB
        (0.15, 0.35),  # CB
        (0.15, 0.65),  # CB
        (0.20, 0.85),  # RB
        (0.40, 0.15),  # LM
        (0.38, 0.38),  # CM
        (0.38, 0.62),  # CM
        (0.40, 0.85),  # RM
        (0.72, 0.40),  # ST
        (0.72, 0.60),  # ST
    ],
    "3-5-2": [
        (0.05, 0.50),  # GK
        (0.18, 0.25),  # LCB
        (0.14, 0.50),  # CB
        (0.18, 0.75),  # RCB
        (0.38, 0.12),  # LWB
        (0.35, 0.36),  # CM
        (0.30, 0.50),  # CDM
        (0.35, 0.64),  # CM
        (0.38, 0.88),  # RWB
        (0.73, 0.40),  # ST
        (0.73, 0.60),  # ST
    ],
    "5-3-2": [
        (0.05, 0.50),  # GK
        (0.20, 0.12),  # LWB
        (0.16, 0.30),  # CB
        (0.14, 0.50),  # CB
        (0.16, 0.70),  # CB
        (0.20, 0.88),  # RWB
        (0.42, 0.30),  # LCM
        (0.38, 0.50),  # CM
        (0.42, 0.70),  # RCM
        (0.72, 0.42),  # ST
        (0.72, 0.58),  # ST
    ],
    "4-2-3-1": [
        (0.05, 0.50),  # GK
        (0.20, 0.15),  # LB
        (0.15, 0.35),  # CB
        (0.15, 0.65),  # CB
        (0.20, 0.85),  # RB
        (0.32, 0.38),  # LDM
        (0.32, 0.62),  # RDM
        (0.55, 0.20),  # LAM
        (0.52, 0.50),  # CAM
        (0.55, 0.80),  # RAM
        (0.75, 0.50),  # ST
    ]
}

AVAILABLE_FORMATIONS = ["4-4-2", "4-3-3", "3-5-2", "5-3-2", "4-2-3-1"]

def get_formation_roles(formation_type):
    return FORMATION_ROLES.get(formation_type, FORMATION_ROLES["4-4-2"])

def get_base_positions(formation_type, attack_dir, custom_coords=None):
    """
    Calculates world pixel coordinates for the chosen formation.
    Supports preset formations or custom coordinate lists.
    attack_dir: +1 for Team A (attacks right), -1 for Team B (attacks left)
    """
    w, h = settings.SCREEN_WIDTH, settings.SCREEN_HEIGHT
    
    if custom_coords and len(custom_coords) == 11:
        norm_coords = custom_coords
    else:
        norm_coords = FORMATION_PRESETS.get(formation_type, FORMATION_PRESETS["4-4-2"])
        
    pos = []
    for x, y in norm_coords:
        if attack_dir == -1:
            x = 1.0 - x  # Mirror horizontally for Team B defending right goal
        pos.append((x * w, y * h))
    return pos

def get_tactical_target(role, base_pos, ball_pos, attack_dir):
    """Dynamic shape logic. Modifies target zone based on role and ball position."""
    target = pygame.math.Vector2(base_pos)
    ball_vec = pygame.math.Vector2(ball_pos)
    
    # Base shift toward ball
    shift_x = (ball_vec.x - target.x) * 0.28
    shift_y = (ball_vec.y - target.y) * 0.20
    
    # Role-specific tactical adjustments
    if role in ["LW", "RW", "LM", "RM", "LWB", "RWB", "LAM", "RAM"]:
        shift_x *= 1.2
        if abs(ball_vec.y - target.y) > 220:
            shift_y *= 0.25  # maintain width
    elif role in ["CDM", "LDM", "RDM"]:
        shift_x *= 0.45  # anchor shield in front of CBs
    elif role in ["CB", "LCB", "RCB", "LB", "RB"]:
        shift_x *= 0.38  # compact backline
        if "CB" in role:
            shift_y *= 0.45
    elif role in ["ST", "CAM"]:
        shift_x = (ball_vec.x - target.x) * 0.55  # press high or join attack
        shift_y *= 0.50

    target.x += shift_x
    target.y += shift_y

    w, h = settings.SCREEN_WIDTH, settings.SCREEN_HEIGHT
    r = settings.PLAYER_RADIUS
    target.x = max(r, min(target.x, w - r))
    target.y = max(r, min(target.y, h - r))
    return target
