import pygame
from engine import settings

# Roles for each index
ROLES_433 = ["GK", "LB", "CB", "CB", "RB", "CDM", "LCM", "RCM", "LW", "ST", "RW"]
ROLES_442 = ["GK", "LB", "CB", "CB", "RB", "LM", "CM", "CM", "RM", "ST", "ST"]

def get_base_positions(formation_type, attack_dir):
    """Calculates home coordinates for the chosen formation."""
    w, h = settings.SCREEN_WIDTH, settings.SCREEN_HEIGHT
    
    # Left-to-right representation
    if formation_type == "4-3-3":
        xs = [0.05, 0.2, 0.15, 0.15, 0.2, 0.3, 0.4, 0.4, 0.7, 0.75, 0.7]
        ys = [0.5, 0.15, 0.35, 0.65, 0.85, 0.5, 0.3, 0.7, 0.15, 0.5, 0.85]
    else: # 4-4-2
        xs = [0.05, 0.2, 0.15, 0.15, 0.2, 0.4, 0.35, 0.35, 0.4, 0.75, 0.75]
        ys = [0.5, 0.15, 0.35, 0.65, 0.85, 0.15, 0.35, 0.65, 0.85, 0.4, 0.6]
        
    pos = []
    for x, y in zip(xs, ys):
        if attack_dir == -1:
            x = 1.0 - x # flip for Team B
        pos.append((x * w, y * h))
    return pos

def get_tactical_target(role, base_pos, ball_pos, attack_dir):
    """Dynamic shape logic. Modifies target zone based on role and ball."""
    target = pygame.math.Vector2(base_pos)
    ball_vec = pygame.math.Vector2(ball_pos)
    
    # Base shift toward ball
    shift_x = (ball_vec.x - target.x) * 0.3
    shift_y = (ball_vec.y - target.y) * 0.2
    
    # Role-specific tweaks
    if role in ["LW", "RW", "LM", "RM"]:
        shift_x *= 1.2
        # Stay wide unless ball is very close
        if abs(ball_vec.y - target.y) > 200:
            shift_y *= 0.2
    elif role == "CDM":
        shift_x *= 0.5 # Stays deeper to protect
    elif role in ["CB", "LB", "RB"]:
        shift_x *= 0.4 # Defensive line stays compact
        if role == "CB": 
            shift_y *= 0.5
    elif role == "ST":
        shift_x = (ball_vec.x - target.x) * 0.6 # Press high
        shift_y *= 0.5

    target.x += shift_x
    target.y += shift_y

    w, h = settings.SCREEN_WIDTH, settings.SCREEN_HEIGHT
    r = settings.PLAYER_RADIUS
    target.x = max(r, min(target.x, w - r))
    target.y = max(r, min(target.y, h - r))
    return target

