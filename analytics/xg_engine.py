import math
import pygame
from engine import settings

class XGEngine:
    """
    Mathematical Expected Goals (xG) Analytics Engine:
    Calculates the statistical probability of a shot resulting in a goal (0.01 to 0.99)
    based on Euclidean distance to goal mouth, geometric visual angle to goal posts,
    goalkeeper proximity, and player shooting skill.
    """
    @staticmethod
    def calculate_xg(shooter_pos, target_goal_center, goalkeeper=None, shooter_profile=None, is_header=False):
        sx, sy = shooter_pos[0], shooter_pos[1]
        gx, gy = target_goal_center[0], target_goal_center[1]

        # 1. Distance to goal center in meters (assuming pitch is ~105m wide, 1280px = 105m -> 1m = 12.2px)
        dist_px = math.hypot(gx - sx, gy - sy)
        dist_m = max(1.0, dist_px / 12.2)

        # 2. Goal mouth visual angle theta (goal width in y is GOAL_HEIGHT = 200px)
        post_top = (gx, settings.GOAL_TOP)
        post_bot = (gx, settings.GOAL_BOTTOM)

        d_top = math.hypot(post_top[0] - sx, post_top[1] - sy)
        d_bot = math.hypot(post_bot[0] - sx, post_bot[1] - sy)
        goal_w = settings.GOAL_HEIGHT

        # Law of cosines to compute angle subtended by the goal mouth
        cos_theta = (d_top**2 + d_bot**2 - goal_w**2) / (2.0 * max(1.0, d_top * d_bot))
        cos_theta = max(-1.0, min(1.0, cos_theta))
        angle_rad = math.acos(cos_theta)
        angle_deg = math.degrees(angle_rad)

        # 3. Goalkeeper positioning factor (is GK covering the angle?)
        gk_penalty = 0.0
        if goalkeeper:
            gk_x, gk_y = goalkeeper.position.x, goalkeeper.position.y
            gk_dist = math.hypot(gk_x - gx, gk_y - gy)
            # If GK is on the goal line between posts, shot is contested
            if settings.GOAL_TOP <= gk_y <= settings.GOAL_BOTTOM and gk_dist < 80:
                gk_penalty = 0.45

        # 4. Logistic regression xG model
        # Base weights calibrated to standard professional football data
        beta_0 = 0.85
        beta_dist = -0.12 * dist_m
        beta_angle = 0.035 * angle_deg
        beta_gk = -gk_penalty

        logit = beta_0 + beta_dist + beta_angle + beta_gk

        # Penalize headers/aerial volleys slightly
        if is_header:
            logit -= 0.6

        # Player shooting attribute modifier
        if shooter_profile and hasattr(shooter_profile, "shooting"):
            shoot_bonus = (shooter_profile.shooting - 75) * 0.02
            logit += shoot_bonus

        # Sigmoid activation
        xg = 1.0 / (1.0 + math.exp(-max(-6.0, min(6.0, logit))))

        # Clamp between 0.02 and 0.95
        return round(float(max(0.02, min(0.95, xg))), 3)

