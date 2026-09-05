import math
import pygame
from engine import settings

class OpponentTendencyProfiler:
    """
    Opponent Tendency Profiler & Adaptive Intelligence Engine:
    Tracks the human player's tactical habits in real-time (flank bias, passing
    choice, shot locations, dribbling tempo) and generates concrete counter-tactical
    adaptations for the RL AI opponent.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        # Flank & Territorial distribution
        self.left_flank_touches = 0    # Y < SCREEN_HEIGHT * 0.35
        self.center_touches = 0        # 0.35 <= Y <= 0.65
        self.right_flank_touches = 0   # Y > SCREEN_HEIGHT * 0.65

        # Passing habits
        self.short_passes = 0
        self.through_balls = 0
        self.crosses = 0

        # Shooting habits
        self.shots = []  # list of {"x": x, "y": y, "xg": xg, "result": str, "time": t}
        self.total_shot_distance = 0.0

        # Possession & tempo
        self.human_possession_time = 0.0
        self.turnovers_forced = 0

        # Active adaptation states
        self.active_counter_strategy = "STANDARD_SCOUT"
        self.adaptation_log = []

    def record_touch(self, pos):
        """Logs ball touch coordinates of the human player."""
        h = settings.SCREEN_HEIGHT
        y = pos[1] if isinstance(pos, (list, tuple)) else pos.y

        if y < h * 0.35:
            self.left_flank_touches += 1
        elif y > h * 0.65:
            self.right_flank_touches += 1
        else:
            self.center_touches += 1

    def record_pass(self, is_through=False, is_cross=False):
        if is_through:
            self.through_balls += 1
        elif is_cross:
            self.crosses += 1
        else:
            self.short_passes += 1

    def record_shot(self, shot_pos, xg, target_goal=(settings.SCREEN_WIDTH, settings.SCREEN_HEIGHT // 2), result="SAVED", match_time=0.0):
        sx = shot_pos[0] if isinstance(shot_pos, (list, tuple)) else shot_pos.x
        sy = shot_pos[1] if isinstance(shot_pos, (list, tuple)) else shot_pos.y
        dist = math.hypot(target_goal[0] - sx, target_goal[1] - sy)
        self.total_shot_distance += dist

        event = {
            "x": round(float(sx), 1),
            "y": round(float(sy), 1),
            "xg": round(float(xg), 3),
            "result": result,
            "time": round(float(match_time), 1)
        }
        self.shots.append(event)

    def record_turnover(self):
        self.turnovers_forced += 1

    def get_flank_distribution(self):
        tot = max(1, self.left_flank_touches + self.center_touches + self.right_flank_touches)
        return {
            "left": round((self.left_flank_touches / tot) * 100, 1),
            "center": round((self.center_touches / tot) * 100, 1),
            "right": round((self.right_flank_touches / tot) * 100, 1),
            "total_touches": tot
        }

    def get_profile_summary(self):
        flanks = self.get_flank_distribution()
        tot_passes = max(1, self.short_passes + self.through_balls + self.crosses)
        through_pct = round((self.through_balls / tot_passes) * 100, 1)

        # Determine dominant attacking flank
        if flanks["left"] > 48.0:
            favored_flank = "LEFT WING"
        elif flanks["right"] > 48.0:
            favored_flank = "RIGHT WING"
        else:
            favored_flank = "CENTRAL BUILDUP"

        # Determine passing pattern
        pass_style = "VERTICAL THROUGH-BALLS" if through_pct > 35.0 else "PATIENT GROUND PASSING"

        # Calculate average shot distance
        avg_shot_dist = (self.total_shot_distance / max(1, len(self.shots))) / 12.2 # convert to meters

        return {
            "favored_flank": favored_flank,
            "flank_percentages": flanks,
            "through_ball_pct": through_pct,
            "pass_style": pass_style,
            "total_shots": len(self.shots),
            "avg_shot_distance_m": round(avg_shot_dist, 1),
            "turnovers_forced": self.turnovers_forced
        }

    def get_counter_strategy(self):
        """
        Computes dynamic mathematical tactical adjustments for the RL AI team.
        Returns:
            dict containing positional shifts, line height adjustments, and pressing boosts.
        """
        summary = self.get_profile_summary()
        flanks = summary["flank_percentages"]
        through_pct = summary["through_ball_pct"]

        adjustments = {
            "flank_shift_y": 0.0,
            "defensive_line_mult": 1.0,
            "press_dist_mult": 1.0,
            "strategy_name": "Standard Balanced Cover",
            "tactical_debrief": "AI maintaining default disciplined shape."
        }

        # 1. Counter Flank Overload
        if flanks["left"] > 48.0:
            adjustments["flank_shift_y"] = -55.0  # Shift AI RB & RCM upwards to choke Left Wing
            adjustments["press_dist_mult"] = 1.25
            adjustments["strategy_name"] = "Overload Right Defense"
            adjustments["tactical_debrief"] = f"Detected {flanks['left']}% attacks on Left Wing. AI fullbacks shifted to double-team your winger."
        elif flanks["right"] > 48.0:
            adjustments["flank_shift_y"] = 55.0   # Shift AI LB & LCM downwards to choke Right Wing
            adjustments["press_dist_mult"] = 1.25
            adjustments["strategy_name"] = "Overload Left Defense"
            adjustments["tactical_debrief"] = f"Detected {flanks['right']}% attacks on Right Wing. AI shifted defensive block south to cut off crosses."

        # 2. Counter Direct Through-Ball Exploit
        if through_pct > 40.0:
            adjustments["defensive_line_mult"] = 0.80  # Drop line 20% deeper to erase space behind
            adjustments["strategy_name"] += " + Deep Sweeper Cover"
            adjustments["tactical_debrief"] += " Human relying on direct through-balls; AI dropping defensive line to eliminate running channels."

        return adjustments

# Global singleton instance for active match tracking
tendency_profiler = OpponentTendencyProfiler()

