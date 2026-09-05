import pygame
from enum import Enum
from engine import settings

class TacticalStrategy(Enum):
    BALANCED = "balanced"
    TIKI_TAKA = "tiki_taka"
    GEGENPRESS = "gegenpress"
    COUNTER_ATTACK = "counter_attack"
    PARK_THE_BUS = "park_the_bus"

TACTICAL_PROFILES = {
    TacticalStrategy.BALANCED: {
        "name": "Balanced",
        "line_shift": 1.0,
        "press_dist": 240.0,
        "tempo": 1.0,
        "pass_style": "mixed",
        "description": "Standard balanced tactical shape maintaining formation discipline."
    },
    TacticalStrategy.TIKI_TAKA: {
        "name": "Tiki-Taka",
        "line_shift": 1.15,
        "press_dist": 220.0,
        "tempo": 1.05,
        "pass_style": "short",
        "description": "High possession, compact midfield triangles, and patient short passing."
    },
    TacticalStrategy.GEGENPRESS: {
        "name": "Gegenpress",
        "line_shift": 1.30,
        "press_dist": 340.0,
        "tempo": 1.15,
        "pass_style": "high_tempo",
        "description": "Aggressive high defensive line, intense pressing to win immediate turnovers."
    },
    TacticalStrategy.COUNTER_ATTACK: {
        "name": "Counter-Attack",
        "line_shift": 0.75,
        "press_dist": 180.0,
        "tempo": 1.20,
        "pass_style": "direct_through",
        "description": "Compact deep low block with rapid through-balls to sprinting attackers."
    },
    TacticalStrategy.PARK_THE_BUS: {
        "name": "Park The Bus",
        "line_shift": 0.60,
        "press_dist": 140.0,
        "tempo": 0.90,
        "pass_style": "clearance",
        "description": "Ultra-defensive low block inside penalty area to protect the lead."
    }
}

class DynamicManagerAI:
    """
    Tactical Strategy Manager:
    Controls team mentality, pressing distance, defensive line height,
    and adaptive game-state transitions (e.g. chasing the game or defending a lead).
    """
    def __init__(self, team, strategy=TacticalStrategy.BALANCED):
        self.team = team
        self.strategy = strategy

    def set_strategy(self, strategy: TacticalStrategy):
        self.strategy = strategy

    def get_profile(self):
        return TACTICAL_PROFILES.get(self.strategy, TACTICAL_PROFILES[TacticalStrategy.BALANCED])

    def update_tactics(self, match):
        """AI team dynamically adapts strategy based on score and time remaining."""
        if self.team.team_id == 0:
            # Human team tactics are controlled by the user
            return

        time_left = match.match_duration - match.time_elapsed
        my_score = match.score[self.team.team_id]
        opp_score = match.score[1 - self.team.team_id]
        score_diff = my_score - opp_score

        # If trailing in second half -> Gegenpress / High Press
        if time_left < 50.0 and score_diff < 0:
            self.strategy = TacticalStrategy.GEGENPRESS
        # If leading in final stretch -> Counter-Attack or Park The Bus
        elif time_left < 40.0 and score_diff > 0:
            self.strategy = TacticalStrategy.COUNTER_ATTACK if score_diff == 1 else TacticalStrategy.PARK_THE_BUS
        elif score_diff == 0 and time_left < 30.0:
            self.strategy = TacticalStrategy.TIKI_TAKA
        else:
            self.strategy = TacticalStrategy.BALANCED

    def get_tactical_shift(self):
        """Line shift multiplier for target positions."""
        return self.get_profile()["line_shift"]

    def get_pressing_distance(self):
        return self.get_profile()["press_dist"]
