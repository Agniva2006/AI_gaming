import pygame
from enum import Enum
from engine import settings

class TacticalStrategy(Enum):
    PARK_THE_BUS = "park_the_bus"
    BALANCED = "balanced"
    HIGH_PRESS = "high_press"
    ALL_OUT_ATTACK = "all_out_attack"

class DynamicManagerAI:
    """
    In-game Manager AI that adjusts tactical strategies dynamically based on score
    and remaining match clock.
    """
    def __init__(self, team):
        self.team = team
        self.strategy = TacticalStrategy.BALANCED

    def update_tactics(self, match):
        time_left = match.match_duration - match.time_elapsed
        my_score = match.score[self.team.team_id]
        opp_score = match.score[1 - self.team.team_id]
        score_diff = my_score - opp_score

        if time_left < 45.0: # Final stretch of match
            if score_diff < 0: # Trailing -> All Out Attack
                self.strategy = TacticalStrategy.ALL_OUT_ATTACK
            elif score_diff > 0: # Leading -> Park The Bus
                self.strategy = TacticalStrategy.PARK_THE_BUS
            else:
                self.strategy = TacticalStrategy.HIGH_PRESS
        else:
            self.strategy = TacticalStrategy.BALANCED

    def get_tactical_shift(self):
        """Returns x-offset multiplier for formation target positions."""
        if self.strategy == TacticalStrategy.ALL_OUT_ATTACK:
            return 1.4
        elif self.strategy == TacticalStrategy.HIGH_PRESS:
            return 1.2
        elif self.strategy == TacticalStrategy.PARK_THE_BUS:
            return 0.6
        return 1.0
