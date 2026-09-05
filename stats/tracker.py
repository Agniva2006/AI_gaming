class MatchStats:
    """
    Tracks comprehensive statistics throughout the match:
    Possession percentages, passes, shots, tackles, and recoveries.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.possession_ticks = {0: 0, 1: 0}
        self.passes_attempted = {0: 0, 1: 0}
        self.shots = {0: 0, 1: 0}
        self.tackles = {0: 0, 1: 0}
        self.last_possessor = None

    def update_possession(self, closest_player):
        """Called every frame to evaluate which player has possession."""
        if closest_player:
            self.possession_ticks[closest_player.team_id] += 1
            if self.last_possessor and self.last_possessor.team_id != closest_player.team_id:
                self.tackles[closest_player.team_id] += 1
            self.last_possessor = closest_player

    def record_pass(self, team_id):
        self.passes_attempted[team_id] += 1

    def record_shot(self, team_id):
        self.shots[team_id] += 1

    def record_tackle(self, team_id):
        self.tackles[team_id] += 1

    def get_possession_pct(self):
        total = max(1, sum(self.possession_ticks.values()))
        return {
            0: round((self.possession_ticks[0] / total) * 100, 1),
            1: round((self.possession_ticks[1] / total) * 100, 1)
        }

# Global singleton
match_stats = MatchStats()
