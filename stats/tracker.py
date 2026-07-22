class MatchStats:
    """
    Phase D: Tracks advanced statistics throughout the match.
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
        """Called every frame to evaluate who is closest to the ball."""
        if closest_player:
            self.possession_ticks[closest_player.team_id] += 1
            
            # Detect change in possession (tackle/recovery)
            if self.last_possessor and self.last_possessor.team_id != closest_player.team_id:
                self.tackles[closest_player.team_id] += 1
                
            self.last_possessor = closest_player

    def record_pass(self, team_id):
        self.passes_attempted[team_id] += 1

    def record_shot(self, team_id):
        self.shots[team_id] += 1

    def get_possession_pct(self):
        total = max(1, sum(self.possession_ticks.values()))
        return {
            0: (self.possession_ticks[0] / total) * 100,
            1: (self.possession_ticks[1] / total) * 100
        }

# Global singleton for easy tracking without dependency injection
match_stats = MatchStats()
