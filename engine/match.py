import pygame
from engine import settings

class Match:
    """
    Manages match state: score, goal detection, kickoffs, and resets.
    """
    def __init__(self, team_a, team_b, ball):
        self.team_a = team_a        # attacks right
        self.team_b = team_b        # attacks left
        self.ball = ball
        self.score = {0: 0, 1: 0}   # team_id -> goals

        self.state = "PLAYING"       # PLAYING, GOAL_SCORED, KICKOFF_PAUSE, FULL_TIME
        self.pause_timer = 0.0
        self.last_scorer = None
        
        # Phase C: 3 minute match
        self.time_elapsed = 0.0
        self.match_duration = 180.0

    def update(self, dt):
        """Check for goals and handle match state transitions."""
        if self.state == "PLAYING":
            self.time_elapsed += dt
            if self.time_elapsed >= self.match_duration:
                self.state = "FULL_TIME"
                return
            self._check_goals()
        elif self.state == "GOAL_SCORED":
            self.pause_timer -= dt
            if self.pause_timer <= 0:
                self._kickoff()
                self.state = "PLAYING"

    def _check_goals(self):
        bx = self.ball.position.x
        by = self.ball.position.y

        # Right goal -> Team A scores
        if bx + self.ball.radius >= settings.SCREEN_WIDTH:
            if settings.GOAL_TOP <= by <= settings.GOAL_BOTTOM:
                self._goal_scored(0)
                return

        # Left goal -> Team B scores
        if bx - self.ball.radius <= 0:
            if settings.GOAL_TOP <= by <= settings.GOAL_BOTTOM:
                self._goal_scored(1)
                return

    def _goal_scored(self, team_id):
        self.score[team_id] += 1
        self.last_scorer = team_id
        self.state = "GOAL_SCORED"
        self.pause_timer = settings.KICKOFF_PAUSE
        self.ball.velocity = pygame.math.Vector2(0, 0)

    def _kickoff(self):
        self.team_a.reset_positions()
        self.team_b.reset_positions()
        self.ball.position = pygame.math.Vector2(settings.SCREEN_WIDTH // 2, settings.SCREEN_HEIGHT // 2)
        self.ball.velocity = pygame.math.Vector2(0, 0)

    @property
    def is_playing(self):
        return self.state == "PLAYING"
