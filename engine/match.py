import pygame
from engine import settings

class Match:
    """
    Manages football match lifecycle:
    Match clock, score tracking, cumulative xG, goal detection,
    offside infractions, and free kick / kickoff resets.
    """
    def __init__(self, team_a, team_b, ball, duration=None):
        self.team_a = team_a        # Attacks right (Human team)
        self.team_b = team_b        # Attacks left (RL AI team)
        self.ball = ball
        self.score = {0: 0, 1: 0}   # team_id -> goals
        self.cumulative_xg = {0: 0.0, 1: 0.0}  # Expected Goals
        self.shot_events = []        # Detailed Opta-style shot event map

        self.state = "PLAYING"       # PLAYING, GOAL_SCORED, OFFSIDE, FULL_TIME
        self.pause_timer = 0.0
        self.last_scorer = None
        self.offside_team = None
        self.time_elapsed = 0.0
        self.match_duration = duration if duration is not None else settings.MATCH_DURATION

    def update(self, dt):
        """Advances match clock and checks for goals or match events."""
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
        elif self.state == "OFFSIDE":
            self.pause_timer -= dt
            if self.pause_timer <= 0:
                self._restart_free_kick()
                self.state = "PLAYING"

    def _check_goals(self):
        bx = self.ball.position.x
        by = self.ball.position.y

        # Right goal -> Team A (Human) scores
        if bx + self.ball.radius >= settings.SCREEN_WIDTH - 2:
            if settings.GOAL_TOP <= by <= settings.GOAL_BOTTOM:
                self._goal_scored(0)
                return

        # Left goal -> Team B (AI) scores
        if bx - self.ball.radius <= 2:
            if settings.GOAL_TOP <= by <= settings.GOAL_BOTTOM:
                self._goal_scored(1)
                return

    def _goal_scored(self, team_id):
        self.score[team_id] += 1
        self.last_scorer = team_id
        if self.shot_events and self.shot_events[-1]["team_id"] == team_id:
            self.shot_events[-1]["result"] = "GOAL"

        self.state = "GOAL_SCORED"
        self.pause_timer = settings.KICKOFF_PAUSE
        self.ball.velocity_3d = pygame.math.Vector3(0, 0, 0)
        self.ball.velocity = pygame.math.Vector2(0, 0)
        self.ball.z = 0.0

    def record_shot(self, team_id, xg, shot_pos=None):
        """Records a shot attempt with its calculated Expected Goal value and coordinates."""
        self.cumulative_xg[team_id] += xg
        sx, sy = shot_pos if shot_pos is not None else (self.ball.position.x, self.ball.position.y)
        self.shot_events.append({
            "team_id": team_id,
            "x": round(float(sx), 1),
            "y": round(float(sy), 1),
            "xg": round(float(xg), 3),
            "result": "SAVED/MISS",
            "time": round(float(self.time_elapsed), 1)
        })

    def trigger_offside(self, team_id):
        """Flags an offside infringement, pausing play for free-kick restart."""
        if self.state == "PLAYING":
            self.state = "OFFSIDE"
            self.offside_team = team_id
            self.pause_timer = 1.2
            self.ball.velocity = pygame.math.Vector2(0, 0)
            self.ball.velocity_3d = pygame.math.Vector3(0, 0, 0)

    def trigger_net_ripple(self, renderer):
        """Generates physics ripple and spark particles on goal nets."""
        if self.last_scorer == 0 and hasattr(renderer, 'right_net'):
            renderer.right_net.trigger_impact(self.ball.position.y, power=270.0)
            if hasattr(renderer, 'particle_system'):
                renderer.particle_system.emit_goal_sparks(settings.SCREEN_WIDTH, self.ball.position.y, count=35)
        elif self.last_scorer == 1 and hasattr(renderer, 'left_net'):
            renderer.left_net.trigger_impact(self.ball.position.y, power=270.0)
            if hasattr(renderer, 'particle_system'):
                renderer.particle_system.emit_goal_sparks(0, self.ball.position.y, count=35)

    def _kickoff(self):
        self.team_a.reset_positions()
        self.team_b.reset_positions()
        self.ball.position = pygame.math.Vector2(settings.SCREEN_WIDTH // 2, settings.SCREEN_HEIGHT // 2)
        self.ball.velocity = pygame.math.Vector2(0, 0)
        self.ball.velocity_3d = pygame.math.Vector3(0, 0, 0)
        self.ball.z = 0.0

    def _restart_free_kick(self):
        """Restarts play from defending team's free kick position."""
        if self.offside_team == 0:
            # Team A was offside -> Team B takes free kick from their defensive half
            restart_x = settings.SCREEN_WIDTH * 0.65
        else:
            restart_x = settings.SCREEN_WIDTH * 0.35
        self.ball.position = pygame.math.Vector2(restart_x, settings.SCREEN_HEIGHT // 2)
        self.ball.velocity = pygame.math.Vector2(0, 0)
        self.ball.velocity_3d = pygame.math.Vector3(0, 0, 0)

    @property
    def is_playing(self):
        return self.state == "PLAYING"

    def get_result_string(self):
        a = self.score[0]
        b = self.score[1]
        if a > b:
            return "WIN"
        elif a < b:
            return "LOSS"
        return "DRAW"
