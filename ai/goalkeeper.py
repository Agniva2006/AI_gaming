import pygame
from enum import Enum
from engine import settings


class GKState(Enum):
    POSITIONING = "positioning"     # tracking ball Y on goal line
    DIVING = "diving"               # lunging toward predicted ball impact
    COMING_OUT = "coming_out"       # rushing out for a 1v1
    RECOVERY = "recovery"           # returning to position after dive


class GoalkeeperAI:
    """
    Dedicated goalkeeper AI with shot prediction and diving.
    Separated from the main AIController because GK logic is fundamentally
    different — it's reactive (responding to shots) rather than proactive.

    Design: The GK predicts where a shot will cross the goal line and
    dives to that Y position. This is how real game AI does it —
    intercept prediction, not just chasing the ball.
    """
    def __init__(self, player, ball, team):
        self.player = player
        self.ball = ball
        self.team = team

        self.state = GKState.POSITIONING
        self.recovery_timer = 0.0
        self.dive_target = None

        # Which side is our goal?
        # Team A (attacks right) → GK defends LEFT (x ≈ 0)
        # Team B (attacks left) → GK defends RIGHT (x ≈ SCREEN_WIDTH)
        if team.attack_direction == 1:
            self.goal_line_x = self.player.home_position.x  # ~80
            self.ball_approaching = lambda vx: vx < 0       # ball moving left toward us
        else:
            self.goal_line_x = self.player.home_position.x  # ~1200
            self.ball_approaching = lambda vx: vx > 0       # ball moving right toward us

    def update(self, dt):
        """Main GK update — runs the state machine."""
        if self.state == GKState.POSITIONING:
            self._state_positioning(dt)
        elif self.state == GKState.DIVING:
            self._state_diving(dt)
        elif self.state == GKState.COMING_OUT:
            self._state_coming_out(dt)
        elif self.state == GKState.RECOVERY:
            self._state_recovery(dt)

    # --- States ---

    def _state_positioning(self, dt):
        """Default state: track ball Y on the goal line."""
        # Check if a shot is incoming
        ball_speed = self.ball.velocity.length()
        if ball_speed > settings.SHOT_SPEED_THRESHOLD and self.ball_approaching(self.ball.velocity.x):
            predicted_y = self._predict_intercept_y()
            if predicted_y is not None:
                # It's a shot on target — DIVE!
                self.dive_target = pygame.math.Vector2(self.goal_line_x, predicted_y)
                self.state = GKState.DIVING
                return

        # Check for 1v1 situation (attacker close with ball)
        nearest_opponent = self._nearest_opponent_with_ball()
        if nearest_opponent:
            dist = nearest_opponent.position.distance_to(self.player.position)
            if dist < settings.GK_COME_OUT_DISTANCE:
                self.state = GKState.COMING_OUT
                return

        # Normal positioning — track ball Y on goal line
        target_y = self.ball.position.y
        target_y = max(settings.GOAL_TOP + self.player.radius,
                       min(target_y, settings.GOAL_BOTTOM - self.player.radius))

        target = pygame.math.Vector2(self.goal_line_x, target_y)
        self._move_toward(target, settings.GK_NORMAL_SPEED)

    def _state_diving(self, dt):
        """Dive toward the predicted intercept point."""
        if self.dive_target is None:
            self.state = GKState.RECOVERY
            return

        dist = self.player.position.distance_to(self.dive_target)
        if dist < 10:
            # Reached dive target — enter recovery
            self.player.velocity = pygame.math.Vector2(0, 0)
            self.state = GKState.RECOVERY
            self.recovery_timer = settings.GK_RECOVERY_TIME
            return

        # Check if ball has passed us (missed the save or was saved)
        ball_past = False
        if self.team.attack_direction == 1:
            ball_past = self.ball.position.x < self.goal_line_x - 50
        else:
            ball_past = self.ball.position.x > self.goal_line_x + 50

        if ball_past or self.ball.velocity.length() < settings.BALL_STOP_THRESHOLD:
            self.state = GKState.RECOVERY
            self.recovery_timer = settings.GK_RECOVERY_TIME
            return

        # Dive at high speed
        self._move_toward(self.dive_target, settings.GK_DIVE_SPEED)

    def _state_coming_out(self, dt):
        """Rush toward the ball for a 1v1."""
        nearest = self._nearest_opponent_with_ball()
        if nearest is None:
            self.state = GKState.POSITIONING
            return

        dist = self.player.position.distance_to(nearest.position)
        if dist > settings.GK_COME_OUT_DISTANCE * 1.5:
            # Opponent moved away — go back
            self.state = GKState.POSITIONING
            return

        # Rush toward the ball
        self._move_toward(self.ball.position, settings.GK_DIVE_SPEED * 0.8)

    def _state_recovery(self, dt):
        """Return to goal center after a dive."""
        self.recovery_timer -= dt
        if self.recovery_timer <= 0:
            self.state = GKState.POSITIONING
            return

        # Slowly return to home position
        home = pygame.math.Vector2(self.player.home_position)
        self._move_toward(home, settings.GK_NORMAL_SPEED * 0.5)

    # --- Utilities ---

    def _predict_intercept_y(self):
        """
        Predict where the ball will cross our goal line.
        Uses simple linear projection (ignoring friction for speed).
        Returns None if ball won't reach goal line.
        """
        bx = self.ball.position.x
        by = self.ball.position.y
        bvx = self.ball.velocity.x
        bvy = self.ball.velocity.y

        if abs(bvx) < 1:
            return None  # ball barely moving horizontally

        # Time to reach goal line
        time_to_goal = (self.goal_line_x - bx) / bvx
        if time_to_goal < 0:
            return None  # ball moving away

        # Predicted Y at intercept
        predicted_y = by + bvy * time_to_goal

        # Check if it's on target (within goal posts)
        if settings.GOAL_TOP <= predicted_y <= settings.GOAL_BOTTOM:
            return predicted_y

        return None  # off target — let it go

    def _nearest_opponent_with_ball(self):
        """Find proximity to goal area when opponent ball carrier approaches."""
        ball_dist_to_goal = abs(self.ball.position.x - self.goal_line_x)
        if ball_dist_to_goal < settings.GK_COME_OUT_DISTANCE:
            if self.ball.velocity.length() < 250:
                return self.ball
        return None


    def _move_toward(self, target, speed):
        """Set player velocity toward a target."""
        direction = target - self.player.position
        if direction.length_squared() > 0:
            direction = direction.normalize()
            self.player.facing = direction.copy()
        self.player.velocity = direction * speed
