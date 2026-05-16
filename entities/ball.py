import pygame
from entities.entity import Entity
from engine import settings


class Ball(Entity):
    """
    The football. Moves via physics (friction, bouncing), not keyboard input.
    Other systems apply force to it through the kick() method.
    """
    def __init__(self, x, y):
        super().__init__(x, y, settings.BALL_RADIUS, settings.BALL_COLOR)
        self.friction = settings.BALL_FRICTION           # deceleration in px/s²
        self.stop_threshold = settings.BALL_STOP_THRESHOLD
        self.bounce_damping = 0.7   # ball keeps 70% speed on wall bounce

    def update(self, dt):
        speed = self.velocity.length()

        if speed > self.stop_threshold:
            # Linear drag: reduce speed by friction * dt each frame
            # This is frame-rate independent because we scale by dt
            drag = self.friction * dt
            new_speed = max(0, speed - drag)
            self.velocity = self.velocity.normalize() * new_speed
        else:
            # Ball is nearly stopped — kill all velocity to prevent drift
            self.velocity = pygame.math.Vector2(0, 0)

        # Apply movement (parent class does: position += velocity * dt)
        super().update(dt)

        # Bounce off pitch boundaries
        self._bounce_off_walls()

    def kick(self, direction, power):
        """
        Apply an impulse to the ball.
        direction: Vector2 (will be normalized)
        power: scalar speed in px/s
        """
        if direction.length_squared() > 0:
            self.velocity = direction.normalize() * power

    def _bounce_off_walls(self):
        # Left wall
        if self.position.x - self.radius < 0:
            self.position.x = self.radius
            self.velocity.x *= -self.bounce_damping

        # Right wall
        if self.position.x + self.radius > settings.SCREEN_WIDTH:
            self.position.x = settings.SCREEN_WIDTH - self.radius
            self.velocity.x *= -self.bounce_damping

        # Top wall
        if self.position.y - self.radius < 0:
            self.position.y = self.radius
            self.velocity.y *= -self.bounce_damping

        # Bottom wall
        if self.position.y + self.radius > settings.SCREEN_HEIGHT:
            self.position.y = settings.SCREEN_HEIGHT - self.radius
            self.velocity.y *= -self.bounce_damping
