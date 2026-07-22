import pygame
from entities.entity import Entity
from engine import settings

class Ball(Entity):
    """
    3D Football entity with gravity, Magnus spin effect, bounce damping,
    and 3D pitch collision resolution.
    """
    GRAVITY = 600.0  # px/s² downwards acceleration

    def __init__(self, x, y):
        super().__init__(x, y, settings.BALL_RADIUS, settings.BALL_COLOR, z=0.0)
        self.friction = settings.BALL_FRICTION
        self.stop_threshold = settings.BALL_STOP_THRESHOLD
        self.bounce_damping = 0.65  # vertical bounce velocity retention
        self.wall_bounce_damping = 0.7
        self.spin = 0.0  # Magnus effect lateral spin (-1.0 to 1.0)
        self.air_drag = 0.15

    def update(self, dt):
        # 1. 2D Ground friction / Air drag
        speed_2d = self.velocity.length()
        if self.z <= 0:
            if speed_2d > self.stop_threshold:
                drag = self.friction * dt
                new_speed = max(0.0, speed_2d - drag)
                self.velocity = self.velocity.normalize() * new_speed
            else:
                self.velocity = pygame.math.Vector2(0, 0)
        else:
            # Air drag scaling
            if speed_2d > 0:
                new_speed = max(0.0, speed_2d * (1.0 - self.air_drag * dt))
                self.velocity = self.velocity.normalize() * new_speed

        # 2. Magnus effect spin curving (cross product of velocity and spin vector)
        if abs(self.spin) > 0.01 and speed_2d > 50:
            # Perpendicular vector to current velocity
            perp = pygame.math.Vector2(-self.velocity.y, self.velocity.x).normalize()
            curve_force = perp * (self.spin * speed_2d * 1.8)
            self.velocity += curve_force * dt
            # Spin decays over time
            self.spin *= (1.0 - 0.5 * dt)

        # 3. Gravity on z-axis
        if self.z > 0 or self.velocity_3d.z > 0:
            self.velocity_3d.z -= self.GRAVITY * dt
        
        # 4. Integrate position using base Entity update
        super().update(dt)

        # 5. Ground bounce check
        if self.position_3d.z <= 0:
            self.position_3d.z = 0.0
            if abs(self.velocity_3d.z) > 40:
                self.velocity_3d.z = -self.velocity_3d.z * self.bounce_damping
            else:
                self.velocity_3d.z = 0.0

        # 6. Pitch wall bounces
        self._bounce_off_walls()

    def kick(self, direction, power, lift=0.0, spin=0.0):
        """
        Apply 3D impulse to the ball.
        direction: Vector2 or Vector3
        power: ground speed in px/s
        lift: upward velocity in px/s for lofted passes or chip shots
        spin: curve force (-1.0 to 1.0)
        """
        dir_vec = pygame.math.Vector2(direction)
        if dir_vec.length_squared() > 0:
            dir_vec = dir_vec.normalize()
            self.velocity_3d.x = dir_vec.x * power
            self.velocity_3d.y = dir_vec.y * power
            self.velocity_3d.z = float(lift)
            self.spin = float(spin)

    def _bounce_off_walls(self):
        gw = settings.GOAL_WIDTH
        gt = settings.GOAL_TOP
        gb = settings.GOAL_BOTTOM
        w = settings.SCREEN_WIDTH
        h = settings.SCREEN_HEIGHT

        # 1. Pitch Top & Bottom Boundaries
        if self.position.y - self.radius < 0:
            self.position.y = self.radius
            self.velocity_3d.y *= -self.wall_bounce_damping

        if self.position.y + self.radius > h:
            self.position.y = h - self.radius
            self.velocity_3d.y *= -self.wall_bounce_damping

        # 2. Left Side (Pitch & Goal Net Box)
        if self.position.x - self.radius < 0:
            if gt <= self.position.y <= gb:
                # Inside left goal mouth -> back net boundary is -gw
                if self.position.x - self.radius < -gw:
                    self.position.x = -gw + self.radius
                    self.velocity_3d.x *= -self.wall_bounce_damping
                # Net top & bottom post boundary inside goal
                if self.position.y - self.radius < gt:
                    self.position.y = gt + self.radius
                    self.velocity_3d.y *= -self.wall_bounce_damping
                elif self.position.y + self.radius > gb:
                    self.position.y = gb - self.radius
                    self.velocity_3d.y *= -self.wall_bounce_damping
            else:
                # Outside goal mouth -> pitch boundary is x = 0
                self.position.x = self.radius
                self.velocity_3d.x *= -self.wall_bounce_damping

        # 3. Right Side (Pitch & Goal Net Box)
        if self.position.x + self.radius > w:
            if gt <= self.position.y <= gb:
                # Inside right goal mouth -> back net boundary is w + gw
                if self.position.x + self.radius > w + gw:
                    self.position.x = w + gw - self.radius
                    self.velocity_3d.x *= -self.wall_bounce_damping
                # Net top & bottom post boundary inside goal
                if self.position.y - self.radius < gt:
                    self.position.y = gt + self.radius
                    self.velocity_3d.y *= -self.wall_bounce_damping
                elif self.position.y + self.radius > gb:
                    self.position.y = gb - self.radius
                    self.velocity_3d.y *= -self.wall_bounce_damping
            else:
                # Outside goal mouth -> pitch boundary is x = w
                self.position.x = w - self.radius
                self.velocity_3d.x *= -self.wall_bounce_damping


