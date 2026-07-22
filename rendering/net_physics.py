import pygame
import math
from engine import settings

class GoalNet:
    """
    Spring-mass grid simulation for realistic goal net bulge/ripple effects 
    when shots hit the net.
    """
    def __init__(self, side="left"):
        self.side = side
        self.rows = 8
        self.cols = 5
        self.points = []
        self.velocities = []

        gw = settings.GOAL_WIDTH
        gt = settings.GOAL_TOP
        gh = settings.GOAL_HEIGHT
        w = settings.SCREEN_WIDTH

        start_x = -gw if side == "left" else w
        end_x = 0 if side == "left" else w + gw

        for r in range(self.rows):
            row_pts = []
            row_vels = []
            y = gt + (gh / (self.rows - 1)) * r
            for c in range(self.cols):
                x = start_x + ((end_x - start_x) / (self.cols - 1)) * c
                row_pts.append(pygame.math.Vector2(x, y))
                row_vels.append(pygame.math.Vector2(0, 0))
            self.points.append(row_pts)
            self.velocities.append(row_vels)

    def trigger_impact(self, impact_y, power=200.0):
        """Displace net grid near the impact point to create physical bulge/ripple."""
        for r in range(self.rows):
            for c in range(self.cols):
                pt = self.points[r][c]
                dist = abs(pt.y - impact_y)
                if dist < 60:
                    force = (1.0 - dist / 60) * power
                    direction = -1.0 if self.side == "left" else 1.0
                    self.velocities[r][c].x += direction * force

    def update(self, dt):
        stiffness = 50.0
        damping = 0.88

        for r in range(self.rows):
            for c in range(1, self.cols - 1): # keep edges pinned
                pt = self.points[r][c]
                vel = self.velocities[r][c]

                # Spring force back to rest position
                # (Simple elastic damping)
                vel.x *= damping
                vel.y *= damping
                pt += vel * dt

    def render(self, screen, camera):
        color = (220, 220, 220)
        for r in range(self.rows):
            for c in range(self.cols):
                p1 = camera.world_to_screen(self.points[r][c])
                if c < self.cols - 1:
                    p2 = camera.world_to_screen(self.points[r][c + 1])
                    pygame.draw.line(screen, color, p1, p2, 1)
                if r < self.rows - 1:
                    p3 = camera.world_to_screen(self.points[r + 1][c])
                    pygame.draw.line(screen, color, p1, p3, 1)
