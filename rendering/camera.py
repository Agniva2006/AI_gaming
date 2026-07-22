import pygame
from engine import settings

class Camera:
    """
    Dynamic Camera tracking system with smooth target interpolation (lerp),
    lead-ahead tracking, contextual zoom (zooming in on goal mouth action),
    and screen coordinate transformation.
    """
    def __init__(self, width=settings.SCREEN_WIDTH, height=settings.SCREEN_HEIGHT):
        self.screen_width = width
        self.screen_height = height
        self.target_pos = pygame.math.Vector2(width // 2, height // 2)
        self.current_pos = pygame.math.Vector2(width // 2, height // 2)
        self.target_zoom = 1.0
        self.current_zoom = 1.0
        self.lerp_speed = 4.0  # interpolation speed

    def update(self, ball, match, dt):
        # Base target is ball position with slight lead-ahead based on ball velocity
        lead = ball.velocity * 0.25 if ball.velocity.length() > 50 else pygame.math.Vector2(0, 0)
        target = ball.position + lead

        # Contextual zoom logic: zoom in when near goals or inside penalty box
        dist_left_goal = target.distance_to(pygame.math.Vector2(0, settings.SCREEN_HEIGHT // 2))
        dist_right_goal = target.distance_to(pygame.math.Vector2(settings.SCREEN_WIDTH, settings.SCREEN_HEIGHT // 2))

        if min(dist_left_goal, dist_right_goal) < 280:
            self.target_zoom = 1.25  # Zoom in for goal mouth action
        elif match and match.state == "GOAL_SCORED":
            self.target_zoom = 1.35  # Celebration zoom
        else:
            self.target_zoom = 1.0   # Standard view

        # Smooth Lerp position
        self.current_pos += (target - self.current_pos) * min(1.0, self.lerp_speed * dt)

        # Smooth Lerp zoom
        self.current_zoom += (self.target_zoom - self.current_zoom) * min(1.0, self.lerp_speed * dt)

        # Clamp camera center to keep pitch within view bounds
        half_w = (self.screen_width / 2) / self.current_zoom
        half_h = (self.screen_height / 2) / self.current_zoom
        self.current_pos.x = max(half_w, min(self.current_pos.x, settings.SCREEN_WIDTH - half_w))
        self.current_pos.y = max(half_h, min(self.current_pos.y, settings.SCREEN_HEIGHT - half_h))

    def world_to_screen(self, pos_vec2):
        """Transform 2D world coordinates into screen coordinates based on camera center and zoom."""
        center_x = self.screen_width / 2
        center_y = self.screen_height / 2

        rel_x = (pos_vec2.x - self.current_pos.x) * self.current_zoom
        rel_y = (pos_vec2.y - self.current_pos.y) * self.current_zoom

        return (int(center_x + rel_x), int(center_y + rel_y))

    def scale(self, length):
        """Scale pixel dimensions by current zoom level."""
        return int(length * self.current_zoom)
