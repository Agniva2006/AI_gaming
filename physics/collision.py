import pygame
import math
from engine import settings

class CollisionSystem:
    """
    Physical Collision & Ball Dribble Interaction System:
    - Smooth 2D dribble contact & ball cushioning (no erratic bouncing).
    - Physical player-player elastic body collisions & shoulder barges.
    - Realistic goalpost (woodwork) bounces with restitution.
    - Pitch boundary clamping.
    """
    POST_RADIUS = 6.0

    def __init__(self, all_players, ball):
        self.players = all_players
        self.ball = ball

        # 4 Goalposts in world space
        w = settings.SCREEN_WIDTH
        gt = settings.GOAL_TOP
        gb = settings.GOAL_BOTTOM
        self.goalposts = [
            pygame.math.Vector2(0, gt),       # Left top post
            pygame.math.Vector2(0, gb),       # Left bottom post
            pygame.math.Vector2(w, gt),       # Right top post
            pygame.math.Vector2(w, gb),       # Right bottom post
        ]

    def update(self):
        """Executes full collision and dribble pipeline."""
        self._resolve_goalpost_collisions()
        self._resolve_player_ball_dribble()
        self._resolve_player_player_collisions()
        self._clamp_player_pitch_bounds()

    def _resolve_goalpost_collisions(self):
        """Checks if the ball hits any of the 4 goalposts (woodwork)."""
        for post in self.goalposts:
            dist = self.ball.position.distance_to(post)
            min_dist = self.ball.radius + self.POST_RADIUS

            # If ball hits the post at ground/crossbar height
            if dist < min_dist and dist > 0 and self.ball.z < 85.0:
                normal = (self.ball.position - post).normalize()
                # Push ball out of collision
                overlap = min_dist - dist
                self.ball.position += normal * overlap

                # Reflect ball velocity with post bounce restitution
                current_speed = self.ball.velocity.length()
                if current_speed > 30:
                    reflected = self.ball.velocity.reflect(normal)
                    self.ball.velocity = reflected * 0.85
                    self.ball.velocity_3d.x = self.ball.velocity.x
                    self.ball.velocity_3d.y = self.ball.velocity.y

    def _resolve_player_ball_dribble(self):
        """
        Realistic 2D Dribbling & Ball Cushioning:
        When a player is near the ball and facing it, the player cushions the ball
        smoothly in front of their feet instead of ping-pong bouncing.
        """
        for player in self.players:
            is_jockey = getattr(player, "is_jockeying", False)
            touch_radius = player.radius + self.ball.radius + (18.0 if is_jockey else 7.0)
            dist_2d = player.position.distance_to(self.ball.position)

            # Ball is within player's immediate feet control zone
            if dist_2d < touch_radius and dist_2d > 0 and self.ball.z <= 40.0:
                to_ball = (self.ball.position - player.position).normalize()
                facing = player.facing.copy()
                if facing.length_squared() > 0:
                    facing = facing.normalize()

                # Defensive Jockey Interception
                if is_jockey and player.velocity.length() < 160:
                    self.ball.velocity *= 0.20
                    self.ball.velocity_3d.x = self.ball.velocity.x
                    self.ball.velocity_3d.y = self.ball.velocity.y
                    self.ball.position = player.position + facing * (player.radius + self.ball.radius + 2.0)
                    self.ball.last_touched_by = getattr(player, 'id', None)
                    continue

                player_speed = player.velocity.length()
                is_sprinting = getattr(player, "is_sprinting", False)

                if player_speed > 30:
                    # Dynamic First Touch / Dribble lead distance
                    # Sprinting = heavy touch (ball pushed further ahead)
                    # Controlled run = velvet cushion close to feet
                    lead_dist = player.radius + self.ball.radius + (18.0 if is_sprinting else 4.0)
                    target_ball_pos = player.position + facing * lead_dist

                    lerp_factor = 0.32 if is_sprinting else 0.52
                    self.ball.position += (target_ball_pos - self.ball.position) * lerp_factor

                    dribble_speed = player_speed * (1.10 if is_sprinting else 1.02)
                    self.ball.velocity = facing * dribble_speed
                    self.ball.velocity_3d.x = self.ball.velocity.x
                    self.ball.velocity_3d.y = self.ball.velocity.y
                    self.ball.last_touched_by = getattr(player, 'id', None)
                else:
                    min_dist = player.radius + self.ball.radius
                    if dist_2d < min_dist:
                        overlap = min_dist - dist_2d
                        self.ball.position += to_ball * overlap
                        self.ball.velocity *= 0.5

    def _resolve_player_player_collisions(self):
        """Resolves elastic physical collisions and shoulder barges between players."""
        num_players = len(self.players)
        for i in range(num_players):
            for j in range(i + 1, num_players):
                p1 = self.players[i]
                p2 = self.players[j]

                diff = p2.position - p1.position
                distance = diff.length()
                min_dist = p1.radius + p2.radius

                if distance < min_dist and distance > 0:
                    overlap = min_dist - distance
                    normal = diff / distance

                    # Displace players equally out of collision
                    p1.position -= normal * (overlap * 0.5)
                    p2.position += normal * (overlap * 0.5)

                    # Transfer physical momentum / shoulder barge
                    relative_vel = p1.velocity - p2.velocity
                    vel_along_normal = relative_vel.dot(normal)

                    if vel_along_normal > 0:
                        restitution = 0.4
                        impulse_mag = (1.0 + restitution) * vel_along_normal * 0.5
                        impulse = normal * impulse_mag
                        p1.velocity -= impulse
                        p2.velocity += impulse

    def _clamp_player_pitch_bounds(self):
        """Keeps players within the pitch boundaries."""
        w, h = settings.SCREEN_WIDTH, settings.SCREEN_HEIGHT
        for player in self.players:
            r = player.radius
            player.position.x = max(r, min(player.position.x, w - r))
            player.position.y = max(r, min(player.position.y, h - r))
