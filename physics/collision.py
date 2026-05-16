import pygame
from engine import settings


class CollisionSystem:
    """
    Handles all collision detection and resolution for all players and the ball.
    """
    def __init__(self, all_players, ball):
        self.players = all_players      # list of ALL 22 players
        self.ball = ball

    def update(self):
        """Run all collision checks. Called once per frame AFTER entity updates."""
        self._resolve_player_ball_collisions()
        self._resolve_player_player_collisions()

    def _resolve_player_ball_collisions(self):
        for player in self.players:
            distance = player.position.distance_to(self.ball.position)
            min_dist = player.radius + self.ball.radius

            if distance < min_dist and distance > 0:
                overlap = min_dist - distance
                push_dir = (self.ball.position - player.position).normalize()

                # Separate
                self.ball.position += push_dir * overlap

                # Dribble: set ball velocity based on player speed
                player_speed = player.velocity.length()
                if player_speed > 0:
                    self.ball.velocity = push_dir * (player_speed * 1.1)

    def _resolve_player_player_collisions(self):
        for i in range(len(self.players)):
            for j in range(i + 1, len(self.players)):
                p1 = self.players[i]
                p2 = self.players[j]

                distance = p1.position.distance_to(p2.position)
                min_dist = p1.radius + p2.radius

                if distance < min_dist and distance > 0:
                    overlap = min_dist - distance
                    push_dir = (p2.position - p1.position).normalize()

                    p1.position -= push_dir * (overlap / 2)
                    p2.position += push_dir * (overlap / 2)
