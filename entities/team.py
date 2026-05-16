import pygame
from entities.players import Player
from engine import settings
from tactics.formations import get_base_positions, ROLES_433, ROLES_442


class Team:
    """
    Holds 11 players. Upgraded to support 4-3-3 and 4-4-2 dynamic formations.
    """
    def __init__(self, team_id, color, formation_type, attack_direction):
        self.team_id = team_id
        self.color = color
        self.formation_type = formation_type
        self.attack_direction = attack_direction
        self.players = []

        if attack_direction == 1:
            self.target_goal = settings.GOAL_RIGHT_CENTER
        else:
            self.target_goal = settings.GOAL_LEFT_CENTER

        # Load roles and base positions
        self.roles = ROLES_433 if formation_type == "4-3-3" else ROLES_442
        self.formation = get_base_positions(formation_type, attack_direction)
        
        self._create_players()

    def _create_players(self):
        for i, (x, y) in enumerate(self.formation):
            role_str = self.roles[i]
            player = Player(x, y, self.team_id, self.color, role_index=i, role_str=role_str)
            self.players.append(player)

    def get_closest_to_ball(self, ball, exclude_gk=False):
        best = None
        best_dist = float('inf')
        for player in self.players:
            if exclude_gk and player.role_str == "GK":
                continue
            dist = player.position.distance_to(ball.position)
            if dist < best_dist:
                best_dist = dist
                best = player
        return best

    def reset_positions(self):
        for i, player in enumerate(self.players):
            player.position = pygame.math.Vector2(self.formation[i])
            player.velocity = pygame.math.Vector2(0, 0)
