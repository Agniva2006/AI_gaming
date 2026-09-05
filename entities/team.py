import pygame
from entities.players import Player
from engine import settings
from tactics.formations import get_base_positions, get_formation_roles

class Team:
    """
    Manages 11 players for a football team.
    Supports tactical formation switching, controlled player designation,
    and coordinate resets for kickoffs.
    """
    def __init__(self, team_id, color, formation_type="4-3-3", attack_direction=1, custom_coords=None):
        self.team_id = team_id
        self.color = color
        self.formation_type = formation_type
        self.attack_direction = attack_direction
        self.custom_coords = custom_coords
        self.players = []

        if attack_direction == 1:
            self.target_goal = settings.GOAL_RIGHT_CENTER
            self.defending_goal = settings.GOAL_LEFT_CENTER
        else:
            self.target_goal = settings.GOAL_LEFT_CENTER
            self.defending_goal = settings.GOAL_RIGHT_CENTER

        self.roles = get_formation_roles(formation_type)
        self.formation = get_base_positions(formation_type, attack_direction, custom_coords)
        self._create_players()

    def _create_players(self):
        self.players = []
        for i, (x, y) in enumerate(self.formation):
            role_str = self.roles[i] if i < len(self.roles) else "SUB"
            player = Player(x, y, self.team_id, self.color, role_index=i, role_str=role_str)
            self.players.append(player)

    def set_formation(self, formation_type, custom_coords=None):
        """Updates team formation and repositions players to the new tactical baseline."""
        self.formation_type = formation_type
        self.custom_coords = custom_coords
        self.roles = get_formation_roles(formation_type)
        self.formation = get_base_positions(formation_type, self.attack_direction, custom_coords)

        for i, player in enumerate(self.players):
            if i < len(self.formation):
                player.home_position = pygame.math.Vector2(self.formation[i])
                player.role_str = self.roles[i] if i < len(self.roles) else "SUB"

    def reset_positions(self):
        """Returns all players to kickoff formation coordinates."""
        for i, player in enumerate(self.players):
            if i < len(self.formation):
                player.position = pygame.math.Vector2(self.formation[i])
                player.velocity = pygame.math.Vector2(0, 0)
                player.current_vel = pygame.math.Vector2(0, 0)
                player.facing = pygame.math.Vector2(1 if self.attack_direction == 1 else -1, 0)

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

    def get_controlled_player(self):
        for p in self.players:
            if p.is_controlled:
                return p
        return None

    def set_controlled_player(self, player_to_control):
        """Activates human control for one player and deactivates for all others."""
        for p in self.players:
            p.is_controlled = (p is player_to_control)
