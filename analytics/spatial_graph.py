import pygame
import numpy as np
from engine import settings

class SpatialGraphAnalytics:
    """
    Spatial Graph & Geometry Engine:
    - Calculates offside line coordinates for VAR review.
    - Evaluates team compactness radii and passing lane blockage.
    - Measures line-breaking pass opportunities.
    """
    @staticmethod
    def calculate_offside_line(defending_team, attack_direction):
        """
        Determines offside line x-coordinate (second-last defending player position).
        """
        outfield_defenders = [p for p in defending_team.players if p.role_str != "GK"]
        if not outfield_defenders:
            return settings.SCREEN_WIDTH // 2

        # Sort by x position depending on attack direction
        if attack_direction == 1: # Team A attacks right -> defender furthest right
            sorted_defs = sorted(outfield_defenders, key=lambda p: p.position.x, reverse=True)
        else: # Team B attacks left -> defender furthest left
            sorted_defs = sorted(outfield_defenders, key=lambda p: p.position.x)

        # Offside line is 2nd deepest defender
        return sorted_defs[0].position.x

    @staticmethod
    def check_offside(passer, receiver, defending_team):
        """Checks if receiver was beyond the offside line at moment of pass."""
        offside_x = SpatialGraphAnalytics.calculate_offside_line(defending_team, passer.team_id)
        if passer.team_id == 0: # Team A attacks right
            return receiver.position.x > offside_x and receiver.position.x > settings.SCREEN_WIDTH // 2
        else: # Team B attacks left
            return receiver.position.x < offside_x and receiver.position.x < settings.SCREEN_WIDTH // 2

    @staticmethod
    def calculate_team_compactness(team):
        """Calculates mean radius of player positions relative to team centroid."""
        positions = [np.array([p.position.x, p.position.y]) for p in team.players if p.role_str != "GK"]
        if not positions:
            return 0.0
        centroid = np.mean(positions, axis=0)
        distances = [np.linalg.norm(pos - centroid) for pos in positions]
        return float(np.mean(distances))
