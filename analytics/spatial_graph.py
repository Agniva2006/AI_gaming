import pygame
import numpy as np
import math
from engine import settings

class SpatialGraphAnalytics:
    """
    Spatial Graph & Tactical Geometry Engine:
    - Calculates dynamic VAR offside trap line coordinates.
    - Evaluates offside infractions on forward passes.
    - Measures real-time team compactness and passing triangles.
    """
    @staticmethod
    def calculate_offside_line(defending_team, attack_direction):
        """
        Determines the offside line x-coordinate (position of the second-last defender,
        which corresponds to the deepest outfield defender excluding the goalkeeper).
        """
        outfield_defenders = [p for p in defending_team.players if p.role_str != "GK"]
        if not outfield_defenders:
            return settings.SCREEN_WIDTH // 2

        if attack_direction == 1:
            # Team attacking RIGHT -> deepest defender has largest X
            sorted_defs = sorted(outfield_defenders, key=lambda p: p.position.x, reverse=True)
            return sorted_defs[0].position.x
        else:
            # Team attacking LEFT -> deepest defender has smallest X
            sorted_defs = sorted(outfield_defenders, key=lambda p: p.position.x)
            return sorted_defs[0].position.x

    @staticmethod
    def is_player_offside(player, defending_team, attack_direction):
        """Checks if an attacking player is currently in an offside position."""
        offside_x = SpatialGraphAnalytics.calculate_offside_line(defending_team, attack_direction)
        halfway_x = settings.SCREEN_WIDTH // 2

        if attack_direction == 1:
            return player.position.x > offside_x and player.position.x > halfway_x
        else:
            return player.position.x < offside_x and player.position.x < halfway_x

    @staticmethod
    def calculate_team_compactness(team):
        """Calculates team compactness in meters (average distance of outfield players to centroid)."""
        positions = [np.array([p.position.x, p.position.y]) for p in team.players if p.role_str != "GK"]
        if not positions:
            return 0.0
        centroid = np.mean(positions, axis=0)
        distances = [np.linalg.norm(pos - centroid) for pos in positions]
        # Convert px to meters (1m = 12.2px)
        return round(float(np.mean(distances) / 12.2), 1)

    @staticmethod
    def get_passing_triangles(team, max_dist=320.0):
        """Extracts tactical passing network connections between open teammates."""
        triangles = []
        outfield = [p for p in team.players if p.role_str != "GK"]
        num = len(outfield)
        for i in range(num):
            for j in range(i + 1, num):
                p1 = outfield[i]
                p2 = outfield[j]
                dist = p1.position.distance_to(p2.position)
                if dist < max_dist:
                    triangles.append((p1.position, p2.position, dist))
        return triangles
