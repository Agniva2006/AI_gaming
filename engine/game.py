import pygame
import sys

from engine import settings
from engine.match import Match
from entities.team import Team
from entities.ball import Ball
from rendering.renderer import Renderer
from physics.collision import CollisionSystem
from ai.ai_controller import AIController
from stats.tracker import match_stats
from debug.overlay import DebugOverlay
from analytics.report import generate_match_report


class Game:
    def __init__(self):
        pygame.init()

        self.screen = pygame.display.set_mode((settings.SCREEN_WIDTH, settings.SCREEN_HEIGHT))
        pygame.display.set_caption(settings.TITLE)

        self.clock = pygame.time.Clock()
        self.running = True

        # --- Create teams ---
        self.team_a = Team(
            team_id=0,
            color=settings.TEAM_A_COLOR,
            formation_type="4-3-3",
            attack_direction=1   # attacks right
        )
        self.team_b = Team(
            team_id=1,
            color=settings.TEAM_B_COLOR,
            formation_type="4-4-2",
            attack_direction=-1  # attacks left
        )

        # Human controls the nearest outfield player on Team A
        self.controlled_player = self.team_a.players[9]  # start with a striker
        self.controlled_player.is_controlled = True

        # --- Create ball ---
        self.ball = Ball(settings.SCREEN_WIDTH // 2, settings.SCREEN_HEIGHT // 2)

        # --- Build entities list (all 22 players + ball) ---
        self.all_players = self.team_a.players + self.team_b.players
        self.entities = self.all_players + [self.ball]

        # --- Systems ---
        self.renderer = Renderer(self.screen)
        self.collision_system = CollisionSystem(self.all_players, self.ball)
        self.match = Match(self.team_a, self.team_b, self.ball)

        # AI controls Team B AND the non-controlled players on Team A
        self.ai_team_b = AIController(self.team_b, self.team_a, self.ball)
        self.ai_team_a = AIController(self.team_a, self.team_b, self.ball)
        
        self.debug_overlay = DebugOverlay()
        self.report_generated = False

    def run(self):
        while self.running:
            dt = self._tick()
            self._handle_events()
            self._update(dt)
            self._render()
        self.quit()

    def _tick(self):
        return self.clock.tick(settings.FPS) / 1000.0

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            if event.type == pygame.KEYDOWN and self.match.is_playing:
                # Space = pass
                if event.key == pygame.K_SPACE:
                    self.controlled_player.pass_ball(self.ball)

                # J = shoot toward Team A's target goal
                if event.key == pygame.K_j:
                    self.controlled_player.shoot(self.ball, self.team_a.target_goal)

                # K = switch to nearest teammate to ball
                if event.key == pygame.K_k:
                    self._switch_player()
                    
            if event.type == pygame.KEYDOWN:
                # F3 = toggle debug overlay
                if event.key == pygame.K_F3:
                    self.debug_overlay.toggle()

    def _switch_player(self):
        """Switch control to the Team A player closest to the ball (excluding GK)."""
        self.controlled_player.is_controlled = False
        new_player = self.team_a.get_closest_to_ball(self.ball, exclude_gk=True)
        if new_player:
            new_player.is_controlled = True
            self.controlled_player = new_player

    def _update(self, dt):
        if not self.match.is_playing:
            self.match.update(dt)
            if self.match.state == "FULL_TIME" and not self.report_generated:
                report = generate_match_report(self.match, match_stats)
                print(report)
                self.report_generated = True
            return

        # 1. AI sets velocities for computer-controlled players
        self.ai_team_b.update(dt)
        self.ai_team_a.update(dt)

        # 2. All entities update (movement)
        for entity in self.entities:
            entity.update(dt)

        # 3. Resolve collisions
        self.collision_system.update()

        # 4. Check match rules (goals, etc.)
        self.match.update(dt)
        
        # 5. Track possession stats
        closest = None
        closest_dist = float('inf')
        for p in self.all_players:
            dist = p.position.distance_to(self.ball.position)
            if dist < closest_dist:
                closest_dist = dist
                closest = p
        if closest_dist < 40:
            match_stats.update_possession(closest)

    def _render(self):
        self.renderer.render(self.entities, self.controlled_player, self.match)
        self.debug_overlay.render(self.screen, self.all_players)
        pygame.display.flip()

    def quit(self):
        pygame.quit()
        sys.exit()
