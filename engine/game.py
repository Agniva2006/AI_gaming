import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
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
from ui.menu import MainMenu
from ui.dashboard import TrainingDashboard
from rl_env.nn_brain import create_neural_brain
from rl_env.trainer import PPOTrainer
from tactics.manager import DynamicManagerAI

class Game:
    """
    Main Game Engine for RL Train Football:
    Handles 100% autonomous multi-agent GNN-PPO matches, live training suites,
    telemetry dashboards, and dynamic camera rendering.
    """
    def __init__(self):
        pygame.init()

        self.screen = pygame.display.set_mode((settings.SCREEN_WIDTH, settings.SCREEN_HEIGHT))
        pygame.display.set_caption(settings.TITLE)

        self.clock = pygame.time.Clock()
        self.running = True
        self.state = "MENU"  # MENU, GAMEPLAY, TRAIN_MODE

        self.menu = MainMenu(self.screen)
        self.neural_brain = create_neural_brain()
        self.trainer = PPOTrainer()
        self.dashboard = TrainingDashboard(self.screen)

        self._init_gameplay()

    def _init_gameplay(self):
        match_stats.reset()
        self.team_a = Team(0, settings.TEAM_A_COLOR, "4-3-3", 1)
        self.team_b = Team(1, settings.TEAM_B_COLOR, "4-4-2", -1)


        # 100% Autonomous — No player is manually keyboard-controlled
        for p in self.team_a.players:
            p.is_controlled = False
        for p in self.team_b.players:
            p.is_controlled = False

        self.ball = Ball(settings.SCREEN_WIDTH // 2, settings.SCREEN_HEIGHT // 2)

        self.all_players = self.team_a.players + self.team_b.players
        self.entities = self.all_players + [self.ball]

        self.renderer = Renderer(self.screen)
        self.collision_system = CollisionSystem(self.all_players, self.ball)
        self.match = Match(self.team_a, self.team_b, self.ball)

        # Autonomous GNN-PPO AI controllers on both teams
        self.ai_team_b = AIController(self.team_b, self.team_a, self.ball, neural_brain=self.neural_brain)
        self.ai_team_a = AIController(self.team_a, self.team_b, self.ball, neural_brain=self.neural_brain)

        self.manager_a = DynamicManagerAI(self.team_a)
        self.manager_b = DynamicManagerAI(self.team_b)

        self.debug_overlay = DebugOverlay()
        self.report_generated = False

    def run(self):
        while self.running:
            dt = self._tick()
            self._handle_events()
            self._update(dt)
            self._render(dt)
        self.quit()

    def _tick(self):
        return self.clock.tick(settings.FPS) / 1000.0

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            if self.state == "MENU":
                action = self.menu.handle_event(event)
                if action in ["neural_match", "voronoi"]:
                    self._init_gameplay()
                    self.state = "GAMEPLAY"
                elif action == "train_mode":
                    self._init_gameplay()
                    self.state = "TRAIN_MODE"
                elif action == "quit":
                    self.running = False
            elif self.state in ["GAMEPLAY", "TRAIN_MODE"]:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.state = "MENU"
                    elif event.key == pygame.K_F3:
                        self.debug_overlay.toggle()

    def _update(self, dt):
        if self.state == "MENU":
            return

        if self.state == "TRAIN_MODE":
            # Run background PPO step & update dashboard telemetry
            step_info = self.trainer.train_step(num_episodes=1)
            self.dashboard.update_metrics(step_info["reward"], step_info["realism"])

        if not self.match.is_playing:
            self.match.update(dt)
            if self.match.state == "GOAL_SCORED":
                self.match.trigger_net_ripple(self.renderer)
            elif self.match.state == "FULL_TIME" and not self.report_generated:
                report = generate_match_report(self.match, match_stats)
                print(report)
                self.report_generated = True
            return

        # 1. Update Dynamic Managers
        self.manager_a.update_tactics(self.match)
        self.manager_b.update_tactics(self.match)

        # 2. Autonomous GNN-PPO AI controllers update
        self.ai_team_b.update(dt)
        self.ai_team_a.update(dt)

        # 3. All entities update (movement + inertia)
        for entity in self.entities:
            entity.update(dt)

        # 4. Resolve collisions
        self.collision_system.update()

        # 5. Check match rules
        self.match.update(dt)

        # 6. Possession tracking
        closest = None
        closest_dist = float('inf')
        for p in self.all_players:
            dist = p.position.distance_to(self.ball.position)
            if dist < closest_dist:
                closest_dist = dist
                closest = p
        if closest_dist < 40:
            match_stats.update_possession(closest)

    def _render(self, dt):
        if self.state == "MENU":
            self.menu.render()
        else:
            self.renderer.render(self.entities, controlled_player=None, match=self.match, dt=dt)
            self.debug_overlay.render(self.screen, self.all_players)

            if self.state == "TRAIN_MODE":
                self.dashboard.render()

        pygame.display.flip()

    def quit(self):
        self.trainer.close()
        pygame.quit()
        sys.exit()
