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
from tactics.manager import DynamicManagerAI, TacticalStrategy
from backend.database import db

class Game:
    """
    Core Football Game Engine:
    Supports Human Player vs Learning RL AI, customizable formations (4-4-2, 4-3-3, 3-5-2, etc.),
    real-time in-game tactical style switching (Tiki-Taka, Gegenpress, Counter-Attack),
    VAR offside line, Expected Goals (xG) tracking, and SQLite persistence.
    """
    def __init__(self):
        pygame.init()

        self.screen = pygame.display.set_mode((settings.SCREEN_WIDTH, settings.SCREEN_HEIGHT))
        pygame.display.set_caption(settings.TITLE)

        self.clock = pygame.time.Clock()
        self.running = True
        self.state = "MENU"  # MENU, GAMEPLAY, TRAIN_MODE
        self.is_human_mode = True
        self.show_tactics = False
        self.show_tacticai = False

        self.menu = MainMenu(self.screen)
        self.neural_brain = create_neural_brain()

        # Load weights if available
        if hasattr(self.neural_brain, "load_weights"):
            ckpt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rl_env", "checkpoints", "ppo_gnn_model.pt")
            self.neural_brain.load_weights(ckpt_path)

        self.trainer = PPOTrainer()
        self.dashboard = TrainingDashboard(self.screen)
        self.debug_overlay = DebugOverlay()

        self.controlled_player = None
        self.manual_switch_cooldown = 0.0
        self.match_processed = False

        self._init_gameplay("4-3-3", "4-4-2", human_mode=True)

    def _init_gameplay(self, human_formation="4-3-3", ai_formation="4-4-2", human_mode=True):
        match_stats.reset()
        from ai.tendency_profiler import tendency_profiler
        tendency_profiler.reset()
        self.is_human_mode = human_mode
        self.match_processed = False

        # 1. Create Teams with chosen formations
        self.team_a = Team(0, settings.TEAM_A_COLOR, human_formation, 1)
        self.team_b = Team(1, settings.TEAM_B_COLOR, ai_formation, -1)

        # 2. Setup Human Controller on Team A (Blue)
        if self.is_human_mode:
            default_p = self.team_a.players[min(9, len(self.team_a.players) - 1)]
            self.team_a.set_controlled_player(default_p)
            self.controlled_player = default_p
        else:
            for p in self.team_a.players:
                p.is_controlled = False
            self.controlled_player = None

        for p in self.team_b.players:
            p.is_controlled = False

        # 3. Create Ball and Systems
        self.ball = Ball(settings.SCREEN_WIDTH // 2, settings.SCREEN_HEIGHT // 2)
        self.all_players = self.team_a.players + self.team_b.players
        self.entities = self.all_players + [self.ball]

        self.renderer = Renderer(self.screen)
        self.collision_system = CollisionSystem(self.all_players, self.ball)
        self.match = Match(self.team_a, self.team_b, self.ball, duration=settings.MATCH_DURATION)

        # 4. AI Controllers
        self.ai_team_b = AIController(self.team_b, self.team_a, self.ball, neural_brain=self.neural_brain, match=self.match)
        self.ai_team_a = AIController(self.team_a, self.team_b, self.ball, neural_brain=None, match=self.match)

        self.manager_a = DynamicManagerAI(self.team_a, strategy=TacticalStrategy.BALANCED)
        self.manager_b = DynamicManagerAI(self.team_b, strategy=TacticalStrategy.BALANCED)

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
                if action == "start_human":
                    self._init_gameplay(self.menu.human_formation, self.menu.ai_formation, human_mode=True)
                    self.state = "GAMEPLAY"
                elif action == "start_ai_vs_ai":
                    self._init_gameplay(self.menu.human_formation, self.menu.ai_formation, human_mode=False)
                    self.state = "GAMEPLAY"
                elif action == "train_mode":
                    self._init_gameplay(self.menu.human_formation, self.menu.ai_formation, human_mode=False)
                    self.state = "TRAIN_MODE"
                elif action == "quit":
                    self.running = False

            elif self.state in ["GAMEPLAY", "TRAIN_MODE"]:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.state = "MENU"
                    elif event.key == pygame.K_F3:
                        self.debug_overlay.toggle()
                    # Toggle tactical overlay (VAR offside line, passing network)
                    elif event.key == pygame.K_t:
                        self.show_tactics = not self.show_tactics
                    # Toggle Google DeepMind TacticAI predictive overlay
                    elif event.key == pygame.K_y:
                        self.show_tacticai = not self.show_tacticai
                    # In-game Tactical Mentality Hotkeys
                    elif event.key == pygame.K_1:
                        self.manager_a.set_strategy(TacticalStrategy.BALANCED)
                    elif event.key == pygame.K_2:
                        self.manager_a.set_strategy(TacticalStrategy.TIKI_TAKA)
                    elif event.key == pygame.K_3:
                        self.manager_a.set_strategy(TacticalStrategy.GEGENPRESS)
                    elif event.key == pygame.K_4:
                        self.manager_a.set_strategy(TacticalStrategy.COUNTER_ATTACK)
                    elif event.key == pygame.K_5:
                        self.manager_a.set_strategy(TacticalStrategy.PARK_THE_BUS)
                    # Manual player switch: SPACE or TAB
                    elif event.key in [pygame.K_SPACE, pygame.K_TAB] and self.is_human_mode:
                        self._cycle_human_player()

    def _cycle_human_player(self):
        """Switches human control to the next outfield teammate."""
        outfield = [p for p in self.team_a.players if p.role_str != "GK"]
        if not outfield:
            return
        current_idx = outfield.index(self.controlled_player) if self.controlled_player in outfield else 0
        next_p = outfield[(current_idx + 1) % len(outfield)]
        self.team_a.set_controlled_player(next_p)
        self.controlled_player = next_p
        self.manual_switch_cooldown = 0.8

    def _auto_switch_human_player(self):
        """Automatically passes control to the teammate possessing or closest to the ball."""
        if self.manual_switch_cooldown > 0:
            return

        closest_teammate = self.team_a.get_closest_to_ball(self.ball, exclude_gk=True)
        if closest_teammate and closest_teammate.can_kick(self.ball):
            if self.controlled_player is not closest_teammate:
                self.team_a.set_controlled_player(closest_teammate)
                self.controlled_player = closest_teammate
            return

        dist_to_ball = self.controlled_player.position.distance_to(self.ball.position) if self.controlled_player else 999
        closest_dist = closest_teammate.position.distance_to(self.ball.position) if closest_teammate else 999
        if closest_dist < dist_to_ball - 80 and closest_dist < 260:
            self.team_a.set_controlled_player(closest_teammate)
            self.controlled_player = closest_teammate

    def _update(self, dt):
        if self.state == "MENU":
            return

        self.manual_switch_cooldown = max(0.0, self.manual_switch_cooldown - dt)

        # Training mode background rollout
        if self.state == "TRAIN_MODE":
            step_info = self.trainer.train_step(num_episodes=1)
            self.dashboard.update_metrics(step_info["reward"], 0.85)

        # Match pause handling (goal, offside, or full-time)
        if not self.match.is_playing:
            self.match.update(dt)
            if self.match.state == "GOAL_SCORED":
                self.match.trigger_net_ripple(self.renderer)
            elif self.match.state == "FULL_TIME" and not self.match_processed:
                self._handle_full_time()
            return

        # 1. Human player input with xG and offside detection
        if self.is_human_mode and self.controlled_player:
            self._auto_switch_human_player()
            self.controlled_player.handle_human_input(
                self.ball,
                self.team_a.players,
                self.team_a.target_goal,
                self.team_b.players,
                match=self.match,
                dt=dt
            )

        # 2. Dynamic tactical managers
        self.manager_a.update_tactics(self.match)
        self.manager_b.update_tactics(self.match)

        # 3. AI Controllers update
        self.ai_team_b.update(dt)
        self.ai_team_a.update(dt)

        # 4. Physical entity movement
        for entity in self.entities:
            entity.update(dt)

        # 5. Resolve collisions & dribble control
        self.collision_system.update()

        # 6. Check match rules & clock
        self.match.update(dt)

        # 7. Update possession stats
        closest = None
        closest_dist = float('inf')
        for p in self.all_players:
            dist = p.position.distance_to(self.ball.position)
            if dist < closest_dist:
                closest_dist = dist
                closest = p
        if closest_dist < 42:
            match_stats.update_possession(closest)

    def _handle_full_time(self):
        """Processes match completion, triggers PPO training, and records to SQLite DB."""
        self.match_processed = True
        
        # 1. Collect real gameplay experience from AI Team B
        buf = self.ai_team_b.get_and_clear_buffer()
        train_result = {"reward": 0.0, "loss": 0.0}
        if buf and len(buf) > 10:
            train_result = self.trainer.train_on_match_buffer(buf)

        # 2. Extract match statistics
        poss_pct = match_stats.get_possession_pct()
        result_str = self.match.get_result_string()
        tac_name = self.manager_a.get_profile()["name"]
        from ai.tendency_profiler import tendency_profiler
        adaptation = tendency_profiler.get_counter_strategy()
        tendency_summary = tendency_profiler.get_profile_summary()

        # 3. Record in SQLite Database with xG, shots map, and tactical adaptation
        try:
            match_id = db.record_match(
                match_type="HUMAN_VS_AI" if self.is_human_mode else "AI_VS_AI",
                human_score=self.match.score[0],
                ai_score=self.match.score[1],
                human_formation=self.team_a.formation_type,
                ai_formation=self.team_b.formation_type,
                possession_human=poss_pct.get(0, 50.0),
                possession_ai=poss_pct.get(1, 50.0),
                shots_human=match_stats.shots.get(0, 0),
                shots_ai=match_stats.shots.get(1, 0),
                passes_human=match_stats.passes_attempted.get(0, 0),
                passes_ai=match_stats.passes_attempted.get(1, 0),
                result=result_str,
                ai_reward=train_result.get("reward", 0.0),
                ai_loss=train_result.get("loss", 0.0),
                duration_seconds=self.match.time_elapsed,
                xg_human=self.match.cumulative_xg[0],
                xg_ai=self.match.cumulative_xg[1],
                tactical_style=tac_name,
                shots_data=self.match.shot_events,
                ai_adaptation={
                    "counter_strategy": adaptation,
                    "tendency_summary": tendency_summary
                }
            )
            print(f"[DATABASE] Match #{match_id} recorded. AI trained with {len(buf)} transitions. xG: {self.match.cumulative_xg[0]:.2f} - {self.match.cumulative_xg[1]:.2f}")
            print(f"[AI ADAPTATION] {adaptation.get('strategy_name', 'Counter')}: {adaptation.get('tactical_debrief', '')}")
        except Exception as e:
            print(f"[DATABASE ERROR] Could not save match: {e}")

        # 4. Generate match report
        report = generate_match_report(self.match, match_stats)
        print(report)

    def _render(self, dt):
        if self.state == "MENU":
            self.menu.render()
        else:
            self.renderer.render(
                self.entities,
                controlled_player=self.controlled_player if self.is_human_mode else None,
                match=self.match,
                dt=dt,
                show_tactics=self.show_tactics,
                tactical_name=self.manager_a.get_profile()["name"],
                show_tacticai=self.show_tacticai
            )
            self.debug_overlay.render(self.screen, self.all_players)

            if self.state == "TRAIN_MODE":
                self.dashboard.render()

        pygame.display.flip()

    def quit(self):
        self.trainer.close()
        pygame.quit()
        sys.exit()
