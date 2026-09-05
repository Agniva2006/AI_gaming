"""
Football RL Environment — Gymnasium-compatible interface.

This wraps the game simulation into the standard RL interface:
    obs, reward, done, truncated, info = env.step(action)
    obs = env.reset()

The environment can run headless (no display) for fast training,
or with rendering for visualization.

Action Space (Discrete, 12 actions):
    0: idle
    1: move up          2: move down
    3: move left        4: move right
    5: move up-left     6: move up-right
    7: move down-left   8: move down-right
    9: pass            10: shoot
    11: switch player

Observation Space (1D float array):
    For each of 22 players: (x, y, vx, vy) normalized to [0,1]
    Ball: (x, y, vx, vy)
    Score: (team_a, team_b)
    Controlled player index
    = 22*4 + 4 + 2 + 1 = 95 values
"""

import os
import numpy as np
import pygame

from engine import settings
from engine.match import Match
from entities.team import Team
from entities.ball import Ball
from physics.collision import CollisionSystem
from ai.ai_controller import AIController


# Action mapping: index -> (dx, dy) movement direction
ACTION_DIRECTIONS = {
    0: (0, 0),      # idle
    1: (0, -1),     # up
    2: (0, 1),      # down
    3: (-1, 0),     # left
    4: (1, 0),      # right
    5: (-1, -1),    # up-left
    6: (1, -1),     # up-right
    7: (-1, 1),     # down-left
    8: (1, 1),      # down-right
}


class FootballEnv:
    """
    RL environment for the football game.

    Usage:
        env = FootballEnv(render=False)
        obs = env.reset()
        for step in range(max_steps):
            action = agent.predict(obs)
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()
    """
    def __init__(self, render=False):
        self.render_enabled = render
        self.obs_size = 95
        self.num_actions = settings.NUM_ACTIONS
        self.max_steps = settings.RL_MAX_STEPS
        self.current_step = 0

        # Initialize pygame
        if not pygame.get_init():
            pygame.init()


        if render:
            self.screen = pygame.display.set_mode(
                (settings.SCREEN_WIDTH, settings.SCREEN_HEIGHT)
            )
            pygame.display.set_caption(settings.TITLE + " [RL Training]")
            from rendering.renderer import Renderer
            self.renderer = Renderer(self.screen)
        else:
            self.screen = pygame.Surface(
                (settings.SCREEN_WIDTH, settings.SCREEN_HEIGHT)
            )
            self.renderer = None

        self.clock = pygame.time.Clock()

        # These get created in reset()
        self.team_a = None
        self.team_b = None
        self.ball = None
        self.match = None
        self.collision_system = None
        self.ai_opponent = None
        self.ai_team_a = None
        self.controlled_player = None
        self.all_players = []

    def reset(self):
        """Reset the environment and return initial observation."""
        self.current_step = 0

        # Create teams
        self.team_a = Team(0, settings.TEAM_A_COLOR, "4-4-2", 1)
        self.team_b = Team(1, settings.TEAM_B_COLOR, "4-4-2", -1)

        # RL agent controls team A's striker
        self.controlled_player = self.team_a.players[9]
        self.controlled_player.is_controlled = True

        # Create ball
        self.ball = Ball(settings.SCREEN_WIDTH // 2, settings.SCREEN_HEIGHT // 2)

        # All players
        self.all_players = self.team_a.players + self.team_b.players

        # Systems
        self.collision_system = CollisionSystem(self.all_players, self.ball)
        self.match = Match(self.team_a, self.team_b, self.ball)

        # AI controls opponent team and Team A's non-controlled players
        self.ai_opponent = AIController(self.team_b, self.team_a, self.ball)
        self.ai_team_a = AIController(self.team_a, self.team_b, self.ball)

        # Track ball X for progress reward
        self._prev_ball_x = self.ball.position.x

        return self._get_observation()

    def step(self, action):
        """
        Execute one step in the environment.
        Returns: (observation, reward, done, info)
        """
        self.current_step += 1
        dt = 1.0 / settings.FPS  # fixed timestep for determinism

        # 1. Apply RL agent's action to controlled player
        self._apply_action(action)

        # 2. AI updates for opponent and teammates
        if self.match.is_playing:
            self.ai_opponent.update(dt)
            self.ai_team_a.update(dt)

        # 3. Update all entities
        entities = self.all_players + [self.ball]
        for entity in entities:
            entity.update(dt)

        # 4. Collisions
        self.collision_system.update()

        # 5. Match rules
        prev_score_a = self.match.score[0]
        prev_score_b = self.match.score[1]
        self.match.update(dt)

        # 6. Calculate reward
        reward = self._calculate_reward(prev_score_a, prev_score_b)

        # 7. Check done
        done = self.current_step >= self.max_steps
        truncated = done  # episode truncated by step limit

        # 8. Render if enabled
        if self.render_enabled and self.renderer:
            self.renderer.render(entities, self.controlled_player, self.match)
            self.clock.tick(settings.FPS)

            # Handle pygame events to prevent window freeze
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

        # 9. Get observation
        obs = self._get_observation()

        info = {
            "score_a": self.match.score[0],
            "score_b": self.match.score[1],
            "step": self.current_step,
        }

        self._prev_ball_x = self.ball.position.x

        return obs, reward, done, info

    def _apply_action(self, action):
        """Convert discrete action to player commands."""
        player = self.controlled_player

        if action in ACTION_DIRECTIONS:
            # Movement action
            dx, dy = ACTION_DIRECTIONS[action]
            direction = pygame.math.Vector2(dx, dy)
            if direction.length_squared() > 0:
                direction = direction.normalize()
                player.facing = direction.copy()
            player.velocity = direction * player.speed
            # Don't read keyboard — override is_controlled behavior
            player.is_controlled = False  # temporarily disable keyboard
        elif action == 9:
            # Pass
            player.is_controlled = False
            player.pass_ball(self.ball)
        elif action == 10:
            # Shoot
            player.is_controlled = False
            player.shoot(self.ball, self.team_a.target_goal)
        elif action == 11:
            # Switch player
            player.is_controlled = False
            new_player = self.team_a.get_closest_to_ball(self.ball, exclude_gk=True)
            if new_player and new_player is not player:
                self.controlled_player = new_player

    def _calculate_reward(self, prev_score_a, prev_score_b):
        """Calculate reward for the current step."""
        reward = 0.0

        # Goal scored by our team
        if self.match.score[0] > prev_score_a:
            reward += settings.RL_REWARD_GOAL

        # Goal conceded
        if self.match.score[1] > prev_score_b:
            reward += settings.RL_REWARD_CONCEDE

        # Small reward for possession (ball near our players)
        nearest_a = self.team_a.get_closest_to_ball(self.ball)
        nearest_b = self.team_b.get_closest_to_ball(self.ball)
        if nearest_a and nearest_b:
            dist_a = nearest_a.position.distance_to(self.ball.position)
            dist_b = nearest_b.position.distance_to(self.ball.position)
            if dist_a < dist_b:
                reward += settings.RL_REWARD_POSSESSION

        # Reward for ball progress toward opponent goal (right side)
        ball_progress = self.ball.position.x - self._prev_ball_x
        reward += ball_progress * getattr(settings, "RL_REWARD_BALL_PROGRESS", 0.001)

        return reward

    def _get_observation(self):
        """
        Build a flat numpy array of the game state.
        All values normalized to roughly [0, 1].
        """
        obs = np.zeros(self.obs_size, dtype=np.float32)
        w = settings.SCREEN_WIDTH
        h = settings.SCREEN_HEIGHT
        max_speed = 800.0  # normalization constant for velocities

        idx = 0
        # All 22 players: (x, y, vx, vy) normalized
        for player in self.all_players:
            obs[idx] = player.position.x / w
            obs[idx + 1] = player.position.y / h
            obs[idx + 2] = player.velocity.x / max_speed
            obs[idx + 3] = player.velocity.y / max_speed
            idx += 4

        # Ball: (x, y, vx, vy)
        obs[idx] = self.ball.position.x / w
        obs[idx + 1] = self.ball.position.y / h
        obs[idx + 2] = self.ball.velocity.x / max_speed
        obs[idx + 3] = self.ball.velocity.y / max_speed
        idx += 4

        # Score
        obs[idx] = self.match.score[0] / 10.0  # normalize assuming max 10 goals
        obs[idx + 1] = self.match.score[1] / 10.0
        idx += 2

        # Controlled player index
        try:
            obs[idx] = self.all_players.index(self.controlled_player) / 22.0
        except ValueError:
            obs[idx] = 0.0

        return obs

    def close(self):
        """Clean up."""
        if pygame.get_init():
            try:
                pygame.quit()
            except Exception:
                pass
