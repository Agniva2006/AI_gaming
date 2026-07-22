"""
Self-Play Training Scaffold for Football RL.

This module provides:
  1. A multi-agent environment where two RL agents control opposing teams
  2. A random agent for testing
  3. A training loop scaffold that works with any RL library

Usage (random agents — test the interface):
    python -m rl_env.self_play

Usage (with stable-baselines3 — install separately):
    See the SB3TrainingExample class below.
"""

import os
import sys
import numpy as np
import pygame
import random

from engine import settings
from engine.match import Match
from entities.team import Team
from entities.ball import Ball
from physics.collision import CollisionSystem
from ai.ai_controller import AIController


# ─────────────────────────────────────────────────────
# Multi-Agent Environment (two RL agents, no built-in AI)
# ─────────────────────────────────────────────────────

ACTION_DIRECTIONS = {
    0: (0, 0), 1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0),
    5: (-1, -1), 6: (1, -1), 7: (-1, 1), 8: (1, 1),
}


class SelfPlayEnv:
    """
    Two-agent environment for self-play training.
    Each agent controls one team. Both receive observations and rewards.

    Usage:
        env = SelfPlayEnv()
        obs_a, obs_b = env.reset()
        while not done:
            action_a = agent_a.predict(obs_a)
            action_b = agent_b.predict(obs_b)
            (obs_a, r_a, done, info), (obs_b, r_b, _, _) = env.step(action_a, action_b)
    """
    def __init__(self, render=False):
        self.render_enabled = render
        self.obs_size = 95
        self.num_actions = settings.NUM_ACTIONS
        self.max_steps = settings.RL_MAX_STEPS
        self.current_step = 0

        if not pygame.get_init():
            pygame.init()


        if render:
            self.screen = pygame.display.set_mode(
                (settings.SCREEN_WIDTH, settings.SCREEN_HEIGHT)
            )
            pygame.display.set_caption(settings.TITLE + " [Self-Play]")
            from rendering.renderer import Renderer
            self.renderer = Renderer(self.screen)
        else:
            self.screen = pygame.Surface(
                (settings.SCREEN_WIDTH, settings.SCREEN_HEIGHT)
            )
            self.renderer = None

        self.clock = pygame.time.Clock()
        self.team_a = None
        self.team_b = None
        self.ball = None
        self.match = None
        self.collision_system = None
        self.controlled_a = None
        self.controlled_b = None
        self.all_players = []

    def reset(self):
        """Reset and return (obs_team_a, obs_team_b)."""
        self.current_step = 0

        self.team_a = Team(0, settings.TEAM_A_COLOR, settings.FORMATION_442_A, 1)
        self.team_b = Team(1, settings.TEAM_B_COLOR, settings.FORMATION_442_B, -1)

        self.controlled_a = self.team_a.players[9]
        self.controlled_b = self.team_b.players[9]

        self.ball = Ball(settings.SCREEN_WIDTH // 2, settings.SCREEN_HEIGHT // 2)
        self.all_players = self.team_a.players + self.team_b.players

        self.collision_system = CollisionSystem(self.all_players, self.ball)
        self.match = Match(self.team_a, self.team_b, self.ball)

        # AI controllers for non-controlled players on each team
        self.ai_a = AIController(self.team_a, self.team_b, self.ball)
        self.ai_b = AIController(self.team_b, self.team_a, self.ball)

        self._prev_ball_x = self.ball.position.x

        return self._get_obs(self.team_a), self._get_obs(self.team_b)

    def step(self, action_a, action_b):
        """
        Both agents act simultaneously.
        Returns: ((obs_a, reward_a, done, info), (obs_b, reward_b, done, info))
        """
        self.current_step += 1
        dt = 1.0 / settings.FPS

        # Apply actions
        self._apply_action(action_a, self.controlled_a, self.team_a)
        self._apply_action(action_b, self.controlled_b, self.team_b)

        # AI for non-controlled players
        if self.match.is_playing:
            self.ai_a.update(dt)
            self.ai_b.update(dt)

        # Update entities
        entities = self.all_players + [self.ball]
        for entity in entities:
            entity.update(dt)

        # Collisions
        self.collision_system.update()

        # Match
        prev_a, prev_b = self.match.score[0], self.match.score[1]
        self.match.update(dt)

        # Rewards (symmetric — one team's goal is the other's concession)
        reward_a = 0.0
        reward_b = 0.0
        if self.match.score[0] > prev_a:
            reward_a += settings.RL_REWARD_GOAL
            reward_b += settings.RL_REWARD_CONCEDE
        if self.match.score[1] > prev_b:
            reward_b += settings.RL_REWARD_GOAL
            reward_a += settings.RL_REWARD_CONCEDE

        done = self.current_step >= self.max_steps

        # Render
        if self.render_enabled and self.renderer:
            self.renderer.render(entities, self.controlled_a, self.match)
            self.clock.tick(settings.FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

        obs_a = self._get_obs(self.team_a)
        obs_b = self._get_obs(self.team_b)

        info = {"score_a": self.match.score[0], "score_b": self.match.score[1]}

        self._prev_ball_x = self.ball.position.x

        return (obs_a, reward_a, done, info), (obs_b, reward_b, done, info)

    def _apply_action(self, action, controlled, team):
        """Apply a discrete action to the controlled player."""
        player = controlled
        player.is_controlled = False  # disable keyboard

        if action in ACTION_DIRECTIONS:
            dx, dy = ACTION_DIRECTIONS[action]
            direction = pygame.math.Vector2(dx, dy)
            if direction.length_squared() > 0:
                direction = direction.normalize()
                player.facing = direction.copy()
            player.velocity = direction * player.speed
        elif action == 9:
            player.pass_ball(self.ball)
        elif action == 10:
            player.shoot(self.ball, team.target_goal)
        elif action == 11:
            new = team.get_closest_to_ball(self.ball, exclude_gk=True)
            if new and new is not player:
                if team.team_id == 0:
                    self.controlled_a = new
                else:
                    self.controlled_b = new

    def _get_obs(self, team):
        """Build observation from a team's perspective."""
        obs = np.zeros(self.obs_size, dtype=np.float32)
        w, h = settings.SCREEN_WIDTH, settings.SCREEN_HEIGHT
        max_speed = 800.0

        idx = 0
        # All players (same order regardless of team for consistency)
        for player in self.all_players:
            obs[idx] = player.position.x / w
            obs[idx + 1] = player.position.y / h
            obs[idx + 2] = player.velocity.x / max_speed
            obs[idx + 3] = player.velocity.y / max_speed
            idx += 4

        # Ball
        obs[idx] = self.ball.position.x / w
        obs[idx + 1] = self.ball.position.y / h
        obs[idx + 2] = self.ball.velocity.x / max_speed
        obs[idx + 3] = self.ball.velocity.y / max_speed
        idx += 4

        # Score
        obs[idx] = self.match.score[0] / 10.0
        obs[idx + 1] = self.match.score[1] / 10.0
        idx += 2

        # Controlled player index
        ctrl = self.controlled_a if team.team_id == 0 else self.controlled_b
        try:
            obs[idx] = self.all_players.index(ctrl) / 22.0
        except ValueError:
            obs[idx] = 0.0

        return obs

    def close(self):
        pygame.quit()


# ─────────────────────────────────────────────────────
# Random Agent (for testing)
# ─────────────────────────────────────────────────────

class RandomAgent:
    """Takes random actions. Used to verify the environment works."""
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def predict(self, obs):
        return random.randint(0, self.num_actions - 1)


# ─────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────

def run_self_play_test(num_episodes=3, render=True):
    """
    Run a self-play test with random agents to verify the environment.
    """
    env = SelfPlayEnv(render=render)
    agent_a = RandomAgent(env.num_actions)
    agent_b = RandomAgent(env.num_actions)

    for episode in range(num_episodes):
        obs_a, obs_b = env.reset()
        total_reward_a = 0.0
        total_reward_b = 0.0
        done = False

        while not done:
            action_a = agent_a.predict(obs_a)
            action_b = agent_b.predict(obs_b)

            (obs_a, r_a, done, info), (obs_b, r_b, _, _) = env.step(action_a, action_b)
            total_reward_a += r_a
            total_reward_b += r_b

        print(f"Episode {episode + 1}/{num_episodes} | "
              f"Score: {info['score_a']}-{info['score_b']} | "
              f"Rewards: A={total_reward_a:.3f}, B={total_reward_b:.3f}")

    env.close()


# ─────────────────────────────────────────────────────
# SB3 Integration Example (uncomment when ready)
# ─────────────────────────────────────────────────────

"""
To train with stable-baselines3:

    pip install stable-baselines3

Then use the single-agent FootballEnv with a Gymnasium wrapper:

    import gymnasium as gym
    from gymnasium import spaces
    from rl_env.football_env import FootballEnv
    from stable_baselines3 import PPO

    class FootballGymWrapper(gym.Env):
        def __init__(self):
            super().__init__()
            self.env = FootballEnv(render=False)
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0,
                shape=(self.env.obs_size,), dtype=np.float32
            )
            self.action_space = spaces.Discrete(self.env.num_actions)

        def reset(self, seed=None, options=None):
            obs = self.env.reset()
            return obs, {}

        def step(self, action):
            obs, reward, done, info = self.env.step(action)
            return obs, reward, done, done, info

        def close(self):
            self.env.close()

    # Train
    env = FootballGymWrapper()
    model = PPO("MlpPolicy", env, verbose=1, n_steps=2048, batch_size=64)
    model.learn(total_timesteps=1_000_000)
    model.save("football_ppo")
"""


if __name__ == "__main__":
    run_self_play_test(num_episodes=3, render=True)
