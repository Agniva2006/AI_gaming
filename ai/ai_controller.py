import pygame
import random
import numpy as np
from engine import settings
from ai.goalkeeper import GoalkeeperAI
from tactics.formations import get_tactical_target

class AIController:
    """
    Tactical AI Controller:
    Coordinates 11 team players, integrates Goalkeeper AI,
    runs PPO neural policy inference for the active chaser/playmaker,
    and gathers online experience trajectories for post-match learning.
    """
    def __init__(self, team, opponent_team, ball, neural_brain=None, match=None):
        self.team = team
        self.opponent = opponent_team
        self.ball = ball
        self.neural_brain = neural_brain
        self.match = match
        self.pass_cooldown = 0.0
        self.shoot_cooldown = 0.0

        # Dedicated Goalkeeper AI
        gk_idx = 0
        if "GK" in self.team.roles:
            gk_idx = self.team.roles.index("GK")
        self.gk_ai = GoalkeeperAI(team.players[gk_idx], ball, team)

        # RL Inference & Experience Buffer
        self.experience_buffer = []  # List of (obs, action, reward, next_obs, done)
        self.decision_timer = 0.0
        self.decision_interval = 0.08  # ~12.5 Hz neural decision rate
        self.last_obs = None
        self.last_action = None
        self.prev_ball_dist_to_goal = None

    def update(self, dt):
        self.pass_cooldown = max(0.0, self.pass_cooldown - dt)
        self.shoot_cooldown = max(0.0, self.shoot_cooldown - dt)
        self.decision_timer = max(0.0, self.decision_timer - dt)

        # 1. Update Goalkeeper if not human-controlled
        gk_idx = 0
        if "GK" in self.team.roles:
            gk_idx = self.team.roles.index("GK")
        if not self.team.players[gk_idx].is_controlled:
            self.gk_ai.update(dt)

        # 2. Find chaser (closest outfield player to ball)
        chaser = self.team.get_closest_to_ball(self.ball, exclude_gk=True)

        for player in self.team.players:
            if player.is_controlled or player.role_str == "GK":
                continue

            if player is chaser:
                if self.neural_brain and self.decision_timer <= 0:
                    self._execute_neural_action(player, dt)
                    self.decision_timer = self.decision_interval
                else:
                    self._chaser_logic(player, dt)
            else:
                self._support_logic(player, dt)

    def _build_observation(self, active_player):
        """Constructs the 95-dimensional normalized state vector."""
        obs = np.zeros(95, dtype=np.float32)
        idx = 0
        w = float(settings.SCREEN_WIDTH)
        h = float(settings.SCREEN_HEIGHT)
        max_speed = 600.0

        all_players = self.team.players + self.opponent.players
        for p in all_players:
            obs[idx] = p.position.x / w
            obs[idx+1] = p.position.y / h
            obs[idx+2] = p.velocity.x / max_speed
            obs[idx+3] = p.velocity.y / max_speed
            idx += 4

        # Ball features
        obs[idx] = self.ball.position.x / w
        obs[idx+1] = self.ball.position.y / h
        obs[idx+2] = self.ball.velocity.x / max_speed
        obs[idx+3] = self.ball.velocity.y / max_speed
        idx += 4

        # Score features
        if self.match:
            obs[idx] = min(1.0, self.match.score.get(self.team.team_id, 0) / 10.0)
            obs[idx+1] = min(1.0, self.match.score.get(1 - self.team.team_id, 0) / 10.0)
        else:
            obs[idx] = 0.0
            obs[idx+1] = 0.0
        idx += 2

        # Active player index
        try:
            obs[idx] = all_players.index(active_player) / 22.0
        except Exception:
            obs[idx] = 0.0

        return obs

    def _execute_neural_action(self, player, dt):
        """Queries neural brain policy, executes action, and records RL transition."""
        obs = self._build_observation(player)
        action = self.neural_brain.predict_action(obs, deterministic=False)

        # Calculate reward for previous step if one exists
        if self.last_obs is not None and self.last_action is not None:
            reward = self._calculate_step_reward()
            self.experience_buffer.append((self.last_obs, self.last_action, reward, obs, False))

        self.last_obs = obs
        self.last_action = action

        # Action Execution
        # 0: idle, 1: up, 2: down, 3: left, 4: right, 5: up-left, 6: up-right, 7: down-left, 8: down-right, 9: pass, 10: shoot, 11: switch
        action_map = {
            1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0),
            5: (-1, -1), 6: (1, -1), 7: (-1, 1), 8: (1, 1)
        }

        if action in action_map:
            dx, dy = action_map[action]
            direction = pygame.math.Vector2(dx, dy)
            if direction.length_squared() > 0:
                direction = direction.normalize()
                player.facing = direction.copy()
            player.velocity = direction * settings.AI_CHASE_SPEED
        elif action == 9 and player.can_kick(self.ball) and self.pass_cooldown <= 0:
            target = self._find_pass_target(player)
            if target:
                player.pass_ball(self.ball, target.position)
                self.pass_cooldown = settings.AI_PASS_COOLDOWN
        elif action == 10 and player.can_kick(self.ball) and self.shoot_cooldown <= 0:
            goal_pos = pygame.math.Vector2(self.team.target_goal)
            dist_to_goal = player.position.distance_to(goal_pos)
            if dist_to_goal < settings.AI_SHOOT_DISTANCE:
                player.shoot(self.ball, self.team.target_goal)
                self.shoot_cooldown = settings.AI_SHOOT_COOLDOWN
            else:
                target = self._find_pass_target(player)
                if target and self.pass_cooldown <= 0:
                    player.pass_ball(self.ball, target.position)
                    self.pass_cooldown = settings.AI_PASS_COOLDOWN

        # Realistic movement toward ball if out of possession
        if not player.can_kick(self.ball) and action in [0, 9, 10, 11]:
            self._chaser_logic(player, dt)

    def _calculate_step_reward(self):
        """Calculates dense reward for the AI agent's tactical performance."""
        reward = 0.0
        goal_pos = pygame.math.Vector2(self.team.target_goal)
        current_ball_dist = self.ball.position.distance_to(goal_pos)

        # Progress reward for moving ball toward opponent goal
        if self.prev_ball_dist_to_goal is not None:
            progress = self.prev_ball_dist_to_goal - current_ball_dist
            reward += progress * 0.003
        self.prev_ball_dist_to_goal = current_ball_dist

        # Small reward for possessing the ball
        closest = self.team.get_closest_to_ball(self.ball)
        if closest and closest.can_kick(self.ball):
            reward += settings.RL_REWARD_POSSESSION

        return reward

    def _chaser_logic(self, player, dt):
        """Rule-based chaser AI with dynamic pressing intensity from tendency profiler."""
        dist = player.position.distance_to(self.ball.position)
        press_mult = 1.0
        if self.team.team_id == 1:
            from ai.tendency_profiler import tendency_profiler
            press_mult = tendency_profiler.get_counter_strategy().get("press_dist_mult", 1.0)

        # In possession of ball
        if player.can_kick(self.ball):
            goal_pos = pygame.math.Vector2(self.team.target_goal)
            dist_to_goal = player.position.distance_to(goal_pos)

            if dist_to_goal < settings.AI_SHOOT_DISTANCE and self.shoot_cooldown <= 0:
                opp_gk = next((p for p in self.opponent.players if p.role_str == "GK"), None)
                player.shoot(self.ball, self.team.target_goal, power_ratio=0.75, match=self.match, opp_gk=opp_gk)
                self.shoot_cooldown = settings.AI_SHOOT_COOLDOWN
            elif self.pass_cooldown <= 0:
                pass_target = self._find_pass_target(player)
                if pass_target:
                    player.pass_ball(self.ball, pass_target.position)
                    self.pass_cooldown = settings.AI_PASS_COOLDOWN
                else:
                    self._move_toward(player, goal_pos, settings.AI_CHASE_SPEED * 0.95 * press_mult)
            else:
                self._move_toward(player, goal_pos, settings.AI_CHASE_SPEED * 0.95 * press_mult)
        else:
            # Sprints to press ball carrier based on counter-tactical intensity
            self._move_toward(player, self.ball.position, settings.AI_CHASE_SPEED * press_mult)

    def _support_logic(self, player, dt):
        """Positioning logic for outfield players with real-time opponent counter-adjustments."""
        target = get_tactical_target(
            player.role_str,
            player.home_position,
            self.ball.position,
            self.team.attack_direction
        )

        # Apply AI Opponent Counter-Tactics against Human tendencies
        if self.team.team_id == 1:
            from ai.tendency_profiler import tendency_profiler
            counter = tendency_profiler.get_counter_strategy()

            # 1. Flank overload shifting (e.g. human attacks left wing -> AI shifts RB/RCM to smother)
            if player.role_str in ["LB", "RB", "CB", "LCB", "RCB", "CDM", "CM", "LCM", "RCM"]:
                target.y += counter.get("flank_shift_y", 0.0) * 0.70

            # 2. Deep sweeper cover if human relies heavily on vertical through-balls
            line_mult = counter.get("defensive_line_mult", 1.0)
            if line_mult < 1.0 and player.role_str in ["CB", "LCB", "RCB", "LB", "RB"]:
                target.x = target.x * line_mult + (settings.SCREEN_WIDTH - 80) * (1.0 - line_mult)

        target.x = max(player.radius, min(target.x, settings.SCREEN_WIDTH - player.radius))
        target.y = max(player.radius, min(target.y, settings.SCREEN_HEIGHT - player.radius))

        dist = player.position.distance_to(target)
        if dist > 12:
            self._move_toward(player, target, settings.AI_SPEED * 0.85)
        else:
            player.velocity = pygame.math.Vector2(0, 0)

    def _move_toward(self, player, target, speed):
        direction = target - player.position
        if direction.length_squared() > 0:
            direction = direction.normalize()
            player.facing = direction.copy()

        mult = player.profile.get_current_speed_mult()
        actual_speed = speed * mult
        player.velocity = direction * actual_speed

    def _find_pass_target(self, passer):
        """Finds open teammate with high tactical value and low interception risk."""
        goal_pos = pygame.math.Vector2(self.team.target_goal)
        best = None
        best_score = float('-inf')

        for teammate in self.team.players:
            if teammate is passer or teammate.role_str == "GK":
                continue

            dist_to_passer = passer.position.distance_to(teammate.position)
            if dist_to_passer < 60 or dist_to_passer > 620:
                continue

            dist_to_goal = teammate.position.distance_to(goal_pos)
            score = -dist_to_goal * 0.8

            # Penalize if an opponent is tightly marking teammate
            opp_closest = self.opponent.get_closest_to_ball(teammate)
            if opp_closest and opp_closest.position.distance_to(teammate.position) < 130:
                score -= 250

            # Forward passing bonus
            forward_progress = (teammate.position.x - passer.position.x) * self.team.attack_direction
            if forward_progress > 50:
                score += forward_progress * 1.2

            if score > best_score:
                best_score = score
                best = teammate

        return best

    def get_and_clear_buffer(self):
        """Returns collected match experience transitions and clears buffer."""
        buf = list(self.experience_buffer)
        self.experience_buffer = []
        return buf
