import pygame
import random
from enum import Enum
from engine import settings
from ai.goalkeeper import GoalkeeperAI
from tactics.formations import get_tactical_target


class AIState(Enum):
    IDLE = "idle"
    CHASE_BALL = "chase_ball"
    SUPPORT = "support"


class AIController:
    """
    Upgraded AI with PyTorch GPU Neural Network inference capabilities
    blended with role-based tactical positioning and FSM goalkeeper logic.
    """
    def __init__(self, team, opponent_team, ball, neural_brain=None):
        self.team = team
        self.opponent = opponent_team
        self.ball = ball
        self.pass_cooldown = 0.0
        self.shoot_cooldown = 0.0
        self.neural_brain = neural_brain  # PyTorch GPU Neural Brain model

        gk_idx = self.team.roles.index("GK")
        self.gk_ai = GoalkeeperAI(team.players[gk_idx], ball, team)

    def update(self, dt):
        self.pass_cooldown = max(0, self.pass_cooldown - dt)
        self.shoot_cooldown = max(0, self.shoot_cooldown - dt)

        gk_idx = self.team.roles.index("GK")
        if not self.team.players[gk_idx].is_controlled:
            self.gk_ai.update(dt)

        chaser = self.team.get_closest_to_ball(self.ball, exclude_gk=True)

        for player in self.team.players:
            if player.is_controlled or player.role_str == "GK":
                continue

            if player is chaser:
                if self.neural_brain:
                    self._execute_neural_action(player, dt)
                else:
                    self._chaser_logic(player, dt)
            else:
                self._support_logic(player, dt)

    def _execute_neural_action(self, player, dt):
        """Executes action predicted by PyTorch CUDA Neural Brain."""
        # 1. Build observation array (95-dim)
        import numpy as np
        obs = np.zeros(95, dtype=np.float32)
        idx = 0
        w = settings.SCREEN_WIDTH
        h = settings.SCREEN_HEIGHT
        max_speed = 800.0
        
        all_players = self.team.players + self.opponent.players
        for p in all_players:
            obs[idx] = p.position.x / w
            obs[idx+1] = p.position.y / h
            obs[idx+2] = p.velocity.x / max_speed
            obs[idx+3] = p.velocity.y / max_speed
            idx += 4
            
        obs[idx] = self.ball.position.x / w
        obs[idx+1] = self.ball.position.y / h
        obs[idx+2] = self.ball.velocity.x / max_speed
        obs[idx+3] = self.ball.velocity.y / max_speed
        idx += 4
        
        # mock score
        obs[idx] = 0.0
        obs[idx+1] = 0.0
        idx += 2
        
        try:
            obs[idx] = all_players.index(player) / 22.0
        except:
            obs[idx] = 0.0
            
        # 2. Query Neural Brain
        action = self.neural_brain.predict_action(obs, deterministic=True)
        
        # 3. Execute Action
        # Mapping: 0: idle, 1: up, 2: down, 3: left, 4: right, 5: up-left, 6: up-right, 7: down-left, 8: down-right, 9: pass, 10: shoot, 11: switch
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
                # Too far to shoot — try a pass instead
                target = self._find_pass_target(player)
                if target and self.pass_cooldown <= 0:
                    player.pass_ball(self.ball, target.position)
                    self.pass_cooldown = settings.AI_PASS_COOLDOWN
            
        # fallback for realism when out of possession or idle
        if not player.can_kick(self.ball) and action in [0, 9, 10, 11]:
             self._chaser_logic(player, dt)


    def _chaser_logic(self, player, dt):
        """The closest player chases the ball. If they have it, evaluate options."""
        if player.can_kick(self.ball):
            if player.decision_timer > 0:
                self._move_toward(player, player.position, 0)
                return

            goal_pos = pygame.math.Vector2(self.team.target_goal)
            dist_to_goal = player.position.distance_to(goal_pos)

            # Priority 1: Shoot only when close to goal AND cooldown allows
            if dist_to_goal < settings.AI_SHOOT_DISTANCE and self.shoot_cooldown <= 0:
                player.shoot(self.ball, self.team.target_goal)
                self.shoot_cooldown = settings.AI_SHOOT_COOLDOWN
                player.decision_timer = (100 - player.profile.composure) * 0.01
            # Priority 2: Always try to pass first when possible
            elif self.pass_cooldown <= 0:
                target = self._find_pass_target(player)
                if target:
                    player.pass_ball(self.ball, target.position)
                    self.pass_cooldown = settings.AI_PASS_COOLDOWN
                    player.decision_timer = (100 - player.profile.composure) * 0.005
                else:
                    # No pass option — dribble toward goal
                    self._move_toward(player, goal_pos, settings.AI_CHASE_SPEED)
            else:
                # Cooldowns active — dribble toward goal
                self._move_toward(player, goal_pos, settings.AI_CHASE_SPEED)
        else:
            self._move_toward(player, self.ball.position, settings.AI_CHASE_SPEED)

    def _support_logic(self, player, dt):
        """Phase B: Support players use role-based tactical targets."""
        target = get_tactical_target(
            player.role_str, 
            player.home_position, 
            self.ball.position, 
            self.team.attack_direction
        )

        target.x = max(player.radius, min(target.x, settings.SCREEN_WIDTH - player.radius))
        target.y = max(player.radius, min(target.y, settings.SCREEN_HEIGHT - player.radius))

        dist = player.position.distance_to(target)
        if dist > 10:
            self._move_toward(player, target, settings.AI_SPEED * 0.8)
        else:
            player.velocity = pygame.math.Vector2(0, 0)

    def _move_toward(self, player, target, speed):
        direction = target - player.position
        if direction.length_squared() > 0:
            direction = direction.normalize()
            player.facing = direction.copy()
            
        # Apply fatigue penalty
        mult = player.profile.get_current_speed_mult()
        actual_speed = speed * mult
        player.velocity = direction * actual_speed

    def _find_pass_target(self, passer):
        """Phase C: Context-aware pass evaluation."""
        goal_pos = pygame.math.Vector2(self.team.target_goal)
        best = None
        best_score = float('-inf')

        for teammate in self.team.players:
            if teammate is passer or teammate.role_str == "GK":
                continue

            dist_to_passer = passer.position.distance_to(teammate.position)
            if dist_to_passer < 80 or dist_to_passer > 600:
                continue

            dist_to_goal = teammate.position.distance_to(goal_pos)
            score = -dist_to_goal

            # Risk evaluation: Is opponent near target?
            opp_closest = self.opponent.get_closest_to_ball(teammate)
            if opp_closest and opp_closest.position.distance_to(teammate.position) < 150:
                score -= 300 # high risk

            # Vision check for through balls
            if passer.profile.vision > 75 and dist_to_goal < 300:
                score += 150

            # Possession recycling (backward pass)
            if dist_to_passer < 250 and random.random() < 0.3:
                score += 80

            if score > best_score:
                best_score = score
                best = teammate

        return best
