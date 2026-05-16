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
    Upgraded AI with context evaluation, decision delay, 
    and role-based tactical positioning (Phases B & C).
    """
    def __init__(self, team, opponent_team, ball):
        self.team = team
        self.opponent = opponent_team
        self.ball = ball
        self.pass_cooldown = 0.0

        gk_idx = self.team.roles.index("GK")
        self.gk_ai = GoalkeeperAI(team.players[gk_idx], ball, team)

    def update(self, dt):
        self.pass_cooldown = max(0, self.pass_cooldown - dt)

        gk_idx = self.team.roles.index("GK")
        if not self.team.players[gk_idx].is_controlled:
            self.gk_ai.update(dt)

        chaser = self.team.get_closest_to_ball(self.ball, exclude_gk=True)

        for player in self.team.players:
            if player.is_controlled or player.role_str == "GK":
                continue

            if player is chaser:
                self._chaser_logic(player, dt)
            else:
                self._support_logic(player, dt)

    def _chaser_logic(self, player, dt):
        """The closest player chases the ball. If they have it, evaluate options."""
        if player.can_kick(self.ball):
            
            # Phase C: Decision delay based on composure
            if player.decision_timer > 0:
                self._move_toward(player, player.position, 0) # Wait and protect ball
                return

            goal_pos = pygame.math.Vector2(self.team.target_goal)
            dist_to_goal = player.position.distance_to(goal_pos)

            if dist_to_goal < settings.AI_SHOOT_DISTANCE:
                player.shoot(self.ball, self.team.target_goal)
                player.decision_timer = (100 - player.profile.composure) * 0.01
            elif self.pass_cooldown <= 0:
                target = self._find_pass_target(player)
                if target:
                    player.pass_ball(self.ball, target.position)
                    self.pass_cooldown = settings.AI_PASS_COOLDOWN
                    # Delay next action
                    player.decision_timer = (100 - player.profile.composure) * 0.005
                else:
                    self._move_toward(player, goal_pos, settings.AI_CHASE_SPEED)
            else:
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
