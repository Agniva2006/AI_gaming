import pygame
import random
import math
from entities.entity import Entity
from engine import settings
from attributes.profile import PlayerProfile
from stats.tracker import match_stats
from analytics.xg_engine import XGEngine
from analytics.spatial_graph import SpatialGraphAnalytics
from ai.tendency_profiler import tendency_profiler

class Player(Entity):
    """
    Football Player Entity:
    Features interactive human controls (WASD/Arrows, Sprint, Pass, Through-Ball, Shoot, Chip, Tackle),
    Expected Goals (xG) shot calculation, offside detection, and aerial volleys/headers.
    """
    def __init__(self, x, y, team_id, color, role_index=0, role_str="CM"):
        super().__init__(x, y, settings.PLAYER_RADIUS, color)
        self.team_id = team_id
        self.role_index = role_index
        self.role_str = role_str
        self.is_controlled = False

        self.profile = PlayerProfile(role_str)
        self.speed = settings.PLAYER_SPEED
        self.sprint_speed = settings.SPRINT_SPEED

        self.facing = pygame.math.Vector2(1 if team_id == 0 else -1, 0)
        self.home_position = pygame.math.Vector2(x, y)
        self.decision_timer = 0.0
        self.current_vel = pygame.math.Vector2(0, 0)
        self.kick_cooldown = 0.0

        # Authentic Gameplay & Physics States
        self.shot_charge = 0.0          # 0.0 to 1.0 power meter
        self.is_charging_shot = False
        self.is_jockeying = False       # Defensive containment stance
        self.is_sprinting = False
        self.first_touch_cooldown = 0.0
        self.knock_ahead_cooldown = 0.0

    @property
    def role(self):
        return self.role_str

    def update(self, dt):
        self.kick_cooldown = max(0.0, self.kick_cooldown - dt)

        # 1. Acceleration & turning inertia
        target_vel = self.velocity.copy()
        accel_rate = 950.0 * dt

        vel_diff = target_vel - self.current_vel
        if vel_diff.length_squared() > 0:
            if vel_diff.length() <= accel_rate:
                self.current_vel = target_vel.copy()
            else:
                self.current_vel += vel_diff.normalize() * accel_rate

        self.velocity = self.current_vel.copy()

        # 2. Fatigue & Stamina Logic
        current_speed_len = self.velocity.length()
        if current_speed_len > self.speed * 1.1:
            self.profile.current_stamina -= 6.0 * dt
        elif current_speed_len < self.speed * 0.4:
            self.profile.current_stamina += 3.5 * dt
        self.profile.current_stamina = max(0.0, min(100.0, self.profile.current_stamina))

        # 3. Apply attributes to actual speed
        mult = self.profile.get_current_speed_mult()
        actual_speed = (self.speed + (self.profile.pace - 75)) * mult

        if current_speed_len > 0:
            is_sprinting = self.is_controlled and (
                pygame.key.get_pressed()[pygame.K_LSHIFT] or pygame.key.get_pressed()[pygame.K_RSHIFT]
            )
            target_mag = min(current_speed_len, actual_speed * (1.45 if is_sprinting else 1.0))
            self.velocity = self.velocity.normalize() * target_mag

        super().update(dt)

        # Clamp to pitch bounds
        self.position.x = max(self.radius, min(self.position.x, settings.SCREEN_WIDTH - self.radius))
        self.position.y = max(self.radius, min(self.position.y, settings.SCREEN_HEIGHT - self.radius))
        self.first_touch_cooldown = max(0.0, self.first_touch_cooldown - dt)
        self.knock_ahead_cooldown = max(0.0, self.knock_ahead_cooldown - dt)

    def handle_human_input(self, ball, teammates, target_goal, opponents, match=None, dt=0.016):
        """Processes keyboard input for the human-controlled player with authentic mechanics."""
        keys = pygame.key.get_pressed()

        direction = pygame.math.Vector2(0, 0)
        if keys[pygame.K_w] or keys[pygame.K_UP]: direction.y -= 1
        if keys[pygame.K_s] or keys[pygame.K_DOWN]: direction.y += 1
        if keys[pygame.K_a] or keys[pygame.K_LEFT]: direction.x -= 1
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]: direction.x += 1

        if direction.length_squared() > 0:
            direction = direction.normalize()
            self.facing = direction.copy()

        is_sprint = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        self.is_sprinting = is_sprint

        # Defensive Jockey Stance: Hold C when not in possession of the ball
        is_jockey_input = (keys[pygame.K_c] or keys[pygame.K_LALT]) and not self.can_kick(ball)
        self.is_jockeying = is_jockey_input

        if self.is_jockeying:
            # Squared hips, containment speed, facing ball directly
            current_speed = self.speed * 0.65
            to_ball = ball.position - self.position
            if to_ball.length_squared() > 0:
                self.facing = to_ball.normalize()
            # Enhanced interception / tackle zone
            if self.position.distance_to(ball.position) < settings.TACKLE_RANGE * 1.4:
                self._attempt_tackle(ball, opponents, jockey=True)
        else:
            current_speed = self.sprint_speed if is_sprint else self.speed

        self.velocity = direction * current_speed

        # Knock-Ahead Skill Burst (Key [E] or Sprint Burst with ball)
        if (keys[pygame.K_e]) and self.can_kick(ball) and self.knock_ahead_cooldown <= 0:
            self.knock_ahead(ball)
            self.knock_ahead_cooldown = 0.6
            return

        # --- Kicking & Power Meter Actions ---
        shoot_key_pressed = keys[pygame.K_j] or keys[pygame.K_z]

        # 1. Shot Power Meter Charging
        if shoot_key_pressed and self.can_kick(ball):
            self.is_charging_shot = True
            self.shot_charge = min(1.0, self.shot_charge + dt * 1.7)
            # Slow player slightly while setting up shooting stance
            self.velocity *= 0.6
            return
        elif self.is_charging_shot:
            # Key released! Execute shot with charged power
            if self.can_kick(ball):
                opp_gk = next((p for p in opponents if p.role_str == "GK"), None)
                self.shoot(ball, target_goal, power_ratio=self.shot_charge, match=match, opp_gk=opp_gk)
                self.kick_cooldown = 0.45
            self.is_charging_shot = False
            self.shot_charge = 0.0
            return

        # 2. Standard Passing / Tackling (K or X)
        if self.can_kick(ball):
            tendency_profiler.record_touch((self.position.x, self.position.y))

        if self.kick_cooldown <= 0:
            if keys[pygame.K_k] or keys[pygame.K_x]:
                if self.can_kick(ball):
                    tendency_profiler.record_pass(is_through=False)
                    target = self._find_directional_pass_target(teammates)
                    if target:
                        opp_team = getattr(match, 'team_b', None) if self.team_id == 0 else getattr(match, 'team_a', None)
                        if opp_team and SpatialGraphAnalytics.is_player_offside(target, opp_team, 1 if self.team_id == 0 else -1):
                            if match: match.trigger_offside(self.team_id)
                        else:
                            self.pass_ball(ball, target.position)
                    else:
                        self.pass_ball(ball, self.position + self.facing * 260)
                    self.kick_cooldown = 0.35
                else:
                    self._attempt_tackle(ball, opponents)
                    self.kick_cooldown = 0.4

            # 3. Through-Ball / Lofted Chip (L or C when possessing ball)
            elif keys[pygame.K_l] or (keys[pygame.K_c] and self.can_kick(ball)):
                if self.can_kick(ball):
                    tendency_profiler.record_pass(is_through=True)
                    target = self._find_directional_pass_target(teammates)
                    if target:
                        lead_vec = target.facing * 95.0 if target.velocity.length() > 20 else target.facing * 45.0
                        target_pos = target.position + lead_vec
                        self.pass_ball(ball, target_pos, lofted=True)
                    else:
                        self.pass_ball(ball, self.position + self.facing * 340, lofted=True)
                    self.kick_cooldown = 0.5

    def _find_directional_pass_target(self, teammates):
        best = None
        best_score = float('-inf')

        for tm in teammates:
            if tm is self or tm.role_str == "GK":
                continue
            to_tm = tm.position - self.position
            dist = to_tm.length()
            if dist < 40 or dist > 650:
                continue

            alignment = self.facing.dot(to_tm.normalize())
            if alignment < 0.15:
                continue

            score = alignment * 220.0 - dist * 0.15
            if score > best_score:
                best_score = score
                best = tm

        return best

    def _attempt_tackle(self, ball, opponents, jockey=False):
        reach = settings.TACKLE_RANGE * (1.4 if jockey else 1.0)
        dist_to_ball = self.position.distance_to(ball.position)
        if dist_to_ball < reach:
            for opp in opponents:
                if opp.position.distance_to(ball.position) < reach + 10:
                    if jockey:
                        # Clean interception & containment trap
                        ball.velocity = self.facing * 60.0
                        ball.last_touched_by = self.id
                    else:
                        poke_dir = self.facing.copy()
                        ball.kick(poke_dir, 360.0)
                    match_stats.record_tackle(self.team_id)
                    break

    def knock_ahead(self, ball):
        """Pushes ball forward into open space for an explosive sprint burst."""
        if self.can_kick(ball):
            burst_dir = self.facing.copy() if self.facing.length_squared() > 0 else pygame.math.Vector2(1, 0)
            ball.kick(burst_dir, settings.PLAYER_SPEED * 1.85, lift=0.0)
            self.profile.current_stamina = max(0.0, self.profile.current_stamina - 4.0)

    def can_kick(self, ball):
        distance = self.position.distance_to(ball.position)
        return (distance < (self.radius + ball.radius + settings.KICK_RANGE)) and (ball.z < 60.0)

    def pass_ball(self, ball, target_pos=None, lofted=False):
        if self.can_kick(ball):
            if target_pos is not None:
                direction = pygame.math.Vector2(target_pos) - self.position
            else:
                direction = self.facing.copy()

            if direction.length_squared() > 0:
                direction = direction.normalize()

            error_range = (100 - self.profile.passing) * 0.20
            direction = direction.rotate(random.uniform(-error_range, error_range))

            lift_val = settings.CHIP_LIFT if lofted else (45.0 if ball.z > 15 else 0.0)
            spin_val = random.uniform(-0.2, 0.2) if self.profile.passing > 80 else 0.0

            ball.kick(direction, settings.PASS_POWER, lift=lift_val, spin=spin_val)
            match_stats.record_pass(self.team_id)

    def shoot(self, ball, target, power_ratio=0.7, chip=False, match=None, opp_gk=None):
        """Executes shot with power meter physics, xG statistical calculation, and spread error."""
        if self.can_kick(ball):
            is_header = (ball.z > 25.0)

            # Calculate mathematical Expected Goals (xG)
            xg = XGEngine.calculate_xg(
                self.position,
                target,
                goalkeeper=opp_gk,
                shooter_profile=self.profile,
                is_header=is_header
            )

            # Inform match of shot and log to shot events
            if match:
                match.record_shot(self.team_id, xg, shot_pos=(self.position.x, self.position.y))
            tendency_profiler.record_shot((self.position.x, self.position.y), xg, target, match_time=match.time_elapsed if match else 0.0)

            goal_pos = pygame.math.Vector2(target)
            target_corner_y = goal_pos.y + random.uniform(-80, 80)
            aim_point = pygame.math.Vector2(goal_pos.x, target_corner_y)

            direction = aim_point - self.position
            if direction.length_squared() > 0:
                direction = direction.normalize()

            # Power ratio physics tiers:
            # Low charge (<0.40): Low driven ground placement shot
            # Sweet spot (0.40 - 0.88): Thunderous on-target strike
            # Overcharge (>0.88): Wild skied shot
            if power_ratio < 0.40:
                shot_power = settings.SHOOT_POWER * (0.65 + power_ratio * 0.5)
                lift_val = 15.0 if ball.z > 12 else 0.0
                error_range = (100 - self.profile.shooting) * 0.12
            elif power_ratio <= 0.88:
                shot_power = settings.SHOOT_POWER * (0.85 + power_ratio * 0.40) + (self.profile.shooting - 75) * 3.0
                lift_val = 320.0 if chip else (115.0 if is_header or random.random() < 0.45 else 10.0)
                error_range = (100 - self.profile.shooting) * 0.16
            else:
                # Skied strike over crossbar
                shot_power = settings.SHOOT_POWER * 1.32
                lift_val = 360.0
                error_range = (100 - self.profile.shooting) * 0.38

            direction = direction.rotate(random.uniform(-error_range, error_range))
            spin_val = random.choice([-0.7, 0.7]) if self.profile.shooting > 80 else 0.0

            ball.kick(direction, shot_power, lift=lift_val, spin=spin_val)
            match_stats.record_shot(self.team_id)