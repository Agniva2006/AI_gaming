import pygame
import random
from entities.entity import Entity
from engine import settings
from attributes.profile import PlayerProfile
from stats.tracker import match_stats

class Player(Entity):
    """
    A football player. Upgraded with PlayerProfile (Phase A) 
    and tactical roles (Phase B).
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

    @property
    def role(self):
        return self.role_str

    def update(self, dt):
        if self.is_controlled:
            self._handle_input()

        # 1. Physical Acceleration & Turning Inertia
        # Target velocity is set by AI/input; actual velocity smoothly accelerates/turns toward target
        target_vel = self.velocity.copy()
        current_vel = getattr(self, 'current_vel', pygame.math.Vector2(0, 0))
        accel_rate = 800.0 * dt  # px/s² linear acceleration rate

        vel_diff = target_vel - current_vel
        if vel_diff.length_squared() > 0:
            if vel_diff.length() <= accel_rate:
                current_vel = target_vel.copy()
            else:
                current_vel += vel_diff.normalize() * accel_rate

        self.current_vel = current_vel
        self.velocity = current_vel.copy()

        # 2. Phase A: Fatigue & Stamina Logic
        current_speed_len = self.velocity.length()
        if current_speed_len > self.speed * 1.1:
            self.profile.current_stamina -= 5.0 * dt  # sprinting drains
        elif current_speed_len < self.speed * 0.5:
            self.profile.current_stamina += 2.0 * dt  # resting recovers
        self.profile.current_stamina = max(0, min(100, self.profile.current_stamina))

        # 3. Apply attributes to actual speed
        mult = self.profile.get_current_speed_mult()
        actual_speed = (self.speed + (self.profile.pace - 75)) * mult
        
        if current_speed_len > 0:
            target_mag = min(current_speed_len, actual_speed * (1.5 if self.is_controlled and pygame.key.get_pressed()[pygame.K_LSHIFT] else 1.0))
            self.velocity = self.velocity.normalize() * target_mag

        super().update(dt)

        self.position.x = max(self.radius, min(self.position.x, settings.SCREEN_WIDTH - self.radius))
        self.position.y = max(self.radius, min(self.position.y, settings.SCREEN_HEIGHT - self.radius))
        
        self.decision_timer = max(0, self.decision_timer - dt)


    def _handle_input(self):
        keys = pygame.key.get_pressed()

        direction = pygame.math.Vector2(0, 0)
        if keys[pygame.K_w]: direction.y -= 1
        if keys[pygame.K_s]: direction.y += 1
        if keys[pygame.K_a]: direction.x -= 1
        if keys[pygame.K_d]: direction.x += 1

        if direction.length_squared() > 0:
            direction = direction.normalize()
            self.facing = direction.copy()

        current_speed = self.sprint_speed if keys[pygame.K_LSHIFT] else self.speed
        self.velocity = direction * current_speed

    def can_kick(self, ball):
        distance = self.position.distance_to(ball.position)
        # Check 2D distance and 3D height threshold (players can reach up to z=60 for headers/volleys)
        return (distance < (self.radius + ball.radius + settings.KICK_RANGE)) and (ball.z < 60.0)

    def pass_ball(self, ball, target_pos=None, lofted=False):
        if self.can_kick(ball):
            if target_pos is not None:
                direction = pygame.math.Vector2(target_pos) - self.position
            else:
                direction = self.facing
            
            if direction.length_squared() > 0:
                direction = direction.normalize()
                
            # Phase C: Passing error margin based on attributes
            error_range = (100 - self.profile.passing) * 0.4
            angle_offset = random.uniform(-error_range, error_range)
            direction = direction.rotate(angle_offset)
            
            lift_val = 250.0 if lofted else (40.0 if ball.z > 15 else 0.0)
            spin_val = random.uniform(-0.3, 0.3) if self.profile.passing > 80 else 0.0
            
            ball.kick(direction, settings.PASS_POWER, lift=lift_val, spin=spin_val)
            match_stats.record_pass(self.team_id)

    def shoot(self, ball, target, chip=False):
        if self.can_kick(ball):
            goal_pos = pygame.math.Vector2(target)
            direction = goal_pos - self.position
            
            if direction.length_squared() > 0:
                direction = direction.normalize()
                
            error_range = (100 - self.profile.shooting) * 0.3
            direction = direction.rotate(random.uniform(-error_range, error_range))
            
            shot_power = settings.SHOOT_POWER + (self.profile.shooting - 75) * 3
            lift_val = 320.0 if chip else (120.0 if random.random() < 0.4 else 0.0)
            spin_val = random.choice([-0.8, 0.8]) if self.profile.shooting > 82 else 0.0
            
            ball.kick(direction, shot_power, lift=lift_val, spin=spin_val)
            match_stats.record_shot(self.team_id)