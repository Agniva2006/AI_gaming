import pygame
import random
import math

class Particle:
    def __init__(self, x, y, vx, vy, color, lifetime, radius=2):
        self.x = float(x)
        self.y = float(y)
        self.vx = float(vx)
        self.vy = float(vy)
        self.color = color
        self.lifetime = float(lifetime)
        self.max_lifetime = float(lifetime)
        self.radius = radius

    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.lifetime -= dt

    @property
    def alive(self):
        return self.lifetime > 0


class ParticleSystem:
    """
    High-performance particle engine for turf dirt, ball streak trails,
    sprint dust, and goal sparks.
    """
    def __init__(self):
        self.particles = []

    def emit_turf(self, x, y, count=5):
        """Emit grass dirt particles when players tackle or sprint hard."""
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(30, 120)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            color = random.choice([(40, 160, 40), (20, 100, 20), (100, 80, 40)])
            self.particles.append(Particle(x, y, vx, vy, color, lifetime=random.uniform(0.2, 0.5), radius=random.randint(2, 4)))

    def emit_ball_trail(self, x, y, count=2):
        """Emit motion streak particles behind fast-moving balls."""
        for _ in range(count):
            vx = random.uniform(-10, 10)
            vy = random.uniform(-10, 10)
            color = (255, 255, 200)
            self.particles.append(Particle(x, y, vx, vy, color, lifetime=0.15, radius=3))

    def emit_goal_sparks(self, x, y, count=30):
        """Emit celebration fireworks/sparks when a goal is scored."""
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(100, 300)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            color = random.choice([(255, 215, 0), (255, 255, 255), (255, 50, 50), (50, 150, 255)])
            self.particles.append(Particle(x, y, vx, vy, color, lifetime=random.uniform(0.5, 1.2), radius=random.randint(3, 6)))

    def update(self, dt):
        for p in self.particles:
            p.update(dt)
        self.particles = [p for p in self.particles if p.alive]

    def render(self, screen, camera):
        for p in self.particles:
            screen_pos = camera.world_to_screen(pygame.math.Vector2(p.x, p.y))
            r = camera.scale(p.radius)
            if r > 0:
                alpha = int(255 * (p.lifetime / p.max_lifetime))
                surf = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
                pygame.draw.circle(surf, (*p.color, max(0, alpha)), (r, r), r)
                screen.blit(surf, (screen_pos[0] - r, screen_pos[1] - r))
