import pygame


# Base class for all physical objects in the game (players, ball, etc.)
# WHY Vector2? It lets us do clean math like: position += velocity * dt
# instead of separate self.x += self.vx * dt AND self.y += self.vy * dt
class Entity:
    def __init__(self, x, y, radius, color):
        self.position = pygame.math.Vector2(x, y)
        self.velocity = pygame.math.Vector2(0, 0)
        self.radius = radius
        self.color = color

    def update(self, dt):
        # Move entity based on its current velocity
        # Child classes set self.velocity, then call super().update(dt)
        self.position += self.velocity * dt