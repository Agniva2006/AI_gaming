import pygame

class Entity:
    """
    Base class for all physical objects in the game.
    Upgraded to 3D coordinate space (x, y, z) for realistic aerial trajectories,
    shadows, and collision volumes.
    """
    def __init__(self, x, y, radius, color, z=0.0):
        self.position_3d = pygame.math.Vector3(float(x), float(y), float(z))
        self.velocity_3d = pygame.math.Vector3(0.0, 0.0, 0.0)
        self.radius = radius
        self.color = color

    @property
    def position(self):
        return pygame.math.Vector2(self.position_3d.x, self.position_3d.y)

    @position.setter
    def position(self, vec2):
        self.position_3d.x = float(vec2.x)
        self.position_3d.y = float(vec2.y)

    @property
    def velocity(self):
        return pygame.math.Vector2(self.velocity_3d.x, self.velocity_3d.y)

    @velocity.setter
    def velocity(self, vec2):
        self.velocity_3d.x = float(vec2.x)
        self.velocity_3d.y = float(vec2.y)

    @property
    def z(self):
        return self.position_3d.z

    @z.setter
    def z(self, val):
        self.position_3d.z = float(val)

    def update(self, dt):
        self.position_3d += self.velocity_3d * dt