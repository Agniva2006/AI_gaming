import pygame
import math
from engine import settings
from rendering.camera import Camera
from rendering.particles import ParticleSystem
from rendering.net_physics import GoalNet

class Renderer:
    """
    Next-Gen 2.5D Renderer featuring camera viewport transformations,
    directional player shadows, 3D ball altitude rendering, particle systems,
    and physical net ripples.
    """
    def __init__(self, screen):
        self.screen = screen
        self.camera = Camera(settings.SCREEN_WIDTH, settings.SCREEN_HEIGHT)
        self.particle_system = ParticleSystem()
        self.left_net = GoalNet("left")
        self.right_net = GoalNet("right")

        self.font = pygame.font.SysFont("Arial", 32, bold=True)
        self.small_font = pygame.font.SysFont("Arial", 18, bold=True)

        # Precompute pitch stripe surfaces
        self.stripe_width = 80

    def render(self, entities, controlled_player=None, match=None, dt=0.016, show_voronoi=False):
        # Find ball entity
        ball = next((e for e in entities if not hasattr(e, 'role_str')), None)

        # Update camera and particles
        if ball:
            self.camera.update(ball, match, dt)
            if ball.velocity.length() > 200:
                self.particle_system.emit_ball_trail(ball.position.x, ball.position.y)

        self.particle_system.update(dt)
        self.left_net.update(dt)
        self.right_net.update(dt)

        # 1. Clear & Draw pitch background and stripes
        self._draw_pitch()
        
        # 1.5 Draw Voronoi Spatial Graph Analytics
        if show_voronoi:
            self._draw_voronoi(entities)

        # 2. Draw goal nets
        self.left_net.render(self.screen, self.camera)
        self.right_net.render(self.screen, self.camera)

        # 3. Draw Player Drop Shadows
        for entity in entities:
            if hasattr(entity, 'role_str'):
                self._draw_player_shadow(entity)

        # 4. Draw Ball Ground Shadow
        if ball:
            self._draw_ball_shadow(ball)

        # 5. Draw Particles beneath entities
        self.particle_system.render(self.screen, self.camera)

        # 6. Draw Player Entities
        for entity in entities:
            if hasattr(entity, 'role_str'): # Is Player
                self._draw_player(entity, entity is controlled_player)

        # 7. Draw Ball Entity in 3D (elevated by ball.z)
        if ball:
            self._draw_ball_3d(ball)

        # 8. Draw HUD Overlay
        if match:
            self._draw_hud(match)

    def _draw_pitch(self):
        self.screen.fill(settings.FIELD_COLOR)
        w, h = settings.SCREEN_WIDTH, settings.SCREEN_HEIGHT

        # Draw alternating grass stripes
        for x in range(0, w, self.stripe_width * 2):
            stripe_rect = pygame.Rect(x, 0, self.stripe_width, h)
            p1 = self.camera.world_to_screen(pygame.math.Vector2(x, 0))
            p2 = self.camera.world_to_screen(pygame.math.Vector2(x + self.stripe_width, h))
            scaled_rect = pygame.Rect(p1[0], 0, p2[0] - p1[0], self.screen.get_height())
            pygame.draw.rect(self.screen, settings.FIELD_COLOR_DARK, scaled_rect)

        lc = settings.LINE_COLOR

        # Pitch Outer Boundary
        p_tl = self.camera.world_to_screen(pygame.math.Vector2(0, 0))
        p_br = self.camera.world_to_screen(pygame.math.Vector2(w, h))
        pygame.draw.rect(self.screen, lc, (p_tl[0], p_tl[1], p_br[0] - p_tl[0], p_br[1] - p_tl[1]), max(1, self.camera.scale(3)))

        # Halfway Line
        p_top = self.camera.world_to_screen(pygame.math.Vector2(w // 2, 0))
        p_bot = self.camera.world_to_screen(pygame.math.Vector2(w // 2, h))
        pygame.draw.line(self.screen, lc, p_top, p_bot, max(1, self.camera.scale(2)))

        # Center Circle
        p_center = self.camera.world_to_screen(pygame.math.Vector2(w // 2, h // 2))
        r_circle = self.camera.scale(settings.CENTER_CIRCLE_RADIUS)
        if r_circle > 0:
            pygame.draw.circle(self.screen, lc, p_center, r_circle, max(1, self.camera.scale(2)))
            pygame.draw.circle(self.screen, lc, p_center, max(1, self.camera.scale(4)))

        # Penalty Areas
        pa_w = settings.PENALTY_AREA_WIDTH
        pa_h = settings.PENALTY_AREA_HEIGHT
        pa_top = settings.PENALTY_AREA_TOP

        # Left Penalty Box
        p_l1 = self.camera.world_to_screen(pygame.math.Vector2(0, pa_top))
        p_l2 = self.camera.world_to_screen(pygame.math.Vector2(pa_w, pa_top + pa_h))
        pygame.draw.rect(self.screen, lc, (p_l1[0], p_l1[1], p_l2[0] - p_l1[0], p_l2[1] - p_l1[1]), max(1, self.camera.scale(2)))

        # Right Penalty Box
        p_r1 = self.camera.world_to_screen(pygame.math.Vector2(w - pa_w, pa_top))
        p_r2 = self.camera.world_to_screen(pygame.math.Vector2(w, pa_top + pa_h))
        pygame.draw.rect(self.screen, lc, (p_r1[0], p_r1[1], p_r2[0] - p_r1[0], p_r2[1] - p_r1[1]), max(1, self.camera.scale(2)))

    def _draw_player_shadow(self, player):
        center = self.camera.world_to_screen(player.position + pygame.math.Vector2(4, 6))
        rx = self.camera.scale(player.radius + 2)
        ry = self.camera.scale((player.radius + 2) * 0.5)

        if rx > 0 and ry > 0:
            surf = pygame.Surface((rx * 2, ry * 2), pygame.SRCALPHA)
            pygame.draw.ellipse(surf, (0, 0, 0, 100), (0, 0, rx * 2, ry * 2))
            self.screen.blit(surf, (center[0] - rx, center[1] - ry))

    def _draw_ball_shadow(self, ball):
        # Shadow position stays on the ground (z=0)
        center = self.camera.world_to_screen(ball.position + pygame.math.Vector2(2, 4))
        shadow_scale = max(0.4, 1.0 - (ball.z / 250.0))
        rx = self.camera.scale(ball.radius * shadow_scale * 1.2)
        ry = self.camera.scale(ball.radius * shadow_scale * 0.6)
        alpha = int(140 * shadow_scale)

        if rx > 0 and ry > 0:
            surf = pygame.Surface((rx * 2, ry * 2), pygame.SRCALPHA)
            pygame.draw.ellipse(surf, (0, 0, 0, alpha), (0, 0, rx * 2, ry * 2))
            self.screen.blit(surf, (center[0] - rx, center[1] - ry))

    def _draw_player(self, player, is_controlled):
        center = self.camera.world_to_screen(player.position)
        r = self.camera.scale(player.radius)

        if r <= 0:
            return

        # Player Body Circle
        pygame.draw.circle(self.screen, player.color, center, r)
        pygame.draw.circle(self.screen, (20, 20, 20), center, r, max(1, self.camera.scale(2)))

        # Facing Direction Line
        face_end = self.camera.world_to_screen(player.position + player.facing * (player.radius + 4))
        pygame.draw.line(self.screen, (255, 255, 255), center, face_end, max(1, self.camera.scale(2)))

        # Controlled Highlight Ring & Indicator
        if is_controlled:
            pygame.draw.circle(self.screen, settings.CONTROLLED_HIGHLIGHT, center, r + self.camera.scale(4), max(1, self.camera.scale(3)))
            # Draw inverted triangle marker above head
            marker_top = (center[0], center[1] - r - self.camera.scale(16))
            marker_l = (center[0] - self.camera.scale(6), center[1] - r - self.camera.scale(24))
            marker_r = (center[0] + self.camera.scale(6), center[1] - r - self.camera.scale(24))
            pygame.draw.polygon(self.screen, (255, 255, 0), [marker_top, marker_l, marker_r])

    def _draw_ball_3d(self, ball):
        # Ball offset vertically by its elevation z
        elevated_pos = pygame.math.Vector2(ball.position.x, ball.position.y - ball.z)
        center = self.camera.world_to_screen(elevated_pos)
        r = self.camera.scale(ball.radius)

        if r > 0:
            pygame.draw.circle(self.screen, ball.color, center, r)
            pygame.draw.circle(self.screen, (30, 30, 30), center, r, max(1, self.camera.scale(1)))
            if abs(ball.spin) > 0.1:
                # Draw spin arc accent
                arc_r = max(1, r - 2)
                pygame.draw.arc(self.screen, (255, 50, 50), (center[0]-arc_r, center[1]-arc_r, arc_r*2, arc_r*2), 0, math.pi, 2)

    def _draw_hud(self, match):
        w = settings.SCREEN_WIDTH

        # Top Bar Surface
        bar_surf = pygame.Surface((280, 70), pygame.SRCALPHA)
        bar_surf.fill((15, 23, 42, 220)) # Dark slate glass
        self.screen.blit(bar_surf, (w // 2 - 140, 10))

        # Score text
        score_text = f"{match.score[0]}   -   {match.score[1]}"
        score_surf = self.font.render(score_text, True, (255, 255, 255))
        self.screen.blit(score_surf, score_surf.get_rect(center=(w // 2, 34)))

        # Labels
        a_lbl = self.small_font.render("BLUE", True, settings.TEAM_A_COLOR)
        b_lbl = self.small_font.render("RED", True, settings.TEAM_B_COLOR)
        self.screen.blit(a_lbl, (w // 2 - 120, 18))
        self.screen.blit(b_lbl, (w // 2 + 75, 18))

        # Clock
        mins = int(match.time_elapsed // 60)
        secs = int(match.time_elapsed % 60)
        clock_str = f"{mins:02d}:{secs:02d}" if match.state != "FULL_TIME" else "FULL TIME"
        clock_surf = self.small_font.render(clock_str, True, (220, 220, 220))
        self.screen.blit(clock_surf, clock_surf.get_rect(center=(w // 2, 60)))

    def _draw_voronoi(self, entities):
        players = [e for e in entities if hasattr(e, 'role_str') and e.role_str != 'GK']
        if not players: return
        
        # We'll use a dynamic surface covering the exact pitch dimensions (world space)
        # to ensure it pans and scales perfectly with the camera.
        step = 40
        w, h = settings.SCREEN_WIDTH, settings.SCREEN_HEIGHT
        
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        
        # Calculate dominance zones
        for x in range(0, w, step):
            for y in range(0, h, step):
                # Find closest player
                closest = min(players, key=lambda p: (p.position.x - x)**2 + (p.position.y - y)**2)
                color = closest.color
                
                # Create semi-transparent overlay tile
                rect = pygame.Rect(x, y, step, step)
                pygame.draw.rect(surf, (color[0], color[1], color[2], 60), rect)
                
        # Draw borders for the cells
        for x in range(0, w, step):
            pygame.draw.line(surf, (255, 255, 255, 15), (x, 0), (x, h))
        for y in range(0, h, step):
            pygame.draw.line(surf, (255, 255, 255, 15), (0, y), (w, y))

        # Project world surface to screen space via camera
        p_tl = self.camera.world_to_screen(pygame.math.Vector2(0, 0))
        p_br = self.camera.world_to_screen(pygame.math.Vector2(w, h))
        
        scaled_w = p_br[0] - p_tl[0]
        scaled_h = p_br[1] - p_tl[1]
        
        if scaled_w > 0 and scaled_h > 0:
            scaled_surf = pygame.transform.scale(surf, (int(scaled_w), int(scaled_h)))
            self.screen.blit(scaled_surf, p_tl)
