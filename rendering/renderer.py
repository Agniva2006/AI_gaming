import pygame
from engine import settings


class Renderer:
    """
    Handles ALL drawing: pitch, entities, HUD.
    Game logic never touches pygame.draw.
    """
    def __init__(self, screen):
        self.screen = screen
        self.font = pygame.font.SysFont("Arial", 36, bold=True)
        self.small_font = pygame.font.SysFont("Arial", 20)

    def render(self, entities, controlled_player=None, match=None):
        # 1. Draw pitch background and markings
        self._draw_pitch()

        # 2. Draw goals
        self._draw_goals()

        # 3. Draw all entities (players + ball)
        for entity in entities:
            pygame.draw.circle(
                self.screen,
                entity.color,
                (int(entity.position.x), int(entity.position.y)),
                entity.radius
            )

        # 4. Draw highlight ring around controlled player
        if controlled_player:
            pygame.draw.circle(
                self.screen,
                settings.CONTROLLED_HIGHLIGHT,
                (int(controlled_player.position.x), int(controlled_player.position.y)),
                controlled_player.radius + 4,
                3  # ring width
            )

        # 5. Draw score HUD
        if match:
            self._draw_hud(match)

    def _draw_pitch(self):
        """Draw the field background and line markings."""
        self.screen.fill(settings.FIELD_COLOR)

        w = settings.SCREEN_WIDTH
        h = settings.SCREEN_HEIGHT
        lc = settings.LINE_COLOR

        # Center line
        pygame.draw.line(self.screen, lc, (w // 2, 0), (w // 2, h), 2)

        # Center circle
        pygame.draw.circle(self.screen, lc, (w // 2, h // 2), settings.CENTER_CIRCLE_RADIUS, 2)

        # Center dot
        pygame.draw.circle(self.screen, lc, (w // 2, h // 2), 5)

        # Pitch border
        pygame.draw.rect(self.screen, lc, (0, 0, w, h), 3)

        # Left penalty area
        pa_w = settings.PENALTY_AREA_WIDTH
        pa_h = settings.PENALTY_AREA_HEIGHT
        pa_top = settings.PENALTY_AREA_TOP
        pygame.draw.rect(self.screen, lc, (0, pa_top, pa_w, pa_h), 2)

        # Right penalty area
        pygame.draw.rect(self.screen, lc, (w - pa_w, pa_top, pa_w, pa_h), 2)

    def _draw_goals(self):
        """Draw the goal areas on both sides."""
        gw = settings.GOAL_WIDTH
        gt = settings.GOAL_TOP
        gh = settings.GOAL_HEIGHT
        w = settings.SCREEN_WIDTH

        # Left goal (white rectangle extending left beyond pitch)
        pygame.draw.rect(self.screen, (255, 255, 255), (-gw, gt, gw, gh), 0)
        pygame.draw.rect(self.screen, (200, 200, 200), (-gw, gt, gw, gh), 3)

        # Right goal
        pygame.draw.rect(self.screen, (255, 255, 255), (w, gt, gw, gh), 0)
        pygame.draw.rect(self.screen, (200, 200, 200), (w, gt, gw, gh), 3)

        # Goal line markers (thicker lines where goals are)
        pygame.draw.line(self.screen, (255, 255, 255), (0, gt), (0, gt + gh), 4)
        pygame.draw.line(self.screen, (255, 255, 255), (w, gt), (w, gt + gh), 4)

    def _draw_hud(self, match):
        """Draw the score and clock at the top of the screen."""
        w = settings.SCREEN_WIDTH

        # Score text
        score_text = f"{match.score[0]}  -  {match.score[1]}"
        text_surface = self.font.render(score_text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(w // 2, 30))

        # Background bar for readability
        bar_rect = pygame.Rect(w // 2 - 120, 5, 240, 65)
        bar_surface = pygame.Surface((240, 65), pygame.SRCALPHA)
        bar_surface.fill((0, 0, 0, 150))
        self.screen.blit(bar_surface, bar_rect)

        self.screen.blit(text_surface, text_rect)

        # Team labels
        a_label = self.small_font.render("BLUE", True, settings.TEAM_A_COLOR)
        b_label = self.small_font.render("RED", True, settings.TEAM_B_COLOR)
        self.screen.blit(a_label, (w // 2 - 100, 12))
        self.screen.blit(b_label, (w // 2 + 58, 12))
        
        # Clock
        minutes = int(match.time_elapsed // 60)
        seconds = int(match.time_elapsed % 60)
        clock_text = f"{minutes:02d}:{seconds:02d}"
        if match.state == "FULL_TIME":
            clock_text = "FULL TIME"
            
        time_surf = self.small_font.render(clock_text, True, (255, 255, 255))
        self.screen.blit(time_surf, (w // 2 - 20, 48))

        # Goal scored flash
        if match.state == "GOAL_SCORED":
            goal_text = self.font.render("GOAL!", True, (255, 255, 0))
            goal_rect = goal_text.get_rect(center=(w // 2, 80))
            self.screen.blit(goal_text, goal_rect)
