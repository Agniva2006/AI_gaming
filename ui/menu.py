import pygame
from engine import settings

class MainMenu:
    """
    Modern glassmorphic main menu for RL Train Football:
    Selection between autonomous GNN-PPO matches, live training dashboard,
    spatial Voronoi analytics, and tactical formation customization.
    """
    def __init__(self, screen):
        self.screen = screen
        self.font = pygame.font.SysFont("Arial", 42, bold=True)
        self.sub_font = pygame.font.SysFont("Arial", 22)
        self.btn_font = pygame.font.SysFont("Arial", 24, bold=True)

        self.options = [
            ("1. RL TACTICAL CLASH (100% Autonomous GNN-PPO Models)", "neural_match"),
            ("2. LIVE PPO SELF-PLAY TRAINING (With Dashboard)", "train_mode"),
            ("3. SPATIAL VORONOI ANALYTICS SHOWCASE", "voronoi"),
            ("4. FORMATION & TACTICAL MANAGER (4-3-3 / 3-5-2 / 4-4-2)", "tactics"),
            ("5. QUIT GAME", "quit")
        ]
        self.selected_idx = 0

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP or event.key == pygame.K_w:
                self.selected_idx = (self.selected_idx - 1) % len(self.options)
            elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                self.selected_idx = (self.selected_idx + 1) % len(self.options)
            elif event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                return self.options[self.selected_idx][1]
        return None

    def render(self):
        w, h = settings.SCREEN_WIDTH, settings.SCREEN_HEIGHT
        self.screen.fill((15, 23, 42))

        # Title Header
        title_surf = self.font.render("⚽ RL TRAIN FOOTBALL PRO", True, (255, 255, 255))
        self.screen.blit(title_surf, title_surf.get_rect(center=(w // 2, 105)))

        subtitle = self.sub_font.render("Autonomous GNN-PPO Policy • GAN Realism Discriminator • Diffusion Scenarios", True, (56, 189, 248))
        self.screen.blit(subtitle, subtitle.get_rect(center=(w // 2, 155)))

        # Card Menu Options
        start_y = 220
        for i, (label, mode) in enumerate(self.options):
            is_selected = (i == self.selected_idx)
            card_rect = pygame.Rect(w // 2 - 320, start_y + i * 72, 640, 58)

            card_surf = pygame.Surface((640, 58), pygame.SRCALPHA)
            bg_color = (30, 41, 59, 240) if not is_selected else (56, 189, 248, 220)
            card_surf.fill(bg_color)

            self.screen.blit(card_surf, card_rect)

            text_color = (15, 23, 42) if is_selected else (241, 245, 249)
            btn_surf = self.btn_font.render(label, True, text_color)
            self.screen.blit(btn_surf, btn_surf.get_rect(center=card_rect.center))

            if is_selected:
                pygame.draw.rect(self.screen, (255, 255, 255), card_rect, 3)

        footer = self.sub_font.render("Press [W/S or UP/DOWN] to Navigate • [ENTER/SPACE] to Launch Mode", True, (148, 163, 184))
        self.screen.blit(footer, footer.get_rect(center=(w // 2, h - 40)))
