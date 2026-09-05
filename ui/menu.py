import pygame
from engine import settings
from tactics.formations import AVAILABLE_FORMATIONS

class MainMenu:
    """
    Tactical Main Menu:
    Configure formations for Human and RL AI, select match modes,
    and launch training or spectator simulations.
    """
    def __init__(self, screen):
        self.screen = screen
        self.font_title = pygame.font.SysFont("Arial", 40, bold=True)
        self.font_sub = pygame.font.SysFont("Arial", 20)
        self.font_btn = pygame.font.SysFont("Arial", 22, bold=True)
        self.font_hint = pygame.font.SysFont("Arial", 16)

        self.human_formation_idx = 1  # 4-3-3 default
        self.ai_formation_idx = 0     # 4-4-2 default
        self.selected_idx = 0

        self.options = [
            "start_human",
            "toggle_human_formation",
            "toggle_ai_formation",
            "start_ai_vs_ai",
            "train_mode",
            "quit"
        ]

    @property
    def human_formation(self):
        return AVAILABLE_FORMATIONS[self.human_formation_idx % len(AVAILABLE_FORMATIONS)]

    @property
    def ai_formation(self):
        return AVAILABLE_FORMATIONS[self.ai_formation_idx % len(AVAILABLE_FORMATIONS)]

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key in [pygame.K_UP, pygame.K_w]:
                self.selected_idx = (self.selected_idx - 1) % len(self.options)
            elif event.key in [pygame.K_DOWN, pygame.K_s]:
                self.selected_idx = (self.selected_idx + 1) % len(self.options)
            elif event.key in [pygame.K_LEFT, pygame.K_a]:
                if self.options[self.selected_idx] == "toggle_human_formation":
                    self.human_formation_idx = (self.human_formation_idx - 1) % len(AVAILABLE_FORMATIONS)
                elif self.options[self.selected_idx] == "toggle_ai_formation":
                    self.ai_formation_idx = (self.ai_formation_idx - 1) % len(AVAILABLE_FORMATIONS)
            elif event.key in [pygame.K_RIGHT, pygame.K_d]:
                if self.options[self.selected_idx] == "toggle_human_formation":
                    self.human_formation_idx = (self.human_formation_idx + 1) % len(AVAILABLE_FORMATIONS)
                elif self.options[self.selected_idx] == "toggle_ai_formation":
                    self.ai_formation_idx = (self.ai_formation_idx + 1) % len(AVAILABLE_FORMATIONS)
            elif event.key in [pygame.K_RETURN, pygame.K_SPACE]:
                action = self.options[self.selected_idx]
                if action == "toggle_human_formation":
                    self.human_formation_idx = (self.human_formation_idx + 1) % len(AVAILABLE_FORMATIONS)
                    return None
                elif action == "toggle_ai_formation":
                    self.ai_formation_idx = (self.ai_formation_idx + 1) % len(AVAILABLE_FORMATIONS)
                    return None
                return action
        return None

    def render(self):
        w, h = settings.SCREEN_WIDTH, settings.SCREEN_HEIGHT
        self.screen.fill((15, 23, 42))

        # Title Header
        title_surf = self.font_title.render("⚽ 2D FOOTBALL: HUMAN VS LEARNING RL AI", True, (255, 255, 255))
        self.screen.blit(title_surf, title_surf.get_rect(center=(w // 2, 85)))

        sub_text = "Multi-Agent PPO Policy • Continuous Match Learning • SQLite DB & Web Gateway"
        sub_surf = self.font_sub.render(sub_text, True, (56, 189, 248))
        self.screen.blit(sub_surf, sub_surf.get_rect(center=(w // 2, 130)))

        # Menu labels
        labels = [
            ("1. 🎮 PLAY MATCH: HUMAN VS RL AI", "start_human"),
            (f"2. 🛡️ YOUR FORMATION:  < {self.human_formation} >  (←/→ to switch)", "toggle_human_formation"),
            (f"3. 🤖 AI FORMATION:    < {self.ai_formation} >  (←/→ to switch)", "toggle_ai_formation"),
            ("4. 👁️ WATCH AI VS AI SPECTATOR MATCH", "start_ai_vs_ai"),
            ("5. 📈 AI TRAINING SUITE & LIVE TELEMETRY", "train_mode"),
            ("6. 🚪 QUIT GAME", "quit")
        ]

        start_y = 190
        card_w, card_h = 720, 52

        for i, (label, action_name) in enumerate(labels):
            is_selected = (i == self.selected_idx)
            card_rect = pygame.Rect(w // 2 - card_w // 2, start_y + i * 66, card_w, card_h)

            card_surf = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
            if is_selected:
                card_surf.fill((56, 189, 248, 230))
                text_color = (15, 23, 42)
            else:
                card_surf.fill((30, 41, 59, 200))
                text_color = (241, 245, 249)

            self.screen.blit(card_surf, card_rect)

            btn_surf = self.font_btn.render(label, True, text_color)
            self.screen.blit(btn_surf, btn_surf.get_rect(center=card_rect.center))

            if is_selected:
                pygame.draw.rect(self.screen, (255, 255, 255), card_rect, 2, border_radius=4)

        # Controls Guide Footer
        footer_box = pygame.Rect(w // 2 - 450, h - 100, 900, 65)
        f_surf = pygame.Surface((900, 65), pygame.SRCALPHA)
        f_surf.fill((15, 23, 42, 230))
        self.screen.blit(f_surf, footer_box)
        pygame.draw.rect(self.screen, (51, 65, 85), footer_box, 1, border_radius=6)

        c1 = self.font_hint.render("CONTROLS: [WASD / Arrows] Move • [L-Shift] Sprint • [K / X] Pass / Tackle • [J / Z] Shoot • [L / C] Chip Pass", True, (226, 232, 240))
        c2 = self.font_hint.render("[SPACE / TAB] Switch Player • [ESC] Menu • [F3] Debug Telemetry • Use [← / →] to adjust Formations", True, (148, 163, 184))
        self.screen.blit(c1, c1.get_rect(center=(w // 2, h - 80)))
        self.screen.blit(c2, c2.get_rect(center=(w // 2, h - 55)))
