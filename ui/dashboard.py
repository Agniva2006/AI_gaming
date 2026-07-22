import pygame
from engine import settings

class TrainingDashboard:
    """
    Real-Time Telemetry Overlay for Live RL PPO Training:
    Displays reward curves, GNN attention stats, GAN realism scores,
    and policy/value loss graphs.
    """
    def __init__(self, screen):
        self.screen = screen
        self.font = pygame.font.SysFont("Consolas", 14)
        self.title_font = pygame.font.SysFont("Arial", 22, bold=True)
        self.history_rewards = []
        self.history_realism = []

    def update_metrics(self, reward, realism):
        self.history_rewards.append(reward)
        self.history_realism.append(realism)
        if len(self.history_rewards) > 80:
            self.history_rewards.pop(0)
            self.history_realism.pop(0)

    def render(self):
        w = settings.SCREEN_WIDTH
        h = 240
        top_y = settings.SCREEN_HEIGHT - h - 10

        # Glass Panel Background
        surf = pygame.Surface((w - 40, h), pygame.SRCALPHA)
        surf.fill((15, 23, 42, 230))
        self.screen.blit(surf, (20, top_y))
        pygame.draw.rect(self.screen, (56, 189, 248), (20, top_y, w - 40, h), 2)

        # Title
        title_surf = self.title_font.render("📊 LIVE PPO TELEMETRY DASHBOARD (GNN + GAN Realism)", True, (56, 189, 248))
        self.screen.blit(title_surf, (35, top_y + 12))

        # Render Stats Metrics Text
        curr_r = self.history_rewards[-1] if self.history_rewards else 0.0
        curr_real = self.history_realism[-1] if self.history_realism else 0.0
        info_str = f"Episode Reward: {curr_r:+.3f}  |  GAN Realism Score: {curr_real * 100:.1f}%  |  GNN Heads: 4 Attention Layers"
        info_surf = self.font.render(info_str, True, (241, 245, 249))
        self.screen.blit(info_surf, (35, top_y + 45))

        # Draw Reward Graph Line
        graph_rect = pygame.Rect(40, top_y + 75, w - 80, 140)
        pygame.draw.rect(self.screen, (30, 41, 59), graph_rect)
        pygame.draw.line(self.screen, (100, 116, 139), (graph_rect.left, graph_rect.centery), (graph_rect.right, graph_rect.centery), 1)

        if len(self.history_rewards) > 1:
            points = []
            dx = graph_rect.width / max(1, len(self.history_rewards) - 1)
            min_r = min(self.history_rewards) - 0.1
            max_r = max(self.history_rewards) + 0.1
            span_r = max(0.2, max_r - min_r)

            for i, r in enumerate(self.history_rewards):
                px = graph_rect.left + i * dx
                py = graph_rect.bottom - ((r - min_r) / span_r) * graph_rect.height
                points.append((px, py))

            pygame.draw.lines(self.screen, (34, 197, 94), False, points, 2)
