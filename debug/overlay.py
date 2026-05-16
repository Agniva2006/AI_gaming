import pygame

class DebugOverlay:
    """
    Phase D: Visual tactical debugging overlay.
    Toggled via F3.
    """
    def __init__(self):
        self.enabled = False
        pygame.font.init()
        self.font = pygame.font.SysFont("Consolas", 12)

    def toggle(self):
        self.enabled = not self.enabled

    def render(self, screen, all_players):
        if not self.enabled:
            return

        for player in all_players:
            # Draw movement vector (intended velocity)
            if player.velocity.length() > 5:
                target_pt = player.position + (player.velocity * 0.5)
                pygame.draw.line(screen, (255, 255, 0), player.position, target_pt, 1)

            # Draw tactical role and stamina
            info = f"{player.role_str} ({int(player.profile.current_stamina)})"
            text = self.font.render(info, True, (200, 200, 200))
            
            # Background rect for text readability
            bg_rect = pygame.Rect(player.position.x - 15, player.position.y + 15, text.get_width(), text.get_height())
            pygame.draw.rect(screen, (0, 0, 0, 150), bg_rect)
            screen.blit(text, (player.position.x - 15, player.position.y + 15))
