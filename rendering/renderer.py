import pygame
import math
from engine import settings
from rendering.camera import Camera
from rendering.particles import ParticleSystem
from rendering.net_physics import GoalNet
from analytics.spatial_graph import SpatialGraphAnalytics

class Renderer:
    """
    Next-Gen 2.5D Football Renderer:
    Camera viewport, drop shadows, 3D ball altitude, goal net ripples,
    VAR offside line, tactical passing network, Expected Goals (xG) HUD, and minimap radar.
    """
    def __init__(self, screen):
        self.screen = screen
        self.camera = Camera(settings.SCREEN_WIDTH, settings.SCREEN_HEIGHT)
        self.particle_system = ParticleSystem()
        self.left_net = GoalNet("left")
        self.right_net = GoalNet("right")

        self.font = pygame.font.SysFont("Arial", 28, bold=True)
        self.score_font = pygame.font.SysFont("Arial", 32, bold=True)
        self.small_font = pygame.font.SysFont("Arial", 15, bold=True)
        self.tiny_font = pygame.font.SysFont("Arial", 11, bold=True)
        self.banner_font = pygame.font.SysFont("Arial", 42, bold=True)

        self.stripe_width = 80

    def render(self, entities, controlled_player=None, match=None, dt=0.016, show_tactics=False, tactical_name="Balanced", show_tacticai=False):
        ball = next((e for e in entities if not hasattr(e, 'role_str')), None)

        if ball:
            self.camera.update(ball, match, dt)
            if ball.velocity.length() > 220:
                self.particle_system.emit_ball_trail(ball.position.x, ball.position.y)

        self.particle_system.update(dt)
        self.left_net.update(dt)
        self.right_net.update(dt)

        # 1. Pitch background and markings
        self._draw_pitch()

        # 2. Tactical Overlay (VAR Offside Line & Passing Triangles)
        if show_tactics and match:
            self._draw_tactical_overlay(match)

        # 2b. TacticAI Predictive Overlay (Google DeepMind Nature 2024)
        if show_tacticai and match and ball:
            self._draw_tacticai_overlay(match, controlled_player, ball)

        # 3. Goal nets
        self.left_net.render(self.screen, self.camera)
        self.right_net.render(self.screen, self.camera)

        # 4. Goalposts (Woodwork)
        self._draw_goalposts()

        # 5. Player drop shadows
        for entity in entities:
            if hasattr(entity, 'role_str'):
                self._draw_player_shadow(entity)

        # 6. Ball shadow based on z elevation
        if ball:
            self._draw_ball_shadow(ball)

        # 7. Ground particles
        self.particle_system.render(self.screen, self.camera)

        # 8. Player entities
        for entity in entities:
            if hasattr(entity, 'role_str'):
                self._draw_player(entity, entity is controlled_player)

        # 9. Ball in 3D (offset vertically by z)
        if ball:
            self._draw_ball_3d(ball)

        # 10. Match HUD with xG
        if match:
            self._draw_hud(match, tactical_name=tactical_name)
            if match.state == "GOAL_SCORED":
                self._draw_goal_banner()
            elif match.state == "OFFSIDE":
                self._draw_offside_banner()
            elif match.state == "FULL_TIME":
                self._draw_full_time_banner(match)

        # 11. Tactical Radar Minimap
        self._draw_radar(entities, ball)

    def _draw_pitch(self):
        self.screen.fill(settings.FIELD_COLOR)
        w, h = settings.SCREEN_WIDTH, settings.SCREEN_HEIGHT

        for x in range(0, w, self.stripe_width * 2):
            p1 = self.camera.world_to_screen(pygame.math.Vector2(x, 0))
            p2 = self.camera.world_to_screen(pygame.math.Vector2(x + self.stripe_width, h))
            scaled_rect = pygame.Rect(p1[0], 0, max(1, p2[0] - p1[0]), self.screen.get_height())
            pygame.draw.rect(self.screen, settings.FIELD_COLOR_DARK, scaled_rect)

        lc = settings.LINE_COLOR

        p_tl = self.camera.world_to_screen(pygame.math.Vector2(0, 0))
        p_br = self.camera.world_to_screen(pygame.math.Vector2(w, h))
        pygame.draw.rect(self.screen, lc, (p_tl[0], p_tl[1], p_br[0] - p_tl[0], p_br[1] - p_tl[1]), max(1, self.camera.scale(3)))

        p_top = self.camera.world_to_screen(pygame.math.Vector2(w // 2, 0))
        p_bot = self.camera.world_to_screen(pygame.math.Vector2(w // 2, h))
        pygame.draw.line(self.screen, lc, p_top, p_bot, max(1, self.camera.scale(2)))

        p_center = self.camera.world_to_screen(pygame.math.Vector2(w // 2, h // 2))
        r_circle = self.camera.scale(settings.CENTER_CIRCLE_RADIUS)
        if r_circle > 0:
            pygame.draw.circle(self.screen, lc, p_center, r_circle, max(1, self.camera.scale(2)))
            pygame.draw.circle(self.screen, lc, p_center, max(1, self.camera.scale(4)))

        pa_w = settings.PENALTY_AREA_WIDTH
        pa_h = settings.PENALTY_AREA_HEIGHT
        pa_top = settings.PENALTY_AREA_TOP

        p_l1 = self.camera.world_to_screen(pygame.math.Vector2(0, pa_top))
        p_l2 = self.camera.world_to_screen(pygame.math.Vector2(pa_w, pa_top + pa_h))
        pygame.draw.rect(self.screen, lc, (p_l1[0], p_l1[1], p_l2[0] - p_l1[0], p_l2[1] - p_l1[1]), max(1, self.camera.scale(2)))

        p_r1 = self.camera.world_to_screen(pygame.math.Vector2(w - pa_w, pa_top))
        p_r2 = self.camera.world_to_screen(pygame.math.Vector2(w, pa_top + pa_h))
        pygame.draw.rect(self.screen, lc, (p_r1[0], p_r1[1], p_r2[0] - p_r1[0], p_r2[1] - p_r1[1]), max(1, self.camera.scale(2)))

    def _draw_goalposts(self):
        w = settings.SCREEN_WIDTH
        gt = settings.GOAL_TOP
        gb = settings.GOAL_BOTTOM
        posts = [(0, gt), (0, gb), (w, gt), (w, gb)]
        r = max(4, self.camera.scale(6))

        for px, py in posts:
            screen_pos = self.camera.world_to_screen(pygame.math.Vector2(px, py))
            # Shadow
            pygame.draw.circle(self.screen, (0, 0, 0, 100), (screen_pos[0] + 2, screen_pos[1] + 3), r)
            # White post with 3D highlight
            pygame.draw.circle(self.screen, (245, 245, 245), screen_pos, r)
            pygame.draw.circle(self.screen, (50, 50, 50), screen_pos, r, 1)

    def _draw_tactical_overlay(self, match):
        """Renders dynamic VAR offside trap lines and tactical passing networks."""
        h = settings.SCREEN_HEIGHT
        # 1. VAR Offside Line for Team B (defending right)
        offside_b_x = SpatialGraphAnalytics.calculate_offside_line(match.team_b, 1)
        p1 = self.camera.world_to_screen(pygame.math.Vector2(offside_b_x, 0))
        p2 = self.camera.world_to_screen(pygame.math.Vector2(offside_b_x, h))

        line_surf = pygame.Surface((4, self.screen.get_height()), pygame.SRCALPHA)
        line_surf.fill((56, 189, 248, 160))
        self.screen.blit(line_surf, (p1[0] - 2, 0))

        tag_surf = self.tiny_font.render("VAR OFFSIDE LINE", True, (56, 189, 248))
        self.screen.blit(tag_surf, (p1[0] + 4, 90))

        # 2. Passing triangles & Passing safety lanes
        triangles = SpatialGraphAnalytics.get_passing_triangles(match.team_a, max_dist=280.0)
        for pos1, pos2, dist in triangles:
            sp1 = self.camera.world_to_screen(pos1)
            sp2 = self.camera.world_to_screen(pos2)
            pygame.draw.line(self.screen, (56, 189, 248, 55), sp1, sp2, 1)

        # 3. Dynamic Pass Safety Rays from active player to options
        ctrl_p = next((p for p in match.team_a.players if p.is_controlled), None)
        if ctrl_p:
            for tm in match.team_a.players:
                if tm is ctrl_p or tm.role_str == "GK":
                    continue
                dist = ctrl_p.position.distance_to(tm.position)
                if dist < 420.0:
                    # Assess passing lane interception risk
                    is_blocked = False
                    is_contested = False
                    lane_vec = tm.position - ctrl_p.position
                    lane_len = lane_vec.length()
                    lane_dir = lane_vec.normalize()

                    for opp in match.team_b.players:
                        to_opp = opp.position - ctrl_p.position
                        proj = to_opp.dot(lane_dir)
                        if 15.0 < proj < lane_len - 15.0:
                            perp_dist = (to_opp - lane_dir * proj).length()
                            if perp_dist < 26.0:
                                is_blocked = True
                                break
                            elif perp_dist < 52.0:
                                is_contested = True

                    lane_color = (239, 68, 68, 140) if is_blocked else ((245, 158, 11, 150) if is_contested else (34, 197, 94, 160))
                    sp1 = self.camera.world_to_screen(ctrl_p.position)
                    sp2 = self.camera.world_to_screen(tm.position)
                    pygame.draw.line(self.screen, lane_color[:3], sp1, sp2, 2 if not is_blocked else 1)

    def _draw_tacticai_overlay(self, match, controlled_player, ball):
        """
        Renders Google DeepMind TacticAI Overlay (Nature Communications 2024):
        - Halo over top predicted receivers with receiver probability percentage
        - Recommended defensive adjustments (What-If analysis) to block passing channels
        - Real-time shot threat indicator
        """
        from ai.tactic_ai import tactic_ai_engine, build_tacticai_graph
        if not tactic_ai_engine:
            return

        team_a = match.team_a.players
        team_b = match.team_b.players
        node_feats, edge_attr = build_tacticai_graph(team_a, team_b, ball)

        try:
            import torch
            with torch.no_grad():
                tactic_ai_engine.eval()
                probs = tactic_ai_engine.predict_receivers(node_feats, edge_attr)[0].cpu().numpy()
                shot_prob, _ = tactic_ai_engine.predict_shot_probability(node_feats, edge_attr)
                shot_p = float(shot_prob.item())

            # 1. Top 3 receivers for Attacking Team (Team A)
            top_rec_indices = np.argsort(probs[:11])[::-1][:3]
            for rank, idx in enumerate(top_rec_indices):
                target_p = team_a[idx]
                if target_p is controlled_player:
                    continue
                prob_pct = int(round(probs[idx] * 100))
                if prob_pct < 4:
                    continue

                sp = self.camera.world_to_screen(target_p.position)
                # Halo color: Gold for #1, Cyan for #2/#3
                halo_color = (245, 158, 11) if rank == 0 else (56, 189, 248)
                halo_r = self.camera.scale(target_p.radius + 12 + rank * 3)
                pygame.draw.circle(self.screen, halo_color, sp, max(1, halo_r), max(1, self.camera.scale(2)))

                # Badge text
                badge_surf = self.tiny_font.render(f"TacticAI #{rank+1} ({prob_pct}%)", True, (255, 255, 255), (15, 23, 42))
                self.screen.blit(badge_surf, (sp[0] - badge_surf.get_width() // 2, sp[1] - halo_r - 14))

            # 2. Defensive Adjustments ("What-If" Analysis)
            adjustments = tactic_ai_engine.recommend_defensive_adjustments(team_a, team_b, ball)
            for adj in adjustments[:4]:
                orig_sp = self.camera.world_to_screen((adj["current_x"], adj["current_y"]))
                sugg_sp = self.camera.world_to_screen((adj["suggested_x"], adj["suggested_y"]))

                # Draw vector line to recommended containment spot
                pygame.draw.line(self.screen, (249, 115, 22), orig_sp, sugg_sp, max(1, self.camera.scale(2)))
                # Ghost containment ring
                ghost_r = self.camera.scale(settings.PLAYER_RADIUS)
                pygame.draw.circle(self.screen, (249, 115, 22), sugg_sp, max(1, ghost_r), max(1, self.camera.scale(1)))

            # 3. TacticAI Status Pill
            pill_surf = self.small_font.render(f"TacticAI | Predicted Shot Threat: {shot_p*100:.1f}%", True, (255, 255, 255))
            pill_rect = pill_surf.get_rect(topleft=(settings.SCREEN_WIDTH - 380, 68))
            bg_rect = pill_rect.inflate(16, 8)
            pygame.draw.rect(self.screen, (15, 23, 42), bg_rect, border_radius=6)
            pygame.draw.rect(self.screen, (56, 189, 248), bg_rect, 1, border_radius=6)
            self.screen.blit(pill_surf, pill_rect)
        except Exception:
            pass

    def _draw_player_shadow(self, player):
        center = self.camera.world_to_screen(player.position + pygame.math.Vector2(3, 5))
        rx = self.camera.scale(player.radius + 2)
        ry = self.camera.scale((player.radius + 2) * 0.5)
        if rx > 0 and ry > 0:
            surf = pygame.Surface((rx * 2, ry * 2), pygame.SRCALPHA)
            pygame.draw.ellipse(surf, (0, 0, 0, 90), (0, 0, rx * 2, ry * 2))
            self.screen.blit(surf, (center[0] - rx, center[1] - ry))

    def _draw_ball_shadow(self, ball):
        center = self.camera.world_to_screen(ball.position + pygame.math.Vector2(2, 4))
        shadow_scale = max(0.35, 1.0 - (ball.z / 260.0))
        rx = self.camera.scale(ball.radius * shadow_scale * 1.2)
        ry = self.camera.scale(ball.radius * shadow_scale * 0.6)
        alpha = int(130 * shadow_scale)
        if rx > 0 and ry > 0:
            surf = pygame.Surface((rx * 2, ry * 2), pygame.SRCALPHA)
            pygame.draw.ellipse(surf, (0, 0, 0, alpha), (0, 0, rx * 2, ry * 2))
            self.screen.blit(surf, (center[0] - rx, center[1] - ry))

    def _draw_player(self, player, is_controlled):
        center = self.camera.world_to_screen(player.position)
        r = self.camera.scale(player.radius)
        if r <= 0:
            return

        # Defensive Jockey containment aura
        if getattr(player, "is_jockeying", False):
            pygame.draw.circle(self.screen, (56, 189, 248, 120), center, r + self.camera.scale(10), max(1, self.camera.scale(2)))

        pygame.draw.circle(self.screen, player.color, center, r)
        pygame.draw.circle(self.screen, (20, 20, 20), center, r, max(1, self.camera.scale(2)))

        face_end = self.camera.world_to_screen(player.position + player.facing * (player.radius + 5))
        pygame.draw.line(self.screen, (255, 255, 255), center, face_end, max(1, self.camera.scale(2)))

        role_label = player.role_str[:2] if player.role_str else ""
        text_surf = self.tiny_font.render(role_label, True, (255, 255, 255))
        self.screen.blit(text_surf, text_surf.get_rect(center=center))

        if is_controlled:
            pygame.draw.circle(self.screen, settings.CONTROLLED_HIGHLIGHT, center, r + self.camera.scale(5), max(1, self.camera.scale(3)))

            marker_tip = (center[0], center[1] - r - self.camera.scale(12))
            marker_l = (center[0] - self.camera.scale(6), center[1] - r - self.camera.scale(22))
            marker_r = (center[0] + self.camera.scale(6), center[1] - r - self.camera.scale(22))
            pygame.draw.polygon(self.screen, (255, 220, 0), [marker_tip, marker_l, marker_r])

            bar_w = self.camera.scale(36)
            bar_h = self.camera.scale(4)
            bar_x = center[0] - bar_w // 2
            bar_y = center[1] - r - self.camera.scale(28)
            stamina_pct = player.profile.current_stamina / 100.0
            stamina_color = (34, 197, 94) if stamina_pct > 0.5 else ((234, 179, 8) if stamina_pct > 0.25 else (239, 68, 68))
            pygame.draw.rect(self.screen, (15, 23, 42), (bar_x - 1, bar_y - 1, bar_w + 2, bar_h + 2))
            pygame.draw.rect(self.screen, stamina_color, (bar_x, bar_y, int(bar_w * stamina_pct), bar_h))

            # Overhead Shot Power Meter
            shot_charge = getattr(player, "shot_charge", 0.0)
            if shot_charge > 0.02:
                pw_w = self.camera.scale(44)
                pw_h = self.camera.scale(6)
                pw_x = center[0] - pw_w // 2
                pw_y = bar_y - pw_h - self.camera.scale(5)
                # Power bar color: Green (safe) -> Yellow (sweet spot) -> Red (skied)
                pw_col = (34, 197, 94) if shot_charge < 0.45 else ((250, 204, 21) if shot_charge <= 0.88 else (239, 68, 68))
                pygame.draw.rect(self.screen, (15, 23, 42), (pw_x - 1, pw_y - 1, pw_w + 2, pw_h + 2))
                pygame.draw.rect(self.screen, pw_col, (pw_x, pw_y, int(pw_w * shot_charge), pw_h))
                pygame.draw.rect(self.screen, (255, 255, 255), (pw_x - 1, pw_y - 1, pw_w + 2, pw_h + 2), 1)

            # Defensive Jockeying Tag
            if getattr(player, "is_jockeying", False):
                jockey_tag = self.tiny_font.render("JOCKEY", True, (56, 189, 248))
                self.screen.blit(jockey_tag, jockey_tag.get_rect(center=(center[0], center[1] + r + 10)))

    def _draw_ball_3d(self, ball):
        elevated_pos = pygame.math.Vector2(ball.position.x, ball.position.y - ball.z)
        center = self.camera.world_to_screen(elevated_pos)
        r = self.camera.scale(ball.radius)
        if r > 0:
            pygame.draw.circle(self.screen, ball.color, center, r)
            pygame.draw.circle(self.screen, (30, 30, 30), center, r, max(1, self.camera.scale(1)))
            if abs(ball.spin) > 0.1:
                arc_r = max(1, r - 2)
                pygame.draw.arc(self.screen, (255, 50, 50), (center[0]-arc_r, center[1]-arc_r, arc_r*2, arc_r*2), 0, math.pi, 2)

    def _draw_hud(self, match, tactical_name="Balanced"):
        w = settings.SCREEN_WIDTH
        bar_w = 460
        bar_surf = pygame.Surface((bar_w, 82), pygame.SRCALPHA)
        bar_surf.fill((15, 23, 42, 235))
        self.screen.blit(bar_surf, (w // 2 - bar_w // 2, 12))
        pygame.draw.rect(self.screen, (51, 65, 85), (w // 2 - bar_w // 2, 12, bar_w, 82), 1, border_radius=8)

        # Team names & xG
        xg_a = match.cumulative_xg.get(0, 0.0)
        xg_b = match.cumulative_xg.get(1, 0.0)

        a_str = f"YOU ({xg_a:.2f} xG)"
        b_str = f"({xg_b:.2f} xG) RL AI"

        a_lbl = self.small_font.render(a_str, True, settings.TEAM_A_COLOR)
        b_lbl = self.small_font.render(b_str, True, settings.TEAM_B_COLOR)
        self.screen.blit(a_lbl, (w // 2 - bar_w // 2 + 16, 20))
        self.screen.blit(b_lbl, (w // 2 + bar_w // 2 - b_lbl.get_width() - 16, 20))

        # Score
        score_str = f"{match.score[0]}  -  {match.score[1]}"
        score_surf = self.score_font.render(score_str, True, (255, 255, 255))
        self.screen.blit(score_surf, score_surf.get_rect(center=(w // 2, 34)))

        # Clock & Tactical Badge
        mins = int(match.time_elapsed // 60)
        secs = int(match.time_elapsed % 60)
        clock_str = f"{mins:02d}:{secs:02d}" if match.state != "FULL_TIME" else "FULL TIME"
        clock_surf = self.small_font.render(clock_str, True, (226, 232, 240))
        self.screen.blit(clock_surf, clock_surf.get_rect(center=(w // 2, 66)))

        # Tactical mentality badge
        tac_surf = self.tiny_font.render(f"TACTIC: {tactical_name.upper()} [1-5 to change | T overlay]", True, (56, 189, 248))
        self.screen.blit(tac_surf, (w // 2 - bar_w // 2 + 16, 62))

    def _draw_goal_banner(self):
        w, h = settings.SCREEN_WIDTH, settings.SCREEN_HEIGHT
        banner_surf = pygame.Surface((480, 90), pygame.SRCALPHA)
        banner_surf.fill((15, 23, 42, 245))
        self.screen.blit(banner_surf, (w // 2 - 240, h // 2 - 120))
        pygame.draw.rect(self.screen, (250, 204, 21), (w // 2 - 240, h // 2 - 120, 480, 90), 3, border_radius=8)

        text = self.banner_font.render("⚽ GOAL! ⚽", True, (250, 204, 21))
        self.screen.blit(text, text.get_rect(center=(w // 2, h // 2 - 75)))

    def _draw_offside_banner(self):
        w, h = settings.SCREEN_WIDTH, settings.SCREEN_HEIGHT
        banner_surf = pygame.Surface((520, 85), pygame.SRCALPHA)
        banner_surf.fill((15, 23, 42, 245))
        self.screen.blit(banner_surf, (w // 2 - 260, h // 2 - 110))
        pygame.draw.rect(self.screen, (239, 68, 68), (w // 2 - 260, h // 2 - 110, 520, 85), 3, border_radius=8)

        t1 = self.font.render("🚩 OFFSIDE (VAR REVIEW)", True, (239, 68, 68))
        t2 = self.small_font.render("Free-kick awarded to defending team", True, (226, 232, 240))
        self.screen.blit(t1, t1.get_rect(center=(w // 2, h // 2 - 82)))
        self.screen.blit(t2, t2.get_rect(center=(w // 2, h // 2 - 50)))

    def _draw_full_time_banner(self, match):
        w, h = settings.SCREEN_WIDTH, settings.SCREEN_HEIGHT
        box_surf = pygame.Surface((600, 180), pygame.SRCALPHA)
        box_surf.fill((15, 23, 42, 245))
        self.screen.blit(box_surf, (w // 2 - 300, h // 2 - 90))
        pygame.draw.rect(self.screen, (56, 189, 248), (w // 2 - 300, h // 2 - 90, 600, 180), 2, border_radius=8)

        result_str = match.get_result_string()
        if result_str == "WIN":
            header = "🏆 FULL TIME — VICTORY!"
            color = (34, 197, 94)
        elif result_str == "LOSS":
            header = "FULL TIME — DEFEAT"
            color = (239, 68, 68)
        else:
            header = "FULL TIME — DRAW"
            color = (250, 204, 21)

        t1 = self.font.render(header, True, color)
        score_summary = f"Score: YOU {match.score[0]} ({match.cumulative_xg[0]:.2f} xG)  -  {match.score[1]} ({match.cumulative_xg[1]:.2f} xG) RL AI"
        t2 = self.small_font.render(score_summary, True, (255, 255, 255))
        t3 = self.small_font.render("AI policy updated with match experience. Press [ESC] for Menu.", True, (148, 163, 184))

        self.screen.blit(t1, t1.get_rect(center=(w // 2, h // 2 - 50)))
        self.screen.blit(t2, t2.get_rect(center=(w // 2, h // 2 - 10)))
        self.screen.blit(t3, t3.get_rect(center=(w // 2, h // 2 + 35)))

    def _draw_radar(self, entities, ball):
        sw, sh = settings.SCREEN_WIDTH, settings.SCREEN_HEIGHT
        rw, rh = 180, 100
        rx = sw - rw - 16
        ry = sh - rh - 16

        radar_surf = pygame.Surface((rw, rh), pygame.SRCALPHA)
        radar_surf.fill((15, 23, 42, 190))
        self.screen.blit(radar_surf, (rx, ry))
        pygame.draw.rect(self.screen, (71, 85, 105), (rx, ry, rw, rh), 1, border_radius=4)
        pygame.draw.line(self.screen, (71, 85, 105), (rx + rw // 2, ry), (rx + rw // 2, ry + rh), 1)

        for e in entities:
            if hasattr(e, 'role_str'):
                bx = rx + int((e.position.x / sw) * rw)
                by = ry + int((e.position.y / sh) * rh)
                col = settings.TEAM_A_COLOR if e.team_id == 0 else settings.TEAM_B_COLOR
                pygame.draw.circle(self.screen, col, (bx, by), 3)

        if ball:
            bx = rx + int((ball.position.x / sw) * rw)
            by = ry + int((ball.position.y / sh) * rh)
            pygame.draw.circle(self.screen, (255, 255, 255), (bx, by), 3)
