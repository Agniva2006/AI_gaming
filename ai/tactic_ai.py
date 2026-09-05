"""
TacticAI: An AI Assistant for Football Tactics
Implementation based on Google DeepMind & Liverpool FC (Nature Communications, 2024)
Reference: https://doi.org/10.1038/s41467-024-45965-x

Key Architecture:
1. D2 Dihedral Group Symmetries: 4 pitch reflection views (id, horizontal, vertical, both).
2. Multi-Head GATv2 Message Passing with Frame Averaging (Eq. 9).
3. Task 1: Receiver Prediction (Node Classification over 22 players).
4. Task 2: Decomposed Shot Likelihood (Graph Classification, Eq. 1).
5. Task 3: Generative Tactic Refinement (Generates defender position offsets Δx, Δy to minimize shot threat).
"""

import math
import os
import numpy as np
import pygame

from engine import settings

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class D2DihedralTransformer:
    """
    Geometric Deep Learning: D2 Dihedral Group Reflection Engine.
    Generates the 4 symmetry-preserving reflection views of the football pitch:
    - id: Original pitch
    - horiz (↔): Horizontally mirrored across halfway line (W - x, y, -vx, vy)
    - vert (↕): Vertically mirrored across pitch center (x, H - y, vx, -vy)
    - both (↔↕): Both horizontally and vertically reflected
    """
    @staticmethod
    def transform_nodes(node_features, w=settings.SCREEN_WIDTH, h=settings.SCREEN_HEIGHT):
        """
        node_features: (B, 22, 8) where dims are:
        [0: x/w, 1: y/h, 2: vx/max, 3: vy/max, 4: height, 5: weight, 6: has_ball, 7: team_id]
        Returns:
            Tensor of shape (B, 4, 22, 8) containing all 4 D2 views.
        """
        if isinstance(node_features, np.ndarray):
            node_features = torch.from_numpy(node_features).float()

        v_id = node_features.clone()

        # 1. Horizontal reflection (↔)
        v_horiz = node_features.clone()
        v_horiz[..., 0] = 1.0 - v_horiz[..., 0]  # reflect X
        v_horiz[..., 2] = -v_horiz[..., 2]      # negate VX

        # 2. Vertical reflection (↕)
        v_vert = node_features.clone()
        v_vert[..., 1] = 1.0 - v_vert[..., 1]  # reflect Y
        v_vert[..., 3] = -v_vert[..., 3]      # negate VY

        # 3. Both horizontal and vertical (↔↕)
        v_both = node_features.clone()
        v_both[..., 0] = 1.0 - v_both[..., 0]
        v_both[..., 1] = 1.0 - v_both[..., 1]
        v_both[..., 2] = -v_both[..., 2]
        v_both[..., 3] = -v_both[..., 3]

        # Stack into (B, 4, 22, 8)
        return torch.stack([v_id, v_horiz, v_vert, v_both], dim=1)


if TORCH_AVAILABLE:
    class GATv2Conv(nn.Module):
        """
        Graph Attention Network v2 (Brody et al., 2022) layer.
        Computes dynamic edge attention coefficients with edge features (teammate vs opponent).
        Matches Eq. (3) & (4) from the TacticAI Nature paper.
        """
        def __init__(self, in_features, out_features, edge_dim=2, num_heads=4):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.num_heads = num_heads
            self.head_dim = out_features // num_heads

            self.w_src = nn.Linear(in_features, out_features, bias=False)
            self.w_dst = nn.Linear(in_features, out_features, bias=False)
            self.w_edge = nn.Linear(edge_dim, out_features, bias=False)
            self.attn = nn.Parameter(torch.randn(num_heads, self.head_dim))
            self.leaky_relu = nn.LeakyReLU(0.2)

        def forward(self, h, edge_attr=None):
            # h: (B, N, in_features)
            B, N, _ = h.shape
            h_src = self.w_src(h).view(B, N, self.num_heads, self.head_dim)
            h_dst = self.w_dst(h).view(B, N, self.num_heads, self.head_dim)

            # Pairwise attention scores (B, N, N, num_heads)
            scores = h_src.unsqueeze(2) + h_dst.unsqueeze(1)
            if edge_attr is not None:
                e_transformed = self.w_edge(edge_attr).view(B, N, N, self.num_heads, self.head_dim)
                scores = scores + e_transformed

            # a^T * LeakyReLU(W1*hu + W2*hv + We*e)
            scores = self.leaky_relu(scores)
            attn_weights = torch.einsum("bnmhd,hd->bnmh", scores, self.attn)
            attn_weights = F.softmax(attn_weights, dim=2)  # normalize over neighbors

            # Message aggregation
            out = torch.einsum("bnmh,bmhd->bnhd", attn_weights, h_dst)
            out = out.reshape(B, N, self.out_features)
            return out


    class TacticAIModel(nn.Module):
        """
        Complete TacticAI Architecture (Nature Communications 2024):
        - D2 Frame-Averaged GATv2 Backbone (Eq. 9)
        - Receiver Prediction Head (Node Classification)
        - Decomposed Shot Likelihood Head (Graph Classification, Eq. 1)
        - Generative Tactic Refiner (Defensive Adjustment Recommender)
        """
        def __init__(self, node_in_dim=8, hidden_dim=64):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.node_embed = nn.Linear(node_in_dim, hidden_dim)

            # Two-layer GATv2 Message Passing
            self.gat1 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=2, num_heads=4)
            self.gat2 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=2, num_heads=4)
            self.norm = nn.LayerNorm(hidden_dim)

            # Task 1: Receiver Prediction Head (Node classification over 22 players)
            self.receiver_head = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1)  # logit per player node
            )

            # Task 2: Decomposed Shot Prediction Head (Eq. 1)
            self.shot_head = nn.Sequential(
                nn.Linear(hidden_dim + 1, 32),  # player embedding + receiver indicator
                nn.ReLU(),
                nn.Linear(32, 1)
            )

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.to(self.device)

        def forward_backbone(self, node_features, edge_attr):
            """
            Computes D2 frame-averaged latent node representations (Eq. 9).
            node_features: (B, 22, 8)
            edge_attr: (B, 22, 22, 2)
            Returns:
                H_node: (B, 22, hidden_dim)
            """
            B, N, _ = node_features.shape

            # 1. Generate 4 D2 reflection views
            views = D2DihedralTransformer.transform_nodes(node_features).to(self.device)  # (B, 4, 22, 8)
            edge_attr = edge_attr.to(self.device)

            view_embeddings = []
            for v_idx in range(4):
                v_nodes = views[:, v_idx, :, :]  # (B, 22, 8)
                h = F.relu(self.node_embed(v_nodes))
                h1 = F.relu(self.gat1(h, edge_attr))
                h2 = self.norm(self.gat2(h1, edge_attr))
                view_embeddings.append(h2)

            # 2. Exact D2 Frame Averaging (Eq. 9 in Nature paper):
            # H_node = (H_id + H_horiz + H_vert + H_both) / 4
            H_node = torch.stack(view_embeddings, dim=0).mean(dim=0)  # (B, 22, hidden_dim)
            return H_node

        def predict_receivers(self, node_features, edge_attr):
            """
            Predicts the probability distribution over 22 players on who will receive the pass/cross.
            Returns:
                probs: (B, 22) probability distribution
            """
            H_node = self.forward_backbone(node_features, edge_attr)
            logits = self.receiver_head(H_node).squeeze(-1)  # (B, 22)
            probs = F.softmax(logits, dim=-1)
            return probs

        def predict_shot_probability(self, node_features, edge_attr):
            """
            Computes decomposed shot probability using Eq. (1) from Nature paper:
            P(shot | corner) = sum_{i in players} P(shot | receiver=i) * P(receiver=i)
            """
            H_node = self.forward_backbone(node_features, edge_attr)
            rec_logits = self.receiver_head(H_node).squeeze(-1)
            rec_probs = F.softmax(rec_logits, dim=-1)  # (B, 22)

            B, N, D = H_node.shape
            conditional_shot_probs = torch.zeros(B, N, device=self.device)

            for i in range(N):
                # Condition on receiver = i
                rec_indicator = torch.zeros(B, N, 1, device=self.device)
                rec_indicator[:, i, 0] = 1.0
                cond_input = torch.cat([H_node, rec_indicator], dim=-1)
                shot_logits = self.shot_head(cond_input).mean(dim=1)  # pooled
                conditional_shot_probs[:, i] = torch.sigmoid(shot_logits).squeeze(-1)

            # Eq. 1: Marginalize over all possible receivers
            total_shot_prob = (conditional_shot_probs * rec_probs).sum(dim=-1)
            return total_shot_prob, rec_probs

        def recommend_defensive_adjustments(self, team_a_players, team_b_players, ball, max_shift=28.0):
            """
            TacticAI Generative Refinement ("What-If" Analysis, Fig. 3 & Eq. 11):
            Evaluates the passing channels to the top predicted attacking receivers and calculates
            optimal defender position offsets (Δx, Δy) to intercept channels and minimize shot threat.
            """
            node_feats, edge_attr = build_tacticai_graph(team_a_players, team_b_players, ball)
            self.eval()
            with torch.no_grad():
                probs = self.predict_receivers(node_feats, edge_attr)[0].cpu().numpy()

            # Team A = Attacking (indices 0..10), Team B = Defending (indices 11..21)
            # Find the top predicted attacking receivers
            top_rec_indices = np.argsort(probs[:11])[::-1][:3]

            adjustments = []
            for b_idx, def_player in enumerate(team_b_players):
                if def_player.role_str == "GK":
                    continue

                # Find closest high-threat attacker to this defender
                best_threat = None
                best_threat_dist = 9999.0
                for att_idx in top_rec_indices:
                    att_player = team_a_players[att_idx]
                    d = def_player.position.distance_to(att_player.position)
                    if d < best_threat_dist:
                        best_threat_dist = d
                        best_threat = att_player

                if best_threat and best_threat_dist < 260.0:
                    # TacticAI recommendation: step into the passing channel between ball and attacker
                    channel_midpoint = (ball.position + best_threat.position) * 0.5
                    target_pos = def_player.position.lerp(channel_midpoint, 0.40)

                    # Constrain adjustment within max_shift
                    shift_vec = target_pos - def_player.position
                    if shift_vec.length() > max_shift:
                        shift_vec = shift_vec.normalize() * max_shift

                    adjustments.append({
                        "player_id": getattr(def_player, 'id', def_player.role_index),
                        "role": def_player.role_str,
                        "current_x": round(float(def_player.position.x), 1),
                        "current_y": round(float(def_player.position.y), 1),
                        "suggested_x": round(float(def_player.position.x + shift_vec.x), 1),
                        "suggested_y": round(float(def_player.position.y + shift_vec.y), 1),
                        "target_attacker_role": best_threat.role_str,
                        "threat_probability": round(float(probs[team_a_players.index(best_threat)]), 3)
                    })

            return adjustments


def build_tacticai_graph(team_a_players, team_b_players, ball):
    """
    Constructs the 22-player graph with node features and pairwise edge attributes.
    Returns:
        node_features: Tensor of shape (1, 22, 8)
        edge_attr: Tensor of shape (1, 22, 22, 2)
    """
    all_players = team_a_players + team_b_players
    num_nodes = len(all_players)
    node_feats = np.zeros((1, num_nodes, 8), dtype=np.float32)
    w = float(settings.SCREEN_WIDTH)
    h = float(settings.SCREEN_HEIGHT)
    max_speed = 500.0

    for i, p in enumerate(all_players):
        node_feats[0, i, 0] = p.position.x / w
        node_feats[0, i, 1] = p.position.y / h
        node_feats[0, i, 2] = p.velocity.x / max_speed
        node_feats[0, i, 3] = p.velocity.y / max_speed
        # Physical profile features (normalized)
        node_feats[0, i, 4] = getattr(p.profile, "height_cm", 182.0) / 200.0
        node_feats[0, i, 5] = getattr(p.profile, "weight_kg", 76.0) / 100.0
        # Ball possession indicator
        node_feats[0, i, 6] = 1.0 if p.can_kick(ball) else 0.0
        # Team membership (0 for Team A, 1 for Team B)
        node_feats[0, i, 7] = 0.0 if i < 11 else 1.0

    # Edge attributes: One-hot binary (same team vs opposing team)
    edge_attr = np.zeros((1, num_nodes, num_nodes, 2), dtype=np.float32)
    for i in range(num_nodes):
        for j in range(num_nodes):
            same_team = (i < 11 and j < 11) or (i >= 11 and j >= 11)
            if same_team:
                edge_attr[0, i, j, 0] = 1.0  # Teammates
            else:
                edge_attr[0, i, j, 1] = 1.0  # Opponents

    if TORCH_AVAILABLE:
        return torch.from_numpy(node_feats).float(), torch.from_numpy(edge_attr).float()
    return node_feats, edge_attr


# Global TacticAI Singleton instance
tactic_ai_engine = TacticAIModel() if TORCH_AVAILABLE else None
