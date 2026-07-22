import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class GraphAttentionLayer(nn.Module):
        """
        Multi-Head Graph Attention (GAT) layer for spatial passing networks.
        Computes attention coefficients alpha_ij between all player nodes
        to model passing channels and defender marking pressure.
        """
        def __init__(self, in_features, out_features, num_heads=4):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.num_heads = num_heads
            self.head_dim = out_features // num_heads

            self.W = nn.Linear(in_features, out_features, bias=False)
            self.attn_src = nn.Parameter(torch.randn(num_heads, self.head_dim))
            self.attn_dst = nn.Parameter(torch.randn(num_heads, self.head_dim))
            self.leaky_relu = nn.LeakyReLU(0.2)

        def forward(self, h, adj_mask=None):
            # h shape: (batch_size, num_nodes, in_features)
            B, N, _ = h.shape
            h_transformed = self.W(h) # (B, N, num_heads * head_dim)
            h_transformed = h_transformed.view(B, N, self.num_heads, self.head_dim)

            # Compute attention scores
            score_src = torch.einsum("bnhd,hd->bnh", h_transformed, self.attn_src)
            score_dst = torch.einsum("bnhd,hd->bnh", h_transformed, self.attn_dst)

            scores = score_src.unsqueeze(2) + score_dst.unsqueeze(1) # (B, N, N, num_heads)
            scores = self.leaky_relu(scores)

            if adj_mask is not None:
                scores = scores.masked_fill(adj_mask.unsqueeze(-1) == 0, -1e9)

            attn_weights = F.softmax(scores, dim=2) # Normalize over neighbors
            out = torch.einsum("bnmh,bmhd->bnhd", attn_weights, h_transformed)
            out = out.reshape(B, N, self.out_features)

            return out, attn_weights.mean(dim=-1) # Return embeddings & average attention map


    class SpatialGNNEncoder(nn.Module):
        """
        Graph Neural Network spatial encoder for 22 players + ball.
        Processes spatial topology into rich tactical graph embeddings.
        """
        def __init__(self, node_in_dim=8, hidden_dim=256):
            super().__init__()
            self.node_embed = nn.Linear(node_in_dim, 128)
            self.gat1 = GraphAttentionLayer(128, 128, num_heads=4)
            self.gat2 = GraphAttentionLayer(128, hidden_dim, num_heads=4)
            self.out_norm = nn.LayerNorm(hidden_dim)

        def forward(self, node_features, pos_coords):
            """
            node_features: Tensor of shape (B, 23, 8)
            pos_coords: Tensor of shape (B, 23, 2)
            """
            B, N, _ = node_features.shape

            # Compute distance-based spatial adjacency mask (connection if dist < 350px)
            diff = pos_coords.unsqueeze(2) - pos_coords.unsqueeze(1) # (B, N, N, 2)
            dists = torch.norm(diff, dim=-1) # (B, N, N)
            adj_mask = (dists < 350.0).float()

            h1 = F.relu(self.node_embed(node_features))
            h_gat1, attn1 = self.gat1(h1, adj_mask)
            h_gat2, attn2 = self.gat2(F.relu(h_gat1), adj_mask)

            # Global Graph Pooling (Mean + Max Pooling across all 23 nodes)
            graph_mean = torch.mean(h_gat2, dim=1)
            graph_max, _ = torch.max(h_gat2, dim=1)
            graph_embedding = self.out_norm(graph_mean + graph_max)

            return graph_embedding, attn2
