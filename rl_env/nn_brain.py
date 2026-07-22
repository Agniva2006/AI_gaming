import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from engine import settings

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from rl_env.gnn_encoder import SpatialGNNEncoder
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class FootballActorCritic(nn.Module):
        """
        Hybrid GNN-PPO Actor-Critic Neural Policy for RL Train Football.
        Combines Graph Attention Network (GAT) spatial graph encoding
        with PPO Policy & Value heads. Supports CUDA GPU acceleration.
        """
        def __init__(self, obs_dim=95, num_actions=12):
            super().__init__()
            self.obs_dim = obs_dim
            self.num_actions = num_actions

            # GNN Spatial Graph Encoder
            self.gnn_encoder = SpatialGNNEncoder(node_in_dim=8, hidden_dim=256)

            # Flat Feature Extractor
            self.fc1 = nn.Linear(obs_dim, 256)
            self.norm1 = nn.LayerNorm(256)

            # Combined Representation Layer
            self.fc2 = nn.Linear(256 + 256, 256)
            self.norm2 = nn.LayerNorm(256)

            # Actor Head (Policy)
            self.actor = nn.Linear(256, num_actions)

            # Critic Head (Value Function)
            self.critic = nn.Linear(256, 1)

            # Device selection (CUDA GPU if available)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.to(self.device)

        def forward(self, x):
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float().to(self.device)
            if x.ndim == 1:
                x = x.unsqueeze(0)

            B = x.shape[0]

            # Construct dummy 23 node features & 2D positions for GNN
            node_feats = torch.zeros(B, 23, 8, device=self.device)
            pos_coords = torch.zeros(B, 23, 2, device=self.device)

            # Populate positions from observation array (22 players + ball)
            for i in range(22):
                pos_coords[:, i, 0] = x[:, i * 4] * settings.SCREEN_WIDTH
                pos_coords[:, i, 1] = x[:, i * 4 + 1] * settings.SCREEN_HEIGHT
                node_feats[:, i, 0] = x[:, i * 4]
                node_feats[:, i, 1] = x[:, i * 4 + 1]
                node_feats[:, i, 2] = x[:, i * 4 + 2]
                node_feats[:, i, 3] = x[:, i * 4 + 3]

            # Ball features
            pos_coords[:, 22, 0] = x[:, 88] * settings.SCREEN_WIDTH
            pos_coords[:, 22, 1] = x[:, 89] * settings.SCREEN_HEIGHT
            node_feats[:, 22, 0] = x[:, 88]
            node_feats[:, 22, 1] = x[:, 89]

            # GNN Spatial Graph Encoding
            graph_embed, attn_weights = self.gnn_encoder(node_feats, pos_coords)

            # Flat feature encoding
            h_flat = F.relu(self.norm1(self.fc1(x)))

            # Combine GNN graph embedding + flat features
            combined = torch.cat([graph_embed, h_flat], dim=-1)
            h = F.relu(self.norm2(self.fc2(combined)))

            logits = self.actor(h)
            value = self.critic(h)

            return logits, value

        def predict_action(self, obs, deterministic=False):
            """Inference method for autonomous live gameplay."""
            self.eval()
            with torch.no_grad():
                logits, _ = self.forward(obs)
                probs = F.softmax(logits, dim=-1)
                if deterministic:
                    action = torch.argmax(probs, dim=-1).item()
                else:
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample().item()
            return action

        def save_weights(self, path):
            torch.save(self.state_dict(), path)

        def load_weights(self, path):
            if os.path.exists(path):
                self.load_state_dict(torch.load(path, map_location=self.device))


class FallbackNumPyBrain:
    def __init__(self, obs_dim=95, num_actions=12):
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        np.random.seed(42)
        self.w1 = np.random.randn(obs_dim, 128) * 0.1
        self.w2 = np.random.randn(128, num_actions) * 0.1

    def predict_action(self, obs, deterministic=False):
        if obs.ndim == 1:
            obs = obs[np.newaxis, :]
        h = np.maximum(0, np.dot(obs, self.w1))
        logits = np.dot(h, self.w2)
        exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)[0]
        if deterministic:
            return int(np.argmax(probs))
        return int(np.random.choice(len(probs), p=probs))


def create_neural_brain(obs_dim=95, num_actions=12):
    if TORCH_AVAILABLE:
        try:
            return FootballActorCritic(obs_dim, num_actions)
        except Exception:
            pass
    return FallbackNumPyBrain(obs_dim, num_actions)
