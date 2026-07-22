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
    class TacticalDiffusionGenerator(nn.Module):
        """
        Lightweight score-based Diffusion model for synthesizing diverse 
        tactical initial scenario states (counter-attacks, wing overloads, low-block traps).
        """
        def __init__(self, state_dim=46, time_steps=20):
            super().__init__()
            self.state_dim = state_dim
            self.time_steps = time_steps

            self.time_embed = nn.Sequential(
                nn.Linear(1, 32),
                nn.SiLU(),
                nn.Linear(32, 32)
            )

            self.net = nn.Sequential(
                nn.Linear(state_dim + 32, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, state_dim)
            )

        def forward(self, x, t):
            t_emb = self.time_embed(t.unsqueeze(-1))
            h = torch.cat([x, t_emb], dim=-1)
            return self.net(h)

        def sample_scenario(self, num_samples=1):
            """
            Reverse diffusion sampling process to generate realistic tactical initial positions.
            """
            self.eval()
            with torch.no_grad():
                x = torch.randn(num_samples, self.state_dim)
                for step in reversed(range(self.time_steps)):
                    t = torch.tensor([[float(step) / self.time_steps]])
                    predicted_noise = self.forward(x, t)
                    x = x - 0.1 * predicted_noise + (0.02 * torch.randn_like(x) if step > 0 else 0)
            return x.numpy()
