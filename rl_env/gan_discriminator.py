import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class MotionRealismDiscriminator(nn.Module):
        """
        GAN Discriminator network evaluating player trajectory realism.
        Discriminates realistic fluid human football movement from unnatural 
        robotic RL jitter. Used to shape RL rewards.
        """
        def __init__(self, seq_len=16, feature_dim=6):
            super().__init__()
            self.seq_len = seq_len
            self.feature_dim = feature_dim  # (vx, vy, ax, ay, angular_vel, stamina)

            self.conv1 = nn.Conv1d(feature_dim, 64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
            self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
            self.fc = nn.Linear(64 * seq_len, 1)

        def forward(self, trajectory_seq):
            """
            trajectory_seq: Tensor of shape (B, seq_len, feature_dim) or (B, feature_dim, seq_len)
            """
            if trajectory_seq.dim() == 2:
                trajectory_seq = trajectory_seq.unsqueeze(0)

            # Ensure (B, feature_dim, seq_len) layout
            if trajectory_seq.shape[1] == self.seq_len:
                trajectory_seq = trajectory_seq.transpose(1, 2)

            h = F.leaky_relu(self.conv1(trajectory_seq), 0.2)
            h = F.leaky_relu(self.conv2(h), 0.2)
            h = F.leaky_relu(self.conv3(h), 0.2)
            h = h.view(h.shape[0], -1)

            realism_score = torch.sigmoid(self.fc(h)) # Output in [0, 1]
            return realism_score

        def evaluate_realism_score(self, trajectory_np):
            """
            Inference wrapper returning float realism score in [0, 1].
            """
            self.eval()
            with torch.no_grad():
                tensor_in = torch.from_numpy(trajectory_np).float()
                score = self.forward(tensor_in).item()
            return score
