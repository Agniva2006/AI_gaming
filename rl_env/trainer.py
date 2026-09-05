import os
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from rl_env.football_env import FootballEnv
    from rl_env.nn_brain import FootballActorCritic
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class PPOTrainer:
    """
    PPO Policy Optimization Trainer:
    Trains GNN Actor-Critic policy from live Human vs AI match trajectories
    or standalone background simulated environments.
    """
    def __init__(self, checkpoint_dir=None):
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.brain = FootballActorCritic() if TORCH_AVAILABLE else None
        
        # Load existing weights if available
        self.ckpt_file = os.path.join(self.checkpoint_dir, "ppo_gnn_model.pt")
        if self.brain and os.path.exists(self.ckpt_file):
            try:
                self.brain.load_weights(self.ckpt_file)
            except Exception:
                pass

        if TORCH_AVAILABLE and self.brain:
            self.optimizer = optim.Adam(self.brain.parameters(), lr=3e-4)
        else:
            self.optimizer = None

        self.env = None

        # PPO Hyperparameters
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_eps = 0.2
        self.ppo_epochs = 3
        self.episode_counter = 0

        self.recent_rewards = []
        self.recent_losses = []

    def train_on_match_buffer(self, trajectory_buffer):
        """
        Performs PPO policy update using real gameplay transitions gathered during a match.
        trajectory_buffer: list of (obs, action, reward, next_obs, done)
        """
        if not TORCH_AVAILABLE or not self.brain or not trajectory_buffer or len(trajectory_buffer) < 8:
            return {"reward": 0.0, "loss": 0.0, "actor_loss": 0.0, "critic_loss": 0.0}

        self.episode_counter += 1
        total_reward = sum(t[2] for t in trajectory_buffer)
        avg_reward = total_reward / len(trajectory_buffer)

        # 1. Forward pass to get values and log_probs
        self.brain.eval()
        processed_steps = []
        with torch.no_grad():
            for obs, action, reward, next_obs, done in trajectory_buffer:
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.brain.device)
                logits, value = self.brain(obs_t)
                dist = torch.distributions.Categorical(logits=logits)
                log_prob = dist.log_prob(torch.tensor(action, device=self.brain.device)).item()
                processed_steps.append((obs, action, reward, next_obs, done, log_prob, value.item()))

        # 2. Compute Generalized Advantage Estimation (GAE)
        returns = []
        advantages = []
        gae = 0.0
        for step in reversed(range(len(processed_steps))):
            obs, action, reward, next_obs, done, log_prob, value = processed_steps[step]
            with torch.no_grad():
                next_obs_t = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(self.brain.device)
                _, next_value = self.brain(next_obs_t)
                next_val = next_value.item()

            mask = 0.0 if done else 1.0
            delta = reward + self.gamma * next_val * mask - value
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + value)

        # 3. PPO Optimization
        self.brain.train()
        b_obs = torch.from_numpy(np.array([t[0] for t in processed_steps], dtype=np.float32)).to(self.brain.device)
        b_actions = torch.tensor([t[1] for t in processed_steps], dtype=torch.long, device=self.brain.device)
        b_old_log_probs = torch.tensor([t[5] for t in processed_steps], dtype=torch.float32, device=self.brain.device)
        b_returns = torch.tensor(returns, dtype=torch.float32, device=self.brain.device)
        b_advantages = torch.tensor(advantages, dtype=torch.float32, device=self.brain.device)

        if len(b_advantages) > 1:
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        total_loss_val = 0.0
        act_loss_val = 0.0
        crit_loss_val = 0.0

        for _ in range(self.ppo_epochs):
            logits, values = self.brain(b_obs)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(b_actions)

            ratio = torch.exp(new_log_probs - b_old_log_probs)
            surr1 = ratio * b_advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * (b_returns - values.squeeze(-1)).pow(2).mean()
            entropy_bonus = 0.01 * dist.entropy().mean()

            loss = actor_loss + critic_loss - entropy_bonus

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.brain.parameters(), 0.5)
            self.optimizer.step()

            total_loss_val += loss.item()
            act_loss_val += actor_loss.item()
            crit_loss_val += critic_loss.item()

        avg_loss = total_loss_val / self.ppo_epochs
        self.recent_rewards.append(avg_reward)
        self.recent_losses.append(avg_loss)
        if len(self.recent_rewards) > 50:
            self.recent_rewards.pop(0)
            self.recent_losses.pop(0)

        # Save checkpoint to disk
        self.brain.save_weights(self.ckpt_file)

        return {
            "reward": round(float(avg_reward), 4),
            "loss": round(float(avg_loss), 4),
            "actor_loss": round(float(act_loss_val / self.ppo_epochs), 4),
            "critic_loss": round(float(crit_loss_val / self.ppo_epochs), 4),
            "episode": self.episode_counter
        }

    def train_step(self, num_episodes=1):
        """Simulates background rollouts using FootballEnv and optimizes policy."""
        if not TORCH_AVAILABLE or not self.brain:
            return {"reward": 0.0, "loss": 0.0, "episode": 1}

        if self.env is None:
            self.env = FootballEnv(render=False)

        total_reward = 0.0
        trajectory_buffer = []

        for _ in range(num_episodes):
            obs = self.env.reset()
            done = False
            ep_reward = 0.0

            step_cnt = 0
            while not done:
                step_cnt += 1
                if step_cnt % 30 == 0:
                    pygame.event.pump()

                self.brain.eval()
                with torch.no_grad():
                    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.brain.device)
                    logits, _ = self.brain(obs_t)
                    probs = torch.distributions.Categorical(logits=logits)
                    action = probs.sample().item()

                next_obs, reward, done, _ = self.env.step(action)
                trajectory_buffer.append((obs, action, reward, next_obs, done))
                ep_reward += reward
                obs = next_obs

            total_reward += ep_reward

        return self.train_on_match_buffer(trajectory_buffer)

    def close(self):
        if self.env is not None:
            self.env.close()
