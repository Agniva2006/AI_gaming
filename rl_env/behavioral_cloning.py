import math
import os
import numpy as np
import pygame

from engine import settings
from rl_env.football_env import FootballEnv, ACTION_DIRECTIONS

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    from rl_env.nn_brain import FootballActorCritic, create_neural_brain
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ExpertFootballPolicy:
    """
    Rule-Based Expert Tactical Policy for Generating Behavioral Cloning Data.
    Analyzes game state / 95-dim observation vector and produces optimal
    discrete football actions (0-11) for passing, shooting, dribbling, and pressing.
    """
    def __init__(self, target_goal=(settings.SCREEN_WIDTH, settings.SCREEN_HEIGHT // 2)):
        self.target_goal = target_goal
        # Normalized unit direction vectors for actions 1 to 8
        self.dir_vectors = {}
        for act, (dx, dy) in ACTION_DIRECTIONS.items():
            if act == 0:
                continue
            v = pygame.math.Vector2(dx, dy)
            if v.length_squared() > 0:
                self.dir_vectors[act] = v.normalize()

    def _best_direction_action(self, target_vec):
        """Finds the discrete movement action (1..8) that best aligns with target_vec."""
        if target_vec.length_squared() < 1e-4:
            return 0
        norm_v = target_vec.normalize()
        best_act = 4  # default right
        best_dot = -999.0
        for act, dir_v in self.dir_vectors.items():
            dot = norm_v.x * dir_v.x + norm_v.y * dir_v.y
            if dot > best_dot:
                best_dot = dot
                best_act = act
        return best_act

    def decide_action(self, obs, env=None):
        """
        Decides expert action from observation array (95-dim).
        Can use direct environment entities if available for maximum precision.
        """
        w = float(settings.SCREEN_WIDTH)
        h = float(settings.SCREEN_HEIGHT)

        if env and hasattr(env, 'controlled_player') and env.controlled_player:
            player = env.controlled_player
            ball = env.ball
            target_goal = pygame.math.Vector2(self.target_goal)
            dist_to_ball = player.position.distance_to(ball.position)
            has_ball = player.can_kick(ball)
            dist_to_goal = player.position.distance_to(target_goal)

            if has_ball:
                # 1. In scoring range: Shoot!
                if dist_to_goal < settings.AI_SHOOT_DISTANCE:
                    return 10  # Shoot
                
                # 2. Check for an open forward teammate to pass to
                best_tm = None
                best_tm_dist = 9999.0
                for tm in env.team_a.players:
                    if tm is not player and tm.role_str != "GK":
                        # Forward teammate ahead of player
                        if tm.position.x > player.position.x + 40:
                            d = tm.position.distance_to(player.position)
                            if 80 < d < 360 and d < best_tm_dist:
                                best_tm_dist = d
                                best_tm = tm
                
                # If there's an open forward teammate and a defender is pressing closely
                nearest_opp = env.team_b.get_closest_to_ball(ball)
                opp_pressing = nearest_opp and nearest_opp.position.distance_to(player.position) < 90
                if best_tm and opp_pressing:
                    return 9  # Pass

                # 3. Dribble toward opponent goal
                to_goal = target_goal - player.position
                # Avoid running straight into the nearest defender by angling slightly
                if nearest_opp and nearest_opp.position.distance_to(player.position) < 70:
                    to_opp = nearest_opp.position - player.position
                    perp = pygame.math.Vector2(-to_opp.y, to_opp.x).normalize() * 40
                    to_goal += perp

                return self._best_direction_action(to_goal)
            else:
                # Out of possession: Sprint to press / intercept ball
                to_ball = ball.position - player.position
                return self._best_direction_action(to_ball)

        # Fallback decoding directly from 95-dim observation array
        # Controlled player index is stored at obs[94]
        player_idx = min(21, int(round(obs[94] * 22.0)))
        px = obs[player_idx * 4] * w
        py = obs[player_idx * 4 + 1] * h
        bx = obs[88] * w
        by = obs[89] * h

        dist_ball = math.hypot(bx - px, by - py)
        dist_goal = math.hypot(w - px, (h / 2.0) - py)

        if dist_ball < 30.0:
            if dist_goal < 320.0:
                return 10  # Shoot
            if px < w * 0.70 and (py < h * 0.25 or py > h * 0.75):
                return 9  # Pass from wide areas into center
            return self._best_direction_action(pygame.math.Vector2(w - px, (h / 2.0) - py))
        else:
            return self._best_direction_action(pygame.math.Vector2(bx - px, by - py))


def generate_expert_dataset(num_episodes=15, max_steps=250):
    """
    Runs headless simulation in FootballEnv and collects expert demonstrations.
    Returns:
        observations: np.ndarray (N, 95)
        actions: np.ndarray (N,)
        values: np.ndarray (N,)
    """
    env = FootballEnv(render=False)
    expert = ExpertFootballPolicy()

    all_obs = []
    all_actions = []
    all_rewards = []
    all_dones = []

    print(f"  -> Collecting expert demonstrations across {num_episodes} simulated episodes...")

    for ep in range(num_episodes):
        obs = env.reset()
        ep_obs = []
        ep_actions = []
        ep_rewards = []
        done = False
        step = 0

        while not done and step < max_steps:
            step += 1
            if step % 50 == 0:
                pygame.event.pump()

            expert_action = expert.decide_action(obs, env=env)
            next_obs, reward, done, _ = env.step(expert_action)

            ep_obs.append(obs)
            ep_actions.append(expert_action)
            ep_rewards.append(reward)

            obs = next_obs

        # Compute discounted returns for Critic value supervision
        gamma = 0.99
        returns = []
        discounted = 0.0
        for r in reversed(ep_rewards):
            discounted = r + gamma * discounted
            returns.insert(0, discounted)

        all_obs.extend(ep_obs)
        all_actions.extend(ep_actions)
        all_rewards.extend(returns)

    env.close()

    obs_arr = np.array(all_obs, dtype=np.float32)
    act_arr = np.array(all_actions, dtype=np.int64)
    val_arr = np.array(all_rewards, dtype=np.float32)

    print(f"  -> Successfully generated {len(obs_arr)} expert state-action demonstration pairs.")
    return obs_arr, act_arr, val_arr


class BCTrainer:
    """
    Supervised Behavioral Cloning Trainer:
    Optimizes the GNN Actor-Critic policy from expert demonstrations using Cross-Entropy
    classification loss on action logits and MSE regression loss on the critic value head.
    """
    def __init__(self, checkpoint_dir=None):
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.brain = FootballActorCritic() if TORCH_AVAILABLE else None

    def train(self, obs_data, action_data, value_data, epochs=5, batch_size=64, lr=1e-3):
        """
        Executes supervised imitation learning optimization loop.
        """
        if not TORCH_AVAILABLE or not self.brain:
            return {"status": "error", "message": "PyTorch not available", "accuracy": 0.0}

        device = self.brain.device
        self.brain.train()

        # Normalize value targets for stable regression
        v_mean = float(value_data.mean())
        v_std = float(value_data.std() + 1e-6)
        norm_values = (value_data - v_mean) / v_std

        dataset = TensorDataset(
            torch.from_numpy(obs_data).float(),
            torch.from_numpy(action_data).long(),
            torch.from_numpy(norm_values).float()
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.brain.parameters(), lr=lr, weight_decay=1e-4)
        criterion_actor = nn.CrossEntropyLoss()
        criterion_critic = nn.MSELoss()

        history = []
        print(f"  -> Commencing Supervised Behavioral Cloning for {epochs} epochs...")

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            epoch_act_loss = 0.0
            epoch_val_loss = 0.0
            correct = 0
            total = 0

            for b_obs, b_act, b_val in dataloader:
                b_obs = b_obs.to(device)
                b_act = b_act.to(device)
                b_val = b_val.to(device)

                logits, values = self.brain(b_obs)

                actor_loss = criterion_actor(logits, b_act)
                critic_loss = criterion_critic(values.squeeze(-1), b_val)
                total_loss = actor_loss + 0.5 * critic_loss

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.brain.parameters(), 0.5)
                optimizer.step()

                epoch_loss += total_loss.item() * len(b_act)
                epoch_act_loss += actor_loss.item() * len(b_act)
                epoch_val_loss += critic_loss.item() * len(b_act)

                preds = torch.argmax(logits, dim=-1)
                correct += (preds == b_act).sum().item()
                total += len(b_act)

            avg_loss = epoch_loss / total
            avg_act = epoch_act_loss / total
            avg_val = epoch_val_loss / total
            acc = (correct / total) * 100.0

            history.append({
                "epoch": epoch,
                "loss": round(avg_loss, 4),
                "actor_loss": round(avg_act, 4),
                "critic_loss": round(avg_val, 4),
                "accuracy": round(acc, 2)
            })

            print(f"    [Epoch {epoch}/{epochs}] Loss: {avg_loss:.4f} | Actor Loss: {avg_act:.4f} | Critic Loss: {avg_val:.4f} | Match Accuracy: {acc:.1f}%")

        # Save pre-trained checkpoint to disk
        target_ckpt = os.path.join(self.checkpoint_dir, "ppo_gnn_model.pt")
        backup_ckpt = os.path.join(self.checkpoint_dir, "bc_pretrained_expert.pt")
        self.brain.save_weights(target_ckpt)
        self.brain.save_weights(backup_ckpt)
        print(f"  -> Saved Behavioral Cloning weights to {target_ckpt} and {backup_ckpt}")

        return {
            "status": "success",
            "samples": len(obs_data),
            "epochs": epochs,
            "final_loss": history[-1]["loss"],
            "accuracy": history[-1]["accuracy"],
            "history": history
        }

