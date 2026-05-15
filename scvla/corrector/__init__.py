"""
Corrector Trainer: RL training of the action correction module.

Uses PPO to train the corrector. The corrector learns to output
residual corrections that improve task success rate.

The key insight: we don't fine-tune the VLA. We only train a tiny
corrector module that adjusts the VLA's actions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Any
from pathlib import Path


class PPOCorrectorTrainer:
    """
    Trains the action corrector using PPO.

    The corrector observes VLA features and outputs a residual correction.
    Reward = 1 if task succeeds, 0 otherwise (dense shaping optional).

    This is intentionally simple. The corrector is tiny (~5K params),
    so PPO converges quickly.
    """

    def __init__(
        self,
        corrector: nn.Module,
        feature_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
        entropy_coeff: float = 0.01,
        device: str = "cpu",
    ):
        self.corrector = corrector.to(device)
        self.device = device
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = entropy_coeff

        self.optimizer = optim.Adam(corrector.parameters(), lr=lr)

        # Exploration noise (decreases during training)
        self.log_std = nn.Parameter(torch.zeros(action_dim, device=device))
        self.std_optimizer = optim.Adam([self.log_std], lr=lr)

    def select_action(
        self,
        features: torch.Tensor,
        vla_action: torch.Tensor,
        explore: bool = True,
    ) -> dict:
        """
        Select a correction action.

        Args:
            features: (1, feature_dim) VLA features
            vla_action: (1, action_dim) original VLA action
            explore: if True, add Gaussian noise for exploration

        Returns:
            dict with: residual, corrected_action, log_prob
        """
        features = features.float().to(self.device)
        vla_action = vla_action.float().to(self.device)

        # Get mean residual from corrector
        mean_residual = self.corrector(features, vla_action)  # (1, action_dim)

        if explore:
            std = torch.exp(self.log_std).expand_as(mean_residual)
            noise = torch.randn_like(mean_residual) * std
            residual = mean_residual + noise
        else:
            residual = mean_residual

        # Compute log probability
        if explore:
            std = torch.exp(self.log_std).expand_as(mean_residual)
            log_prob = -0.5 * ((residual - mean_residual) / std) ** 2 - torch.log(std) - 0.5 * np.log(2 * np.pi)
            log_prob = log_prob.sum(dim=-1)  # (1,)
        else:
            log_prob = torch.zeros(1, device=self.device)

        corrected_action = vla_action + residual

        return {
            "residual": residual.detach(),
            "corrected_action": corrected_action.detach(),
            "log_prob": log_prob.detach(),
            "mean_residual": mean_residual.detach(),
        }

    def compute_returns(self, rewards: list) -> torch.Tensor:
        """Compute discounted returns."""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32, device=self.device)

    def update(
        self,
        features_list: list,
        vla_actions_list: list,
        residuals_list: list,
        log_probs_list: list,
        rewards_list: list,
    ) -> dict:
        """
        PPO update step.

        Args:
            features_list: list of (1, feature_dim) tensors
            vla_actions_list: list of (1, action_dim) tensors
            residuals_list: list of (1, action_dim) tensors
            log_probs_list: list of (1,) tensors
            rewards_list: list of floats

        Returns:
            dict with loss metrics
        """
        # Stack
        features = torch.cat(features_list, dim=0)  # (T, feature_dim)
        vla_actions = torch.cat(vla_actions_list, dim=0)  # (T, action_dim)
        old_residuals = torch.cat(residuals_list, dim=0)  # (T, action_dim)
        old_log_probs = torch.cat(log_probs_list, dim=0)  # (T,)
        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=self.device)

        # Compute returns
        returns = self.compute_returns(rewards_list)
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Re-compute log probs with current policy
        mean_residuals = self.corrector(features, vla_actions)  # (T, action_dim)
        std = torch.exp(self.log_std).expand_as(mean_residuals)
        new_log_probs = -0.5 * ((old_residuals - mean_residuals) / std) ** 2 - torch.log(std) - 0.5 * np.log(2 * np.pi)
        new_log_probs = new_log_probs.sum(dim=-1)  # (T,)

        # PPO ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * returns
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * returns

        # Policy loss
        policy_loss = -torch.min(surr1, surr2).mean()

        # Entropy bonus (encourages exploration)
        entropy = 0.5 * torch.log(2 * np.pi * std ** 2).sum(dim=-1).mean()
        entropy_loss = -self.entropy_coeff * entropy

        # Total loss
        loss = policy_loss + entropy_loss

        self.optimizer.zero_grad()
        self.std_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.corrector.parameters(), 0.5)
        self.optimizer.step()
        self.std_optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "entropy": entropy.item(),
            "mean_ratio": ratio.mean().item(),
        }

    def train_episode(
        self,
        vla: Any,
        env: Any,
        feature_extractor: Any,
        max_steps: int = 300,
        instruction: str = "",
    ) -> dict:
        """
        Run one episode with the corrector and collect data for PPO update.

        Returns:
            dict with episode data (features, actions, rewards, etc.)
        """
        features_list = []
        vla_actions_list = []
        residuals_list = []
        log_probs_list = []
        rewards_list = []

        observation, info = env.reset()

        for t in range(max_steps):
            # Get VLA features
            features = feature_extractor.extract(observation, instruction)

            # Get VLA action
            vla_action = vla.predict_action(observation, instruction)
            if isinstance(vla_action, np.ndarray):
                vla_action = torch.tensor(vla_action, dtype=torch.float32).unsqueeze(0)

            # Get correction
            result = self.select_action(features, vla_action, explore=True)

            # Execute corrected action
            corrected_action = result["corrected_action"].cpu().numpy().squeeze()
            next_obs, reward, terminated, truncated, step_info = env.step(corrected_action)
            done = terminated or truncated

            # Store
            features_list.append(features)
            vla_actions_list.append(vla_action)
            residuals_list.append(result["residual"])
            log_probs_list.append(result["log_prob"])
            rewards_list.append(reward)

            observation = next_obs

            if done:
                break

        success = info.get("success", False) or info.get("is_success", False)

        return {
            "features": features_list,
            "vla_actions": vla_actions_list,
            "residuals": residuals_list,
            "log_probs": log_probs_list,
            "rewards": rewards_list,
            "success": bool(success),
            "total_reward": sum(rewards_list),
            "steps": len(rewards_list),
        }

    def train(
        self,
        vla: Any,
        env: Any,
        feature_extractor: Any,
        n_episodes: int = 1000,
        update_every: int = 10,
        max_steps: int = 300,
        instruction: str = "",
        verbose: bool = True,
    ) -> dict:
        """
        Full training loop.

        Collects episodes, updates corrector every `update_every` episodes.
        """
        # Buffers for PPO update
        all_features = []
        all_vla_actions = []
        all_residuals = []
        all_log_probs = []
        all_rewards = []

        history = {"episode_reward": [], "episode_success": [], "policy_loss": []}
        success_window = []

        for ep in range(n_episodes):
            ep_data = self.train_episode(
                vla, env, feature_extractor, max_steps, instruction
            )

            # Accumulate
            all_features.extend(ep_data["features"])
            all_vla_actions.extend(ep_data["vla_actions"])
            all_residuals.extend(ep_data["residuals"])
            all_log_probs.extend(ep_data["log_probs"])
            all_rewards.extend(ep_data["rewards"])

            success_window.append(float(ep_data["success"]))
            history["episode_reward"].append(ep_data["total_reward"])
            history["episode_success"].append(float(ep_data["success"]))

            # Update every N episodes
            if (ep + 1) % update_every == 0 and len(all_rewards) > 0:
                # Use sparse rewards: 1 for last step if success, 0 otherwise
                sparse_rewards = [0.0] * len(all_rewards)
                if ep_data["success"]:
                    sparse_rewards[-1] = 1.0

                metrics = self.update(
                    all_features, all_vla_actions, all_residuals,
                    all_log_probs, sparse_rewards,
                )
                history["policy_loss"].append(metrics["policy_loss"])

                # Clear buffers
                all_features.clear()
                all_vla_actions.clear()
                all_residuals.clear()
                all_log_probs.clear()
                all_rewards.clear()

                if verbose and (ep + 1) % 50 == 0:
                    recent_success = np.mean(success_window[-50:])
                    print(
                        f"Episode {ep+1}/{n_episodes} | "
                        f"Success: {recent_success:.1%} | "
                        f"Loss: {metrics['policy_loss']:.4f} | "
                        f"Std: {torch.exp(self.log_std).mean().item():.3f}"
                    )

        return history

    def save(self, path: str):
        """Save corrector weights."""
        torch.save({
            "corrector": self.corrector.state_dict(),
            "log_std": self.log_std.data,
        }, path)
        print(f"Corrector saved to {path}")

    def load(self, path: str):
        """Load corrector weights."""
        ckpt = torch.load(path, map_location=self.device)
        self.corrector.load_state_dict(ckpt["corrector"])
        self.log_std.data = ckpt["log_std"]
        print(f"Corrector loaded from {path}")
