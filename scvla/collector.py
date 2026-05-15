"""
Rollout Collector: runs VLA in simulation, collects features + actions + labels.

This is the data collection pipeline. It:
1. Runs the VLA in a simulation environment
2. Extracts intermediate features at each step
3. Records actions and outcomes (success/failure)
4. Saves everything for detector training
"""

import torch
import numpy as np
import pickle
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path


@dataclass
class StepData:
    """Data from a single environment step."""
    timestep: int
    observation: dict
    instruction: str
    action: np.ndarray
    features: Optional[torch.Tensor] = None
    reward: float = 0.0
    done: bool = False
    info: dict = field(default_factory=dict)


@dataclass
class RolloutData:
    """Data from a complete rollout (episode)."""
    episode_id: int
    steps: List[StepData]
    success: bool
    total_reward: float

    @property
    def features(self) -> torch.Tensor:
        """Stack all features from this rollout."""
        return torch.stack([s.features for s in self.steps if s.features is not None])

    @property
    def actions(self) -> torch.Tensor:
        """Stack all actions from this rollout."""
        return torch.tensor(np.stack([s.action for s in self.steps]))

    @property
    def labels(self) -> torch.Tensor:
        """Binary label for each step: 1 if rollout succeeded, 0 otherwise."""
        return torch.tensor([float(self.success)] * len(self.steps))


class RolloutCollector:
    """
    Collects rollouts from a VLA in a simulation environment.

    Usage:
        collector = RolloutCollector(vla, env, feature_extractor)
        rollouts = collector.collect(n_episodes=100)
        collector.save(rollouts, "rollouts.pkl")
    """

    def __init__(
        self,
        vla: Any,
        env: Any,
        feature_extractor: Any,
        max_steps_per_episode: int = 300,
    ):
        self.vla = vla
        self.env = env
        self.feature_extractor = feature_extractor
        self.max_steps = max_steps_per_episode

    def collect_episode(self, episode_id: int, instruction: str = "") -> RolloutData:
        """Collect a single episode rollout."""
        steps = []
        total_reward = 0.0

        observation, info = self.env.reset()

        for t in range(self.max_steps):
            # Extract features from VLA
            features = self.feature_extractor.extract(observation, instruction)

            # Get VLA action
            action = self.vla.predict_action(observation, instruction)
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()

            # Step environment
            next_obs, reward, terminated, truncated, step_info = self.env.step(action)
            done = terminated or truncated

            step = StepData(
                timestep=t,
                observation=observation,
                instruction=instruction,
                action=action,
                features=features,
                reward=reward,
                done=done,
                info=step_info,
            )
            steps.append(step)
            total_reward += reward

            observation = next_obs

            if done:
                break

        success = info.get("success", False) or info.get("is_success", False)

        return RolloutData(
            episode_id=episode_id,
            steps=steps,
            success=bool(success),
            total_reward=total_reward,
        )

    def collect(
        self,
        n_episodes: int,
        instruction: str = "",
        verbose: bool = True,
    ) -> List[RolloutData]:
        """Collect multiple episodes."""
        rollouts = []

        for i in range(n_episodes):
            rollout = self.collect_episode(i, instruction)
            rollouts.append(rollout)

            if verbose and (i + 1) % 10 == 0:
                successes = sum(1 for r in rollouts if r.success)
                print(
                    f"Episode {i+1}/{n_episodes} | "
                    f"Success rate: {successes}/{len(rollouts)} = "
                    f"{successes/len(rollouts):.1%}"
                )

        if verbose:
            successes = sum(1 for r in rollouts if r.success)
            print(f"\nFinal: {successes}/{n_episodes} = {successes/n_episodes:.1%}")

        return rollouts

    def save(self, rollouts: List[RolloutData], path: str):
        """Save rollouts to disk."""
        # Convert tensors to CPU numpy for pickling
        serializable = []
        for r in rollouts:
            steps = []
            for s in r.steps:
                feat = s.features.cpu().numpy() if s.features is not None else None
                steps.append({
                    "timestep": s.timestep,
                    "action": s.action,
                    "features": feat,
                    "reward": s.reward,
                    "done": s.done,
                })
            serializable.append({
                "episode_id": r.episode_id,
                "steps": steps,
                "success": r.success,
                "total_reward": r.total_reward,
            })

        with open(path, "wb") as f:
            pickle.dump(serializable, f)

        print(f"Saved {len(rollouts)} rollouts to {path}")

    @staticmethod
    def load(path: str) -> List[RolloutData]:
        """Load rollouts from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        rollouts = []
        for r in data:
            steps = []
            for s in r["steps"]:
                feat = torch.tensor(s["features"]) if s["features"] is not None else None
                steps.append(StepData(
                    timestep=s["timestep"],
                    observation={},  # not saved to save space
                    instruction="",
                    action=s["action"],
                    features=feat,
                    reward=s["reward"],
                    done=s["done"],
                ))
            rollouts.append(RolloutData(
                episode_id=r["episode_id"],
                steps=steps,
                success=r["success"],
                total_reward=r["total_reward"],
            ))

        print(f"Loaded {len(rollouts)} rollouts from {path}")
        return rollouts
