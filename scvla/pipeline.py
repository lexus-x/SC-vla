"""
Main SC-VLA Module: the end-to-end pipeline.

Combines feature extraction, detection, and correction into a single
easy-to-use interface.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass

from .correction import VCM, VCMConfig, SuccessDetector, ActionCorrector
from .features import FeatureExtractor, OctoFeatureExtractor
from .collector import RolloutCollector, RolloutData
from .detector import DetectorTrainer
from .corrector import PPOCorrectorTrainer


@dataclass
class SCVLAConfig:
    """Configuration for the full SC-VLA pipeline."""

    # Feature extraction
    feature_layer: int = -1

    # Detector
    detector_hidden: int = 64
    detector_epochs: int = 50
    detector_lr: float = 1e-3

    # Corrector
    corrector_hidden: int = 64
    corrector_lr: float = 3e-4
    correction_threshold: float = 0.5
    max_correction: float = 0.3

    # Training
    n_rollout_episodes: int = 100
    n_corrector_episodes: int = 1000
    corrector_update_every: int = 10
    max_steps_per_episode: int = 300

    # Device
    device: str = "cpu"


class SCVLA:
    """
    Self-Correcting VLA: end-to-end pipeline.

    Usage:
        scvla = SCVLA(vla, env, config)
        scvla.train()                    # full training pipeline
        action = scvla.predict(obs, instr)  # inference with correction
    """

    def __init__(
        self,
        vla: Any,
        env: Any,
        config: Optional[SCVLAConfig] = None,
        vla_type: str = "auto",
    ):
        self.vla = vla
        self.env = env
        self.config = config or SCVLAConfig()

        # Detect VLA type and create feature extractor
        self.feature_extractor = self._create_feature_extractor(vla_type)

        # Probe feature dimension
        self.feature_dim = self._probe_feature_dim()
        self.action_dim = self._probe_action_dim()

        # Create VCM (detector + corrector)
        vcm_config = VCMConfig(
            feature_layer=self.config.feature_layer,
            detector_hidden=self.config.detector_hidden,
            corrector_hidden=self.config.corrector_hidden,
            correction_threshold=self.config.correction_threshold,
            max_correction=self.config.max_correction,
            device=self.config.device,
        )
        self.vcm = VCM(
            feature_dim=self.feature_dim,
            action_dim=self.action_dim,
            config=vcm_config,
        ).to(self.config.device)

        # Trainers
        self.detector_trainer = DetectorTrainer(
            self.vcm.detector,
            lr=self.config.detector_lr,
            device=self.config.device,
        )
        self.corrector_trainer = PPOCorrectorTrainer(
            self.vcm.corrector,
            feature_dim=self.feature_dim,
            action_dim=self.action_dim,
            lr=self.config.corrector_lr,
            device=self.config.device,
        )

        # Collector
        self.collector = RolloutCollector(
            vla=vla,
            env=env,
            feature_extractor=self.feature_extractor,
            max_steps_per_episode=self.config.max_steps_per_episode,
        )

    def _create_feature_extractor(self, vla_type: str) -> FeatureExtractor:
        """Create the right feature extractor for the VLA type."""
        if vla_type == "octo":
            return OctoFeatureExtractor(self.vla, layer=self.config.feature_layer)
        elif vla_type == "auto":
            return FeatureExtractor(self.vla, layer=self.config.feature_layer)
        else:
            return FeatureExtractor(self.vla, layer=self.config.feature_layer)

    def _probe_feature_dim(self) -> int:
        """Probe the VLA to determine feature dimensionality."""
        try:
            # Create a dummy observation
            dummy_obs = {"image": np.zeros((224, 224, 3), dtype=np.uint8)}
            features = self.feature_extractor.extract(dummy_obs, "")
            return features.shape[-1]
        except Exception:
            # Fallback: common dimensions
            print("Warning: Could not probe feature dim. Using default 768.")
            return 768

    def _probe_action_dim(self) -> int:
        """Probe the VLA to determine action dimensionality."""
        try:
            dummy_obs = {"image": np.zeros((224, 224, 3), dtype=np.uint8)}
            action = self.vla.predict_action(dummy_obs, "")
            if isinstance(action, np.ndarray):
                return action.shape[-1]
            elif isinstance(action, torch.Tensor):
                return action.shape[-1]
        except Exception:
            pass
        # Fallback
        print("Warning: Could not probe action dim. Using default 7.")
        return 7

    def train(self, instruction: str = ""):
        """
        Full training pipeline:
        1. Collect rollouts (features + labels)
        2. Train detector (supervised)
        3. Train corrector (PPO)
        """
        print("=" * 60)
        print("SC-VLA Training Pipeline")
        print("=" * 60)

        # Step 1: Collect rollouts
        print("\n[1/3] Collecting rollouts...")
        rollouts = self.collector.collect(
            n_episodes=self.config.n_rollout_episodes,
            instruction=instruction,
        )
        self.collector.save(rollouts, "rollouts.pkl")

        # Step 2: Train detector
        print("\n[2/3] Training detector...")
        features, actions, labels = self._prepare_detector_data(rollouts)
        detector_history = self.detector_trainer.train(
            features, actions, labels,
            epochs=self.config.detector_epochs,
        )
        self.detector_trainer.save("detector_best.pt")

        # Step 3: Train corrector
        print("\n[3/3] Training corrector (PPO)...")
        corrector_history = self.corrector_trainer.train(
            self.vla, self.env, self.feature_extractor,
            n_episodes=self.config.n_corrector_episodes,
            update_every=self.config.corrector_update_every,
            max_steps=self.config.max_steps_per_episode,
            instruction=instruction,
        )
        self.corrector_trainer.save("corrector_best.pt")

        # Save full module
        self.save("scvla_best.pt")

        return {
            "detector": detector_history,
            "corrector": corrector_history,
        }

    def _prepare_detector_data(
        self, rollouts: list
    ) -> tuple:
        """Prepare features, actions, labels from rollouts."""
        all_features = []
        all_actions = []
        all_labels = []

        for rollout in rollouts:
            for step in rollout.steps:
                if step.features is not None:
                    all_features.append(step.features)
                    all_actions.append(torch.tensor(step.action, dtype=torch.float32))
                    all_labels.append(float(rollout.success))

        features = torch.stack(all_features)
        actions = torch.stack(all_actions)
        labels = torch.tensor(all_labels)

        return features, actions, labels

    @torch.no_grad()
    def predict(
        self,
        observation: dict,
        instruction: str = "",
        apply_correction: bool = True,
    ) -> dict:
        """
        Predict action with optional correction.

        Args:
            observation: dict with 'image', etc.
            instruction: language instruction
            apply_correction: whether to apply correction

        Returns:
            dict with: action, p_success, corrected, correction
        """
        features = self.feature_extractor.extract(observation, instruction)

        vla_action = self.vla.predict_action(observation, instruction)
        if isinstance(vla_action, np.ndarray):
            vla_action = torch.tensor(vla_action, dtype=torch.float32).unsqueeze(0)
        elif isinstance(vla_action, torch.Tensor):
            vla_action = vla_action.unsqueeze(0) if vla_action.dim() == 1 else vla_action

        return self.vcm.predict(features, vla_action, apply_correction)

    def save(self, path: str):
        """Save the full SC-VLA module."""
        torch.save({
            "vcm": self.vcm.state_dict(),
            "config": self.config,
            "feature_dim": self.feature_dim,
            "action_dim": self.action_dim,
        }, path)
        print(f"SC-VLA saved to {path}")

    def load(self, path: str):
        """Load the full SC-VLA module."""
        ckpt = torch.load(path, map_location=self.config.device)
        self.vcm.load_state_dict(ckpt["vcm"])
        print(f"SC-VLA loaded from {path}")

    def cleanup(self):
        """Clean up hooks."""
        self.feature_extractor.cleanup()
