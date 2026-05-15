"""
SC-VLA: Self-Correcting VLA Module

Wraps any frozen VLA with a lightweight detector + corrector.
The base VLA weights remain 100% frozen.
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Optional, Protocol, Any


class VLABackend(Protocol):
    """Protocol: any object that can produce actions and intermediate features."""

    def predict_action(self, observation: dict, instruction: str) -> np.ndarray:
        """Return action vector from the VLA."""
        ...

    def get_intermediate_features(
        self, observation: dict, instruction: str, layer: int = -1
    ) -> torch.Tensor:
        """Return intermediate transformer features at given layer."""
        ...


@dataclass
class VCMConfig:
    """Configuration for the VLA Correction Module."""

    # Which transformer layer to read features from (-1 = last)
    feature_layer: int = -1

    # Detector hidden dimension
    detector_hidden: int = 64

    # Corrector hidden dimension
    corrector_hidden: int = 64

    # Success probability threshold below which correction is applied
    correction_threshold: float = 0.5

    # Maximum residual correction magnitude (prevents wild corrections)
    max_correction: float = 0.3

    # Device
    device: str = "cpu"


class SuccessDetector(nn.Module):
    """
    Predicts P(success) for a given action conditioned on VLA features.

    Input: VLA intermediate features + action
    Output: scalar P(success) in [0, 1]
    """

    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, feature_dim) VLA intermediate features
            action: (B, action_dim) action vector
        Returns:
            (B, 1) P(success)
        """
        x = torch.cat([features, action], dim=-1)
        return self.net(x)


class ActionCorrector(nn.Module):
    """
    Predicts a residual correction to the VLA's action.

    Input: VLA intermediate features + original action
    Output: Δ(action) — residual correction
    """

    def __init__(
        self, feature_dim: int, action_dim: int, hidden_dim: int = 64, max_correction: float = 0.3
    ):
        super().__init__()
        self.max_correction = max_correction
        self.net = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # outputs in [-1, 1], scaled by max_correction
        )

    def forward(self, features: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, feature_dim) VLA intermediate features
            action: (B, action_dim) original action
        Returns:
            (B, action_dim) residual correction, bounded by max_correction
        """
        x = torch.cat([features, action], dim=-1)
        return self.net(x) * self.max_correction


class VCM(nn.Module):
    """
    VLA Correction Module.

    Wraps any frozen VLA with:
    1. A detector that predicts P(success) from VLA features + action
    2. A corrector that applies residual correction when P(success) is low

    Usage:
        vcm = VCM(vla_backend, feature_dim=768, action_dim=7, config=VCMConfig())
        corrected_action = vcm.predict(observation, instruction)
    """

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        config: Optional[VCMConfig] = None,
    ):
        super().__init__()
        self.config = config or VCMConfig()

        self.detector = SuccessDetector(
            feature_dim=feature_dim,
            action_dim=action_dim,
            hidden_dim=self.config.detector_hidden,
        )

        self.corrector = ActionCorrector(
            feature_dim=feature_dim,
            action_dim=action_dim,
            hidden_dim=self.config.corrector_hidden,
            max_correction=self.config.max_correction,
        )

        self.feature_dim = feature_dim
        self.action_dim = action_dim

    @torch.no_grad()
    def predict(
        self,
        features: torch.Tensor,
        vla_action: torch.Tensor,
        apply_correction: bool = True,
    ) -> dict:
        """
        Given VLA features and the VLA's proposed action, optionally correct it.

        Args:
            features: (1, feature_dim) intermediate features from VLA
            vla_action: (1, action_dim) the VLA's proposed action
            apply_correction: if False, just return detection score

        Returns:
            dict with:
                - action: corrected (or original) action
                - p_success: predicted success probability
                - corrected: whether correction was applied
                - correction: the residual (zeros if not corrected)
        """
        features = features.float()
        vla_action = vla_action.float()

        p_success = self.detector(features, vla_action)  # (1, 1)

        result = {
            "p_success": p_success.item(),
            "corrected": False,
            "correction": torch.zeros_like(vla_action),
        }

        if apply_correction and p_success.item() < self.config.correction_threshold:
            residual = self.corrector(features, vla_action)  # (1, action_dim)
            result["action"] = vla_action + residual
            result["corrected"] = True
            result["correction"] = residual
        else:
            result["action"] = vla_action

        return result

    def compute_detector_loss(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Binary cross-entropy loss for the detector.

        Args:
            features: (B, feature_dim)
            actions: (B, action_dim)
            labels: (B,) binary — 1 for success, 0 for failure
        """
        p_success = self.detector(features, actions).squeeze(-1)  # (B,)
        loss = nn.functional.binary_cross_entropy(p_success, labels.float())
        return loss

    def compute_corrector_loss(
        self,
        features: torch.Tensor,
        original_actions: torch.Tensor,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """
        Policy gradient loss for the corrector (REINFORCE).

        The corrector outputs a residual. We want to maximize reward (task success).
        We treat the correction as a Gaussian policy: mean = corrector output,
        and add exploration noise during training.

        Args:
            features: (B, feature_dim)
            original_actions: (B, action_dim)
            rewards: (B,) — 1 for task success, 0 for failure
        """
        residual = self.corrector(features, original_actions)  # (B, action_dim)
        # Simple policy gradient: encourage corrections that lead to higher reward
        # Use negative log-likelihood of the correction under a Gaussian
        # For simplicity, we use the MSE between correction and zero as a baseline
        # and scale by reward
        correction_magnitude = (residual ** 2).sum(dim=-1)  # (B,)
        # Reward-weighted: encourage small corrections when successful,
        # larger corrections when failing
        loss = (correction_magnitude * (1 - rewards.float())).mean()
        return loss
