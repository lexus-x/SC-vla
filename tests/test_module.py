"""
Tests for the SC-VLA correction module.

These tests verify the module's interface and basic functionality
without requiring a real VLA or simulation environment.
"""

import torch
import numpy as np
import pytest
from scvla.correction import VCM, VCMConfig, SuccessDetector, ActionCorrector


class TestSuccessDetector:
    """Tests for the success detector."""

    def test_output_shape(self):
        detector = SuccessDetector(feature_dim=768, action_dim=7, hidden_dim=64)
        features = torch.randn(4, 768)
        actions = torch.randn(4, 7)
        output = detector(features, actions)
        assert output.shape == (4, 1)

    def test_output_range(self):
        """Output should be in [0, 1] (sigmoid)."""
        detector = SuccessDetector(feature_dim=128, action_dim=7)
        features = torch.randn(100, 128)
        actions = torch.randn(100, 7)
        output = detector(features, actions)
        assert (output >= 0).all() and (output <= 1).all()

    def test_different_batch_sizes(self):
        detector = SuccessDetector(feature_dim=64, action_dim=4)
        for batch_size in [1, 8, 32, 128]:
            features = torch.randn(batch_size, 64)
            actions = torch.randn(batch_size, 4)
            output = detector(features, actions)
            assert output.shape == (batch_size, 1)


class TestActionCorrector:
    """Tests for the action corrector."""

    def test_output_shape(self):
        corrector = ActionCorrector(feature_dim=768, action_dim=7, hidden_dim=64)
        features = torch.randn(4, 768)
        actions = torch.randn(4, 7)
        residual = corrector(features, actions)
        assert residual.shape == (4, 7)

    def test_output_bounded(self):
        """Output should be bounded by max_correction."""
        max_corr = 0.3
        corrector = ActionCorrector(feature_dim=128, action_dim=7, max_correction=max_corr)
        features = torch.randn(100, 128)
        actions = torch.randn(100, 7)
        residual = corrector(features, actions)
        assert (residual.abs() <= max_corr + 1e-6).all()

    def test_zero_input_zero_tendency(self):
        """With zero features, output should tend toward zero (tanh)."""
        corrector = ActionCorrector(feature_dim=64, action_dim=4, max_correction=0.1)
        features = torch.zeros(10, 64)
        actions = torch.zeros(10, 4)
        residual = corrector(features, actions)
        # Should be small (tanh(0) = 0, but weights are random)
        assert residual.abs().mean() < 0.2


class TestVCM:
    """Tests for the full VLA Correction Module."""

    def test_predict_no_correction(self):
        """When P(success) is high, no correction should be applied."""
        vcm = VCM(feature_dim=128, action_dim=7, config=VCMConfig(
            correction_threshold=0.5, device="cpu"
        ))

        features = torch.randn(1, 128)
        action = torch.randn(1, 7)

        result = vcm.predict(features, action, apply_correction=True)
        assert "action" in result
        assert "p_success" in result
        assert "corrected" in result
        assert result["action"].shape == (1, 7)

    def test_predict_returns_dict(self):
        vcm = VCM(feature_dim=64, action_dim=4)
        features = torch.randn(1, 64)
        action = torch.randn(1, 4)
        result = vcm.predict(features, action)
        assert isinstance(result, dict)
        assert all(k in result for k in ["action", "p_success", "corrected", "correction"])

    def test_detector_loss(self):
        vcm = VCM(feature_dim=64, action_dim=4)
        features = torch.randn(10, 64)
        actions = torch.randn(10, 4)
        labels = torch.tensor([1, 0, 1, 1, 0, 0, 1, 1, 0, 1], dtype=torch.float32)
        loss = vcm.compute_detector_loss(features, actions, labels)
        assert loss.ndim == 0  # scalar
        assert loss.item() > 0

    def test_corrector_loss(self):
        vcm = VCM(feature_dim=64, action_dim=4)
        features = torch.randn(10, 64)
        actions = torch.randn(10, 4)
        rewards = torch.tensor([1, 0, 1, 1, 0, 0, 1, 1, 0, 1], dtype=torch.float32)
        loss = vcm.compute_corrector_loss(features, actions, rewards)
        assert loss.ndim == 0

    def test_save_load(self, tmp_path):
        vcm = VCM(feature_dim=64, action_dim=4)
        path = str(tmp_path / "vcm_test.pt")
        torch.save(vcm.state_dict(), path)

        vcm2 = VCM(feature_dim=64, action_dim=4)
        vcm2.load_state_dict(torch.load(path))

        # Check parameters match
        for p1, p2 in zip(vcm.parameters(), vcm2.parameters()):
            assert torch.allclose(p1, p2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
