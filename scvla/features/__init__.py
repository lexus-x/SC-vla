"""
Feature Extractor: reads intermediate transformer features from VLAs.

Supports: Octo, OpenVLA, and any transformer-based VLA that exposes
intermediate layer outputs.
"""

import torch
import numpy as np
from typing import Optional, Protocol, Any


class FeatureExtractor:
    """
    Extracts intermediate transformer features from a VLA.

    For transformer-based VLAs, this hooks into the attention layers
    and extracts the hidden state at a specified layer.
    """

    def __init__(self, vla: Any, layer: int = -1):
        """
        Args:
            vla: The VLA model instance
            layer: Which transformer layer to read from (-1 = last)
        """
        self.vla = vla
        self.layer = layer
        self._features: Optional[torch.Tensor] = None
        self._hook_handle = None

    def _register_hook(self):
        """Register a forward hook to capture intermediate features."""
        # Find the transformer backbone
        backbone = self._find_transformer_backbone()
        if backbone is None:
            raise ValueError(
                "Could not find transformer backbone in VLA. "
                "Supported: Octo, OpenVLA, π₀. "
                "For other VLAs, subclass FeatureExtractor."
            )

        # Get the target layer
        layers = self._get_transformer_layers(backbone)
        if layers is None or len(layers) == 0:
            raise ValueError("Could not find transformer layers in backbone.")

        target_layer = layers[self.layer]

        def hook_fn(module, input, output):
            # output might be a tuple (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                self._features = output[0].detach()
            else:
                self._features = output.detach()

        self._hook_handle = target_layer.register_forward_hook(hook_fn)

    def _find_transformer_backbone(self):
        """Find the transformer backbone in the VLA model."""
        vla = self.vla

        # Octo: uses a transformer with OctoTransformer
        if hasattr(vla, "transformer"):
            return vla.transformer
        if hasattr(vla, "model") and hasattr(vla.model, "transformer"):
            return vla.model.transformer

        # OpenVLA: uses a language model backbone
        if hasattr(vla, "model") and hasattr(vla.model, "model"):
            return vla.model.model
        if hasattr(vla, "language_model"):
            return vla.language_model

        # π₀: uses a flow-matching backbone
        if hasattr(vla, "backbone"):
            return vla.backbone

        # Generic: look for any attribute named "layers" that contains nn.Module
        for name, module in vla.named_children():
            if hasattr(module, "layers") and isinstance(module.layers, (list, torch.nn.ModuleList)):
                return module

        return None

    def _get_transformer_layers(self, backbone):
        """Get the list of transformer layers from a backbone."""
        # Common patterns
        if hasattr(backbone, "layers"):
            layers = backbone.layers
            if isinstance(layers, (list, torch.nn.ModuleList)):
                return layers
        if hasattr(backbone, "block"):
            layers = backbone.block
            if isinstance(layers, (list, torch.nn.ModuleList)):
                return layers
        if hasattr(backbone, "h"):
            layers = backbone.h
            if isinstance(layers, (list, torch.nn.ModuleList)):
                return layers

        return None

    def extract(
        self,
        observation: dict,
        instruction: str,
        action: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """
        Run a forward pass through the VLA and extract intermediate features.

        Args:
            observation: dict with 'image', 'proprioception', etc.
            instruction: language instruction
            action: optional — if provided, the VLA will be run with this action
                    (needed for some VLAs that require action input)

        Returns:
            features: (1, feature_dim) tensor
        """
        if self._hook_handle is None:
            self._register_hook()

        self._features = None

        # Run VLA forward pass to trigger the hook
        self._vla_forward(observation, instruction, action)

        if self._features is None:
            raise RuntimeError(
                "Hook did not capture features. "
                "The VLA forward pass may not have triggered the hooked layer."
            )

        # Pool: if features are (1, seq_len, dim), take mean over seq
        features = self._features
        if features.dim() == 3:
            features = features.mean(dim=1)  # (1, dim)

        return features

    def _vla_forward(self, observation: dict, instruction: str, action: Optional[np.ndarray]):
        """
        Run the VLA's forward pass. Override for specific VLA implementations.
        """
        vla = self.vla

        # Try common VLA interfaces
        # Octo: vla(observation, instruction)
        if hasattr(vla, "__call__"):
            try:
                vla(observation, instruction)
                return
            except TypeError:
                pass

        # OpenVLA: vla.predict(observation, instruction)
        if hasattr(vla, "predict"):
            try:
                vla.predict(observation, instruction)
                return
            except TypeError:
                pass

        raise NotImplementedError(
            f"Cannot call VLA of type {type(vla)}. "
            "Subclass FeatureExtractor and override _vla_forward()."
        )

    def cleanup(self):
        """Remove the forward hook."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None


class OctoFeatureExtractor(FeatureExtractor):
    """
    Feature extractor specifically for Octo VLAs.

    Octo uses a transformer-based architecture with observation tokens
    and action tokens. We hook into the transformer to read features
    at a specified layer.
    """

    def _find_transformer_backbone(self):
        """Octo-specific backbone detection."""
        vla = self.vla

        # Octo model structure: vla.model.transformer or vla.transformer
        if hasattr(vla, "model"):
            model = vla.model
            if hasattr(model, "transformer"):
                return model.transformer
            if hasattr(model, "modules"):
                for module in model.modules():
                    if hasattr(module, "layers") and isinstance(
                        getattr(module, "layers", None), (list, torch.nn.ModuleList)
                    ):
                        return module

        # Direct access
        if hasattr(vla, "transformer"):
            return vla.transformer

        return super()._find_transformer_backbone()

    def _vla_forward(self, observation: dict, instruction: str, action: Optional[np.ndarray]):
        """Octo-specific forward pass."""
        vla = self.vla

        # Octo expects: vla(observation, goal) or vla.predict(observation, goal)
        if hasattr(vla, "predict"):
            vla.predict(observation, instruction)
        elif callable(vla):
            vla(observation, instruction)
        else:
            raise RuntimeError("Cannot call Octo model")
