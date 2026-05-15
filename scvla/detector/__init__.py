"""
Detector Trainer: supervised training of the success detector.

The detector learns to predict P(success) from VLA features + action.
Training data comes from rollout collection: each step gets a binary label
(1 if the episode succeeded, 0 if it failed).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple
from sklearn.metrics import roc_auc_score
from pathlib import Path


class DetectorTrainer:
    """
    Trains the success detector on collected rollout data.

    Usage:
        trainer = DetectorTrainer(detector, lr=1e-3)
        trainer.train(features, actions, labels, epochs=50)
    """

    def __init__(
        self,
        detector: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "cpu",
    ):
        self.detector = detector.to(device)
        self.device = device
        self.optimizer = optim.Adam(
            detector.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.criterion = nn.BCELoss()

    def train_epoch(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int = 256,
    ) -> float:
        """Train for one epoch. Returns average loss."""
        self.detector.train()
        n = features.shape[0]
        indices = torch.randperm(n)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]

            feat_batch = features[idx].to(self.device)
            act_batch = actions[idx].to(self.device)
            label_batch = labels[idx].to(self.device)

            self.optimizer.zero_grad()
            p_success = self.detector(feat_batch, act_batch).squeeze(-1)
            loss = self.criterion(p_success, label_batch)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def evaluate(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict:
        """Evaluate detector. Returns AUC and accuracy."""
        self.detector.eval()
        feat = features.to(self.device)
        act = actions.to(self.device)

        p_success = self.detector(feat, act).squeeze(-1).cpu().numpy()
        label_np = labels.numpy()

        # AUC
        try:
            auc = roc_auc_score(label_np, p_success)
        except ValueError:
            auc = 0.0

        # Accuracy at threshold 0.5
        predictions = (p_success > 0.5).astype(float)
        accuracy = (predictions == label_np).mean()

        return {"auc": auc, "accuracy": accuracy}

    def train(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        labels: torch.Tensor,
        epochs: int = 50,
        batch_size: int = 256,
        val_split: float = 0.2,
        verbose: bool = True,
    ) -> dict:
        """
        Full training loop with validation.

        Args:
            features: (N, feature_dim) all features
            actions: (N, action_dim) all actions
            labels: (N,) binary labels (1=success, 0=failure)
            epochs: number of training epochs
            batch_size: batch size
            val_split: fraction of data for validation
            verbose: print progress

        Returns:
            dict with training history
        """
        n = features.shape[0]
        n_val = int(n * val_split)
        n_train = n - n_val

        # Split
        perm = torch.randperm(n)
        train_idx, val_idx = perm[:n_train], perm[n_train:]

        train_feat, val_feat = features[train_idx], features[val_idx]
        train_act, val_act = actions[train_idx], actions[val_idx]
        train_lab, val_lab = labels[train_idx], labels[val_idx]

        history = {"train_loss": [], "val_auc": [], "val_accuracy": []}
        best_auc = 0.0

        for epoch in range(epochs):
            loss = self.train_epoch(train_feat, train_act, train_lab, batch_size)
            val_metrics = self.evaluate(val_feat, val_act, val_lab)

            history["train_loss"].append(loss)
            history["val_auc"].append(val_metrics["auc"])
            history["val_accuracy"].append(val_metrics["accuracy"])

            if val_metrics["auc"] > best_auc:
                best_auc = val_metrics["auc"]

            if verbose and (epoch + 1) % 5 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Loss: {loss:.4f} | "
                    f"Val AUC: {val_metrics['auc']:.3f} | "
                    f"Val Acc: {val_metrics['accuracy']:.3f}"
                )

        if verbose:
            print(f"\nBest Val AUC: {best_auc:.3f}")

        return history

    def save(self, path: str):
        """Save detector weights."""
        torch.save(self.detector.state_dict(), path)
        print(f"Detector saved to {path}")

    def load(self, path: str):
        """Load detector weights."""
        self.detector.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Detector loaded from {path}")
