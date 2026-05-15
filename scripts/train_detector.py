#!/usr/bin/env python3
"""
Script 3: Train the success detector (supervised).

The detector learns to predict P(success) from VLA features + action.

Usage:
    python scripts/train_detector.py --data features.pkl --epochs 50
"""

import argparse
import sys
import os
import torch
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scvla.detector import DetectorTrainer
from scvla.correction import SuccessDetector


def main():
    parser = argparse.ArgumentParser(description="Train success detector")
    parser.add_argument("--data", type=str, default="features.pkl", help="Rollout data file")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--output", type=str, default="detector_best.pt", help="Output file")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data}...")
    with open(args.data, "rb") as f:
        rollouts = pickle.load(f)

    # Prepare features, actions, labels
    all_features = []
    all_actions = []
    all_labels = []

    for r in rollouts:
        for step in r["steps"]:
            if step["features"] is not None:
                all_features.append(torch.tensor(step["features"]))
                all_actions.append(torch.tensor(step["action"], dtype=torch.float32))
                all_labels.append(float(r["success"]))

    features = torch.stack(all_features)
    actions = torch.stack(all_actions)
    labels = torch.tensor(all_labels)

    print(f"Data: {features.shape[0]} samples, {labels.sum().int()} success, {(1-labels).sum().int()} failure")
    print(f"Feature dim: {features.shape[-1]}, Action dim: {actions.shape[-1]}")

    # Create detector
    detector = SuccessDetector(
        feature_dim=features.shape[-1],
        action_dim=actions.shape[-1],
        hidden_dim=args.hidden,
    )

    # Train
    trainer = DetectorTrainer(detector, lr=args.lr, device=args.device)
    history = trainer.train(
        features, actions, labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # Save
    trainer.save(args.output)

    # Print final metrics
    print("\n" + "=" * 60)
    print("DETECTOR TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best Val AUC: {max(history['val_auc']):.3f}")
    print(f"Best Val Accuracy: {max(history['val_accuracy']):.3f}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
