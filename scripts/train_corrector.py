#!/usr/bin/env python3
"""
Script 4: Train the action corrector (PPO).

The corrector learns residual corrections that improve task success rate.
Only the corrector is trained — the VLA remains frozen.

Usage:
    python scripts/train_corrector.py --vla octo --env libero --episodes 1000
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scvla.correction import ActionCorrector
from scvla.corrector import PPOCorrectorTrainer
from scvla.features import FeatureExtractor, OctoFeatureExtractor


def main():
    parser = argparse.ArgumentParser(description="Train action corrector (PPO)")
    parser.add_argument("--vla", type=str, default="octo", help="VLA model name")
    parser.add_argument("--env", type=str, default="libero", help="Environment name")
    parser.add_argument("--episodes", type=int, default=1000, help="Training episodes")
    parser.add_argument("--update-every", type=int, default=10, help="PPO update frequency")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--max-correction", type=float, default=0.3, help="Max correction magnitude")
    parser.add_argument("--layer", type=int, default=-1, help="Feature layer")
    parser.add_argument("--output", type=str, default="corrector_best.pt", help="Output file")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--max-steps", type=int, default=300, help="Max steps per episode")
    args = parser.parse_args()

    # Load VLA
    from run_baseline import load_vla, load_env

    vla = load_vla(args.vla)
    env = load_env(args.env)

    # Create feature extractor
    if args.vla == "octo":
        extractor = OctoFeatureExtractor(vla, layer=args.layer)
    else:
        extractor = FeatureExtractor(vla, layer=args.layer)

    # Probe dimensions
    feature_dim = 768  # default, will be auto-detected
    action_dim = 7

    # Create corrector
    corrector = ActionCorrector(
        feature_dim=feature_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden,
        max_correction=args.max_correction,
    )

    # Train
    trainer = PPOCorrectorTrainer(
        corrector=corrector,
        feature_dim=feature_dim,
        action_dim=action_dim,
        lr=args.lr,
        device=args.device,
    )

    history = trainer.train(
        vla=vla,
        env=env,
        feature_extractor=extractor,
        n_episodes=args.episodes,
        update_every=args.update_every,
        max_steps=args.max_steps,
    )

    # Save
    trainer.save(args.output)

    # Print summary
    print("\n" + "=" * 60)
    print("CORRECTOR TRAINING COMPLETE")
    print("=" * 60)
    if history["episode_success"]:
        print(f"Final Success Rate: {history['episode_success'][-1]:.1%}")
        print(f"Best Success Rate: {max(history['episode_success']):.1%}")
    print(f"Saved to: {args.output}")

    extractor.cleanup()


if __name__ == "__main__":
    main()
