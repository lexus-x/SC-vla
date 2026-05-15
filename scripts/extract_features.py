#!/usr/bin/env python3
"""
Script 2: Extract features from VLA rollouts.

Runs the VLA in simulation, extracts intermediate transformer features
at each step, and saves everything for detector training.

Usage:
    python scripts/extract_features.py --vla octo --env libero --episodes 100
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scvla.features import FeatureExtractor, OctoFeatureExtractor
from scvla.collector import RolloutCollector


def main():
    parser = argparse.ArgumentParser(description="Extract VLA features from rollouts")
    parser.add_argument("--vla", type=str, default="octo", help="VLA model name")
    parser.add_argument("--env", type=str, default="libero", help="Environment name")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--layer", type=int, default=-1, help="Transformer layer to read")
    parser.add_argument("--output", type=str, default="features.pkl", help="Output file")
    parser.add_argument("--max-steps", type=int, default=300, help="Max steps per episode")
    args = parser.parse_args()

    # Load VLA (reuse from run_baseline)
    from run_baseline import load_vla, load_env

    vla = load_vla(args.vla)
    env = load_env(args.env)

    # Create feature extractor
    if args.vla == "octo":
        extractor = OctoFeatureExtractor(vla, layer=args.layer)
    else:
        extractor = FeatureExtractor(vla, layer=args.layer)

    # Collect rollouts with features
    collector = RolloutCollector(
        vla=vla,
        env=env,
        feature_extractor=extractor,
        max_steps_per_episode=args.max_steps,
    )

    rollouts = collector.collect(n_episodes=args.episodes)
    collector.save(rollouts, args.output)

    # Print summary
    n_success = sum(1 for r in rollouts if r.success)
    n_failure = len(rollouts) - n_success
    n_steps = sum(len(r.steps) for r in rollouts)

    print(f"\nSummary:")
    print(f"  Episodes: {len(rollouts)} ({n_success} success, {n_failure} failure)")
    print(f"  Total steps: {n_steps}")
    print(f"  Features per step: {rollouts[0].steps[0].features.shape}")
    print(f"  Saved to: {args.output}")

    extractor.cleanup()


if __name__ == "__main__":
    main()
