#!/usr/bin/env python3
"""
Script 5: Evaluate VLA with SC-VLA correction.

Compares baseline VLA vs. VLA+SC-VLA on the same tasks.

Usage:
    python scripts/evaluate.py --vla octo --env libero --module scvla_best.pt --episodes 100
"""

import argparse
import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scvla.pipeline import SCVLA, SCVLAConfig


def evaluate(vla, env, scvla, n_episodes: int, max_steps: int = 300, use_correction: bool = True):
    """Evaluate VLA with or without correction."""
    successes = 0
    total_rewards = []
    corrections_applied = 0
    total_steps = 0

    for ep in range(n_episodes):
        observation, info = env.reset()
        total_reward = 0.0
        ep_corrections = 0

        for t in range(max_steps):
            if use_correction and scvla is not None:
                result = scvla.predict(observation, "", apply_correction=True)
                action = result["action"].cpu().numpy().squeeze()
                if result["corrected"]:
                    ep_corrections += 1
            else:
                action = vla.predict_action(observation, "")
                if hasattr(action, "cpu"):
                    action = action.cpu().numpy()

            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            total_steps += 1

            if terminated or truncated:
                break

        success = info.get("success", False) or info.get("is_success", False)
        successes += int(success)
        total_rewards.append(total_reward)
        corrections_applied += ep_corrections

        if (ep + 1) % 10 == 0:
            print(
                f"Episode {ep+1}/{n_episodes} | "
                f"Success: {successes}/{ep+1} = {successes/(ep+1):.1%} | "
                f"Corrections: {ep_corrections}"
            )

    return {
        "success_rate": successes / n_episodes,
        "avg_reward": sum(total_rewards) / len(total_rewards),
        "corrections_applied": corrections_applied,
        "correction_rate": corrections_applied / max(total_steps, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLA with SC-VLA correction")
    parser.add_argument("--vla", type=str, default="octo", help="VLA model name")
    parser.add_argument("--env", type=str, default="libero", help="Environment name")
    parser.add_argument("--module", type=str, default="scvla_best.pt", help="SC-VLA checkpoint")
    parser.add_argument("--episodes", type=int, default=100, help="Evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=300, help="Max steps per episode")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    args = parser.parse_args()

    # Load VLA
    from run_baseline import load_vla, load_env

    vla = load_vla(args.vla)
    env = load_env(args.env)

    # Load SC-VLA
    print(f"Loading SC-VLA from {args.module}...")
    scvla = SCVLA(vla, env, config=SCVLAConfig(device=args.device))
    scvla.load(args.module)

    # Evaluate WITHOUT correction (baseline)
    print("\n" + "=" * 60)
    print("BASELINE (no correction)")
    print("=" * 60)
    baseline_results = evaluate(vla, env, None, args.episodes, args.max_steps, use_correction=False)

    # Evaluate WITH correction
    print("\n" + "=" * 60)
    print("WITH SC-VLA CORRECTION")
    print("=" * 60)
    corrected_results = evaluate(vla, env, scvla, args.episodes, args.max_steps, use_correction=True)

    # Summary
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<25} {'Baseline':>12} {'SC-VLA':>12} {'Delta':>12}")
    print("-" * 61)
    print(
        f"{'Success Rate':<25} {baseline_results['success_rate']:>11.1%} "
        f"{corrected_results['success_rate']:>11.1%} "
        f"{corrected_results['success_rate'] - baseline_results['success_rate']:>+11.1%}"
    )
    print(
        f"{'Avg Reward':<25} {baseline_results['avg_reward']:>12.2f} "
        f"{corrected_results['avg_reward']:>12.2f} "
        f"{corrected_results['avg_reward'] - baseline_results['avg_reward']:>+12.2f}"
    )
    print(f"{'Corrections Applied':<25} {'N/A':>12} {corrected_results['corrections_applied']:>12}")
    print(f"{'Correction Rate':<25} {'N/A':>12} {corrected_results['correction_rate']:>11.1%}")


if __name__ == "__main__":
    main()
