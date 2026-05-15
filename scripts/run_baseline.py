#!/usr/bin/env python3
"""
Script 1: Run baseline VLA (no correction) to establish performance.

Usage:
    python scripts/run_baseline.py --vla octo --env libero --episodes 100
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_vla(vla_name: str):
    """Load a VLA model."""
    if vla_name == "octo":
        try:
            from octo.model.octo_model import OctoModel
            print("Loading Octo model...")
            model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small")
            return model
        except ImportError:
            print("ERROR: octo-model not installed. pip install octo-model")
            sys.exit(1)
    elif vla_name == "openvla":
        try:
            from transformers import AutoModelForVision2Seq
            print("Loading OpenVLA model...")
            model = AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b")
            return model
        except ImportError:
            print("ERROR: transformers not installed. pip install transformers")
            sys.exit(1)
    else:
        print(f"ERROR: Unknown VLA '{vla_name}'. Supported: octo, openvla")
        sys.exit(1)


def load_env(env_name: str):
    """Load a simulation environment."""
    if env_name == "libero":
        try:
            from libero.libero import benchmark
            print("Loading LIBERO environment...")
            benchmark = benchmark.get_benchmark()
            task_suite = benchmark.get_task_suite("libero_spatial")
            return task_suite
        except ImportError:
            print("ERROR: libero not installed. pip install libero")
            sys.exit(1)
    elif env_name == "metaworld":
        try:
            import metaworld
            print("Loading MetaWorld environment...")
            ml1 = metaworld.ML1("reach-v2")
            env = ml1.train_classes["reach-v2"]()
            return env
        except ImportError:
            print("ERROR: metaworld not installed. pip install metaworld")
            sys.exit(1)
    elif env_name == "simpler":
        try:
            from simpler_env import SIMPLEREnv
            print("Loading SimplerEnv...")
            env = SIMPLEREnv("google_robot_pick_coke_can")
            return env
        except ImportError:
            print("ERROR: simpler_env not installed.")
            sys.exit(1)
    else:
        print(f"ERROR: Unknown env '{env_name}'. Supported: libero, metaworld, simpler")
        sys.exit(1)


def run_baseline(vla, env, n_episodes: int, max_steps: int = 300):
    """Run baseline VLA without correction."""
    successes = 0
    total_rewards = []

    for ep in range(n_episodes):
        observation, info = env.reset()
        total_reward = 0.0

        for t in range(max_steps):
            # Get action from VLA
            action = vla.predict_action(observation, "")
            if hasattr(action, "cpu"):
                action = action.cpu().numpy()

            # Step environment
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        success = info.get("success", False) or info.get("is_success", False)
        successes += int(success)
        total_rewards.append(total_reward)

        if (ep + 1) % 10 == 0:
            print(
                f"Episode {ep+1}/{n_episodes} | "
                f"Success: {successes}/{ep+1} = {successes/(ep+1):.1%} | "
                f"Avg Reward: {sum(total_rewards)/len(total_rewards):.2f}"
            )

    print("\n" + "=" * 60)
    print("BASELINE RESULTS")
    print("=" * 60)
    print(f"Episodes: {n_episodes}")
    print(f"Success Rate: {successes}/{n_episodes} = {successes/n_episodes:.1%}")
    print(f"Avg Reward: {sum(total_rewards)/len(total_rewards):.2f}")


def main():
    parser = argparse.ArgumentParser(description="Run baseline VLA (no correction)")
    parser.add_argument("--vla", type=str, default="octo", help="VLA model name")
    parser.add_argument("--env", type=str, default="libero", help="Environment name")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=300, help="Max steps per episode")
    args = parser.parse_args()

    vla = load_vla(args.vla)
    env = load_env(args.env)
    run_baseline(vla, env, args.episodes, args.max_steps)


if __name__ == "__main__":
    main()
