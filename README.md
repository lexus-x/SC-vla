# SC-VLA: Self-Correcting Vision-Language-Action Module

A lightweight, VLA-agnostic correction module that detects and corrects
likely-failed actions using the VLA's own internal features.

## Problem

Current VLAs (Octo, OpenVLA, π₀, etc.) are **reactive**: they produce actions
from a single observation with no self-correction mechanism. When the VLA makes
a mistake — wrong grasp target, oscillating trajectory, imprecise positioning —
there is no feedback loop to catch and fix it before execution.

## Solution

SC-VLA wraps any frozen VLA with two tiny modules (~10K params total):

1. **Detector**: Reads intermediate transformer features → predicts P(success)
2. **Corrector**: If P(success) is low → applies a learned residual correction

The base VLA weights remain 100% frozen. Only the correction module is trained.

## How It Differs

| Method | Detects | Corrects | VLA-agnostic | Params | VLA frozen? |
|--------|---------|----------|---------------|--------|-------------|
| SAFE (2025) | ✅ | ❌ | ✅ | ~10K | ✅ |
| RoboFAC (2026) | ✅ | ✅ | ✅ (external) | 7B | ✅ |
| SC-VLA (2026) | ✅ | ✅ | ❌ | VLA+ | ❌ |
| AFIL (2026) | ✅ | ✅ | ❌ (diffusion) | VLA+ | ❌ |
| AR-VLA (2026) | ✅ | ✅ | ❌ | VLA+ | ❌ |
| **Ours** | ✅ | ✅ | ✅ | ~10K | ✅ |

Key distinction: **No prior work is simultaneously lightweight, VLA-agnostic,
and capable of both detection AND correction with a frozen base VLA.**

## Architecture

```
Observation + Instruction
        │
        ▼
   ┌──────────┐
   │   VLA    │  (frozen — Octo, OpenVLA, π₀, etc.)
   │ backbone │
   └────┬─────┘
        │ intermediate features (layer N)
        ▼
   ┌──────────────┐
   │  Detector    │  MLP: features → P(success)
   │  (~5K params)│
   └──────┬───────┘
          │
    P(success) < threshold?
          │
    ┌─────┴──────┐
    │ Yes        │ No → use original VLA action
    ▼
┌──────────────┐
│  Corrector   │  MLP: features → Δ(action)
│  (~5K params)│
└──────┬───────┘
       │
       ▼
  action + Δ(action) = corrected action
```

## Quick Start

```bash
pip install -e .

# 1. Run baseline (no correction)
python scripts/run_baseline.py --vla octo --env libero --episodes 100

# 2. Extract features from rollouts
python scripts/extract_features.py --vla octo --env libero --episodes 100

# 3. Train detector
python scripts/train_detector.py --data features.pkl --epochs 50

# 4. Train corrector (PPO in simulation)
python scripts/train_corrector.py --vla octo --env libero --episodes 1000

# 5. Evaluate with correction
python scripts/evaluate.py --vla octo --env libero --module vcm_best.pt
```

## Supported VLAs

Any transformer-based VLA where you can read intermediate layer features:
- Octo
- OpenVLA
- π₀ / π₀-FAST
- RDT-1B
- GR00T

## Supported Environments

Any simulation environment with:
- Ground-truth success/failure labels
- Standard gym interface (`env.reset()`, `env.step(action)`)

Tested on: LIBERO, MetaWorld, SimplerEnv, RoboTwin

## Requirements

- Python 3.10+
- PyTorch 2.0+
- GPU with ≥16GB VRAM (for VLA inference)

## License

MIT
