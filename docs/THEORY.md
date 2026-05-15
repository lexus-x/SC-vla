# Theory: Why SC-VLA Works

## The Problem: Reactive VLAs

Current VLAs (Octo, OpenVLA, π₀) are **reactive**: they map observation → action
with no self-correction mechanism. This creates three failure modes:

1. **Action oscillation**: Without memory of past actions, the VLA may alternate
   between conflicting plans across consecutive frames.

2. **Imprecise execution**: The VLA correctly identifies the target but produces
   slightly wrong joint commands. A small correction could fix this.

3. **Cascading errors**: A single imprecise action puts the robot in an
   out-of-distribution state, causing subsequent actions to degrade.

## The Insight: VLA Features Encode Success Prediction

SAFE (2025) demonstrated that VLA internal features contain sufficient information
to predict task success/failure. The transformer's hidden states encode not just
"what to do" but also "how confident I am this will work."

SC-VLA exploits this: the detector reads the VLA's own uncertainty signal and
triggers correction when confidence is low.

## Why Correction Works

When the detector flags a low-confidence action, the corrector applies a small
residual adjustment. This works because:

1. **Most failures are small**: The VLA is usually *almost* right. The target
   object is identified, the approach direction is correct, but the final
   positioning is off by a few centimeters. A residual correction can fix this.

2. **The corrector learns from simulation**: By training with PPO in simulation,
   the corrector learns which types of corrections lead to success. It doesn't
   need to understand the full task — just how to adjust actions.

3. **Bounded corrections**: The max_correction parameter prevents wild adjustments.
   If the VLA is fundamentally wrong (wrong object, wrong task), the corrector
   won't try to fix it — it can only make small adjustments.

## Comparison with Related Work

### SAFE (2025)
SAFE detects failures using VLA features but doesn't correct them.
SC-VLA adds the correction step: when the detector says "this will fail,"
the corrector adjusts the action.

### RoboFAC (2026)
RoboFAC uses a separate 7B MLLM for failure analysis and correction.
SC-VLA uses the VLA's own features with a ~10K param module.
Trade-off: RoboFAC is more capable (can reason about failures semantically)
but much heavier. SC-VLA is lightweight but limited to residual corrections.

### SC-VLA (2026, Tan et al.)
The similarly-named SC-VLA uses sparse world imagination + RL fine-tuning
of the VLA itself. Our module is:
- Post-hoc (VLA stays frozen)
- VLA-agnostic (works on any transformer VLA)
- Lighter (no world model, just two small MLPs)

### AFIL (2026)
AFIL uses dual action generators (success + failure) for diffusion-based VLAs.
SC-VLA works on any VLA type (autoregressive, diffusion, or direct regression).

## Training Dynamics

### Detector Training
- Supervised: binary cross-entropy on success/failure labels
- Data: collected from VLA rollouts in simulation
- Converges quickly (~20 epochs) because the signal is clear

### Corrector Training
- RL (PPO): the corrector outputs a residual, gets rewarded for task success
- The corrector is tiny (~5K params), so PPO converges in ~500 episodes
- Exploration noise decreases over time (standard PPO schedule)

## Limitations

1. **Can't fix fundamental errors**: If the VLA identifies the wrong object,
   no residual correction will help.

2. **Simulation-to-real gap**: The corrector is trained in simulation.
   Sim-to-real transfer requires domain randomization.

3. **Adds inference latency**: The detector + corrector add ~1-2ms per step
   (tiny MLP forward pass). Negligible for most control frequencies.

4. **Task-specific training**: The corrector needs to be retrained for each
   new environment. The detector may transfer better (success/failure
   features are more general).
