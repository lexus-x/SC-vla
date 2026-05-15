# EID: Embodiment-Invariant Dynamics for Vision-Language-Action Models

## One-Sentence Thesis

**Environment dynamics are robot-invariant: the physics of "what happens to an object when it is contacted" is determined by the contact interaction itself—not by which robot produced it—and a VLA that learns dynamics in this contact-centric, embodiment-invariant space will transfer across morphologies without per-robot fine-tuning.**

---

## Why This Is Novel

The field in 2025–2026 has three approaches to cross-embodiment:

| Approach | Example | Shares perception? | Shares dynamics? | Cross-morphology? |
|----------|---------|-------------------|-----------------|-------------------|
| Co-training + per-robot heads | RT-X, Octo | ✅ | ❌ | Partial |
| Universal action spaces | UniAct, OPFA | ✅ | ❌ (shared action, not dynamics) | ✅ |
| Cross-embodiment world models | He et al. 2025 | ✅ | ✅ | ❌ (hands only) |
| **EID (proposed)** | — | ✅ | **✅** | **✅** |

**Nobody has demonstrated a learned dynamics model that is both embodiment-invariant AND works across fundamentally different morphologies (arms, hands, mobile manipulators).**

- He et al. (Nov 2025) conjecture "environment dynamics are embodiment-invariant" but only prove it for dexterous hands using particle representations.
- UniAct (CVPR 2025) and OPFA (ICRA 2026) share action spaces but learn no shared dynamics.
- UniVLA (RSS 2025) learns latent actions from video but not dynamics.

---

## The Mechanism: Contact-Point Dynamics (CPD)

### Core Insight

When ANY robot manipulates an object, the physics is determined by **three things only**:
1. Where on the object surface contact occurs
2. What wrench (force/torque) is applied at each contact
3. The object's own state (pose, shape, mass, friction)

The robot's joint configuration, link lengths, and kinematic structure are **irrelevant** to the object dynamics once contact is established.

### Representation

```
Contact-Point State: C ∈ ℝ^(N×9)
  N = max 8 contact points
  Each row: [contact_position(3), applied_force(3), contact_normal+slip(3)]

Object State: z_obj ∈ ℝ^128
  Learned latent from vision encoder (ViT → object embedding)
```

**Why this is embodiment-invariant**: A 7-DOF arm making one contact, a 16-DOF hand making three contacts, and a mobile manipulator making two contacts all populate different rows of the SAME matrix C. The dynamics model doesn't know or care how the contacts were achieved.

### Architecture

```
                    ┌─────────────────────────────┐
                    │   Shared Dynamics Model      │
                    │   (Transformer, ~1M params)  │
                    │                              │
                    │  tokens = [z_obj; C₁...C₈]  │
                    │  ↓                           │
                    │  TransformerEncoder(6 layers) │
                    │  ↓                           │
                    │  Δz_obj = MLP(object_token)  │  ← predicts object state change
                    │  C' = MLP(contact_tokens)    │  ← predicts contact evolution
                    │  wrench = MLP(object_token)  │  ← net wrench (F=ma check)
                    └──────────────┬───────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
     ┌────────▼────────┐  ┌───────▼───────┐  ┌────────▼────────┐
     │  Franka Panda   │  │  Allegro Hand │  │  Stretch RE2    │
     │  (7-DOF arm)    │  │  (16-DOF)     │  │  (mobile manip) │
     │                 │  │               │  │                 │
     │  Contact        │  │  Contact      │  │  Contact        │
     │  Estimator      │  │  Estimator    │  │  Estimator      │
     │  (tiny MLP)     │  │  (tiny MLP)   │  │  (tiny MLP)     │
     └─────────────────┘  └───────────────┘  └─────────────────┘
     
     ══════════════════════════════════════════════════════════════
     Dynamics model: 0 embodiment-specific parameters
     Per-robot: only the contact estimator (~5K params)
     New robot = train contact estimator only. Dynamics transfer zero-shot.
     ══════════════════════════════════════════════════════════════
```

### Training

**Stage 1: Contact Estimation (per-robot, supervised in simulation)**
- Small MLP: `(joint_state, joint_velocities, joint_torques) → C ∈ ℝ^(N×9)`
- Supervised with ground-truth contact data from physics engine
- ~1 hour of training per robot

**Stage 2: Dynamics Model (shared, trained on multi-robot data)**
- Loss = MSE(Δz_pred, Δz_gt) + MSE(C_pred, C_gt) + λ·L_physics
- L_physics: enforce F=ma consistency in latent space
- Trained on pooled data from all robots (batched, balanced)
- ~10 hours on single GPU

**Stage 3: VLA Integration (contact target head)**
- New MLP head on VLA: visual features → target contact state C_target
- CPD model refines C_target → refined action
- Trained end-to-end with task success loss

### Transfer to New Robot

```
1. Freeze dynamics model (zero-shot)
2. Train new robot's contact estimator (~1 hour, simulation)
3. Run
```

No retraining of the dynamics model. No retraining of the VLA.

---

## Measurable Hypotheses

**H1 (Dynamics Invariance):** A dynamics model trained jointly on Franka Panda (7-DOF arm) and Allegro Hand (16-DOF dexterous hand) data achieves ≤10% object state prediction error degradation when evaluated on tasks executed by either robot, compared to per-robot dynamics baselines.

**H2 (Transfer Efficiency):** A new robot (UR5e, 6-DOF) achieves ≥80% of its single-robot baseline success rate after training ONLY its contact estimator on 50 demonstrations, while the dynamics model and VLA remain frozen.

**H3 (Zero-Shot Object Generalization):** The dynamics model, trained on pick-and-place with cubes and cylinders, achieves ≥60% success rate on a novel object (sphere) when used with any of the three robots.

---

## The Minimal Experiment

### Core Claim (falsifiable)
Object dynamics during pushing are a function of (object state, task-space interaction) — NOT of which robot arm is doing the pushing. If true, a single dynamics model should predict object motion from both robots with zero robot-specific adaptation.

### Setup
- **Robots:** Franka Panda (7-DOF) + UR5e (6-DOF) — different joint count, different kinematics, different workspace shapes, but both terminate in a parallel gripper pressing against a table.
- **Environment:** ManiSkill3 (GPU-parallelized at 30K+ FPS, supports 20+ robots, same-task-swap-robot via `gym.make(..., robot=...)`, ground-truth object state, contact force data via SAPIEN)
- **Task:** Push cube to a target position. Randomized: initial cube pose, target pose, robot starting configuration.

### Data Collection (~30 min wall-clock)
- 2000 trajectories per robot via scripted pushing policy
- Record per timestep: `(object_pose_7D, ee_pose_7D, ee_delta_3D, gripper_state)`
- ~50 timesteps/trajectory → 100k transitions per robot

### Architecture (ALL shared, zero robot-specific parameters)
```
object_pose ────▶ Obs Encoder ──▶ z_t ∈ ℝ³² (latent object state)
ee_pose ────────▶ (shared)
gripper ────────▶

ee_delta_3D ────▶ Action Encoder ──▶ a ∈ ℝ¹² (latent action)
(task-space)      (shared)

(z_t, a) ───────▶ Dynamics Model ──▶ z_{t+1}
                   (shared)

z_{t+1} ────────▶ Obs Decoder ──▶ object_pose_{t+1}
                   (shared)
```

**Key:** Action = end-effector delta in task space (Δx, Δy, Δz), NOT joint-space commands. The same Δee produces the same object motion regardless of which arm generated it.

### The Three Models (controlled comparison)
| Model | Training Data | What it tests |
|---|---|---|
| M_Franka | Franka only (2000 traj) | Upper bound on Franka |
| M_UR5e | UR5e only (2000 traj) | Upper bound on UR5e |
| M_mixed | Both robots (4000 traj, shuffled) | **The hypothesis: shared dynamics works** |

All three: identical architecture, lr, epochs. Adam, lr=1e-3, 100 epochs, batch 256. ~2 hours on A100.

### Metrics

**1. Transfer Ratio (the money metric)**
```
Transfer Ratio = MSE_mixed→Franka / MSE_Franka→Franka
```
How much performance degrades when using mixed model vs. single-robot model.

**2. Wrong-robot baseline (the floor)**
M_UR5e applied to Franka data. Should FAIL — confirms invariance is non-trivial.

**3. Latent dynamics alignment**
For matched pushing scenarios (same cube pose, same push direction): cosine similarity of latent transitions across robots. If dynamics are invariant: vectors should be parallel.

### Pre-Registered Success Criteria
| Criterion | Threshold | Meaning |
|---|---|---|
| Transfer Ratio | ≤ 1.15 | Mixed model ≤15% worse than single-robot |
| Wrong-robot fails | MSE_UR5e→Franka > 2× MSE_Franka→Franka | Robot-specific models DON'T transfer |
| Latent alignment | Cosine similarity > 0.7 | Shared latent space captures same dynamics |

**Pass:** ALL three met → full-scale training justified.
**Fail:** ANY fails → redirect to robot-specific representations.

### Compute: ~4 hours on 1× A100

### Environment Choice (justified)

| Environment | Same Task × N Robots | GT State | Force/Torque | Speed |
|---|---|---|---|---|
| **ManiSkill3** ✅ | ✅ (built-in swap) | ✅ | ✅ (SAPIEN contacts) | 30K+ FPS |
| Isaac Lab | ✅ (configurable) | ✅ | ✅ (native) | GPU parallel |
| RoboTwin 2.0 | ✅ (5 platforms) | ✅ | ⚠️ | GPU parallel |
| MetaWorld ❌ | ❌ (Sawyer only) | ✅ | ⚠️ | CPU |
| LIBERO ❌ | ❌ (Franka only) | ✅ | ⚠️ | CPU |

ManiSkill3 wins: `gym.make("PickCube-v1", robot="panda")` vs `gym.make("PickCube-v1", robot="ur5e")` — same task, different robots, ground-truth everything.

---

## Honest Assessment

**What's genuinely novel:** The explicit decomposition of dynamics (shared, contact-centric) from kinematics (per-robot, contact estimation) in a VLA context. He et al. proved the concept for hands; we extend it to arbitrary morphologies.

**What's incremental:** "Physics is robot-invariant" is obvious. The contribution is proving it works for VLA transfer, not inventing the concept.

**What I'm uncertain about:**
1. Is contact estimation from joint torque accurate enough? (5% sensor error → noisy contacts)
2. Does the latent action space preserve enough information for precise manipulation?
3. Does ManiSkill3 actually support the same task across different robots with comparable difficulty?

**What would make this foundational:**
If the dynamics model predicts object motion for a robot it has NEVER seen, with only a new contact estimator trained on a handful of demos — that's not "5% better." That's "physics transfers across morphologies."

---

## Existing Work (Sources)

| Paper | Date | Venue | Key Contribution |
|-------|------|-------|-----------------|
| He et al. — Cross-Embodiment World Models | Nov 2025 | arXiv | Particle-based dynamics for dexterous hands. Conjectures "dynamics are embodiment-invariant." |
| UniAct | Jan 2025 | CVPR 2025 | Universal action space with per-embodiment decoders |
| OPFA | Mar 2026 | ICRA 2026 | Geometry-aware latent actions, unified decoder, 11 end-effectors |
| UniVLA | May 2025 | RSS 2025 | Task-centric latent actions from video |
| X-VLA | Oct 2025 | ICLR 2026 | Soft-prompted cross-embodiment |
| Data Analogies (Finn) | Mar 2026 | CoRL 2026 | Paired demos matter for morphology transfer |
| FAST/π₀-FAST | Jan 2025 | RSS 2025 | Universal action tokenizer (DCT-based) |
| RT-X / Open X-Embodiment | Oct 2023 | ICRA 2024 | Co-training on 22 robots, shared perception |
| GCNT | May 2025 | arXiv | Graph-based morphology-agnostic policy (locomotion) |
