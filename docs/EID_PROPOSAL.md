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

## Minimal Experiment

### Setup
- **Robots:** Franka Panda (7-DOF) + xArm6 (6-DOF) — different DOFs, same gripper type
- **Held-out robot:** WidowX (4-DOF) — for transfer test
- **Environment:** ManiSkill3 (SAPIEN simulator, supports all three robots, ground-truth contact data)
- **Task:** Push object to target pose (3 objects: cube, cylinder, sphere)
- **Data:** 1000 demonstrations per robot via scripted policy + noise

### Protocol
1. Collect demonstrations from Franka + xArm6
2. Train contact estimators for both (supervised in sim)
3. Train shared dynamics model on pooled data
4. Evaluate: does shared dynamics model predict equally well for both robots?
5. Transfer test: train WidowX contact estimator on 50 demos. Success rate?

### Baselines
- Per-robot dynamics (same architecture, separate weights)
- Task-space dynamics (end-effector pose → object pose, fails for different morphologies)
- Raw VLA (no dynamics model, just joint-space policy)
- UniAct-style universal action space (no dynamics)

### Metrics
- Object pose prediction MSE (dynamics accuracy)
- Task success rate (end-to-end)
- Cross-embodiment transfer gap (shared vs per-robot)
- Adapter sample efficiency (demos needed for 80% baseline)

### Timeline
- Days 1-2: Set up ManiSkill3 environments, collect data
- Days 3-4: Implement contact estimators + dynamics model
- Day 5: Train and evaluate
- Day 6: Transfer experiment
- Day 7: Write up results

### Success Criteria
- H1: ≤10% prediction error degradation → dynamics invariance holds
- H2: ≥80% success with 50 demos → transfer efficiency validated
- If BOTH hold → paper is viable
- If EITHER fails → learn exactly where invariance breaks → still a valuable negative result

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
