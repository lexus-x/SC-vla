# Literature Search Results: Embodiment-Invariant Latent Dynamics

## What Already Exists

| Paper | Date | Shared Dynamics? | Morphology Scope | Key Limitation |
|-------|------|-----------------|-----------------|----------------|
| **OPFA** (ICRA 2026) | Mar 2026 | Latent actions, unified decoder | 11 end-effectors | Policy-level, not dynamics-level |
| **UniAct** (CVPR 2025) | Jan 2025 | Universal action space | Multiple robots | Per-embodiment decode heads needed |
| **He et al.** (world models) | Nov 2025 | **YES** — particle dynamics | Dexterous hands only | Only hands, not arbitrary morphologies |
| **UniVLA** (RSS 2025) | May 2025 | Latent actions from video | Cross-embodiment | Policy, not dynamics |
| **X-VLA** (ICLR 2026) | Oct 2025 | Soft prompts | Cross-embodiment | No shared dynamics model |
| **Data Analogies** (CoRL 2026) | Mar 2026 | N/A (data method) | Morphology transfer | Proves paired demos matter, not a model |
| **GCNT** (2025) | May 2025 | Graph policy | Modular locomotion | Not manipulation |

## The Honest Gap

All three properties simultaneously:

1. ✅ **Learned dynamics model** (not just policy) — He et al. does this
2. ✅ **Embodiment-invariant** — He et al. does this
3. ❌ **Works across fundamentally different morphologies** — NOBODY does this

He et al. is closest but limited to dexterous hands. They explicitly conjecture
"environment dynamics are embodiment-invariant" but only prove it for hand manipulation.

**Nobody has shown a dynamics model that works across, say, a Franka arm AND a
dexterous hand AND a mobile manipulator — fundamentally different kinematics.**

## The Specific Unsolved Question

"If I apply a 6D wrench (force/torque) at a contact point on an object,
the object's motion is determined by physics, not by which robot applied the wrench.
Can we learn this mapping from robot interaction data across diverse morphologies?"

This is NOT a policy question. It's a dynamics question.
Current VLAs learn policies (observation → action).
Nobody learns embodiment-invariant dynamics (action + state → next state) that transfer.
