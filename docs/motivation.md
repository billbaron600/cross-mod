<link rel="stylesheet" href="assets/css/site.css">

<nav class="topbar">
  <a class="brand" href="index.html">Cross‑Mod</a>
  <div class="links">
    <a href="motivation.html">Motivation</a>
    <a href="method.html">Method</a>
    <a href="video-rollouts.html">Video Rollouts</a>
    <a href="ablations.html">Ablations</a>
    <a href="results-analysis.html">Results / Analysis</a>
    <a href="rl-refinement-future.html">RL Refinement / Future Work</a>
    <a href="https://arxiv.org/abs/2509.21107" target="_blank" rel="noopener">Paper</a>
    <a href="https://github.com/billbaron600/cross-mod/tree/main" target="_blank" rel="noopener">Code</a>
  </div>
</nav>

# Motivation

## Dual‑Process Theory as a Design Lens

For decades, dual‑process theory has characterized human cognition as an interaction between fast, automatic heuristics (**System 1**) and slower, deliberate reasoning (**System 2**). In modern robotics/AI vernacular, large reasoning models—LLMs/VLMs—are often described as “System‑2 thinkers,” planning explicitly and composing symbolic steps before acting (e.g., work inspired by Noam Brown’s “System 2” framing; robotics systems like VoxPoser, HAMSTER, Hi‑Robot, and MOKA that use reasoning models for high‑level planning).

Most existing pipelines hand off from this high‑level planner to fairly **traditional actuation layers**:
- **Planner/value‑field → controller** (e.g., VoxPoser generates 3D value maps for MPC),
- **Grasp/waypoint templates + IK sampling** (e.g., MOKA),
- **RL policies** trained end‑to‑end (e.g., HAMSTER),
- **Vision‑Language‑Action** policies (e.g., Hi‑Robot).

These are powerful, but they insert a **thick interface** between reasoning and motion: the planner reasons abstractly, while the controller or learned policy attempts to realize that intent.

### Our premise
**Give the System‑2 reasoner a more direct path to actuation.** In CrossInstruct, the reasoning VLM:
1) infers the task and decomposes it into steps,  
2) proposes *semantic high‑precision points* (descriptors),  
3) issues **pointing commands** to a fine‑tuned model (Molmo) for pixel‑accurate keypoints in each calibrated view,  
4) uses those keypoints as scaffolds to **sketch per‑view trajectories**,  
5) **lifts the sketches into 3D** via multi‑view geometry, and  
6) outputs **end‑effector orientations and gripper open/close** along the 3D path, which a lightweight planner tracks (IK with singularity mitigation, no collision optimization).  
This makes the reasoner part of the *actuation loop*, not just the task planner. (See the framework and examples in **Fig. 3–4** of the paper for the precision‑coupled pipeline and 2D→3D lifting.) :contentReference[oaicite:1]{index=1}

### Why suppress “System 1” early?
In many human sensorimotor settings, the automatic controller (S1) must be **actively suppressed** so a rule‑based controller (S2) can succeed, before later re‑automatization:

- **Laparoscopic fulcrum effect (manipulation).** Handle‑left → tip‑right; S1 “move hand toward target” amplifies error. S2 must apply the explicit inversion rule until it’s proceduralized.
- **“Backwards bicycle” / reversed steering (locomotion).** Reflexive counter‑steer becomes catastrophic; S2 takes over with slow rule‑based corrections until a new S1 policy is cached.
- **Mirror‑drawing / anti‑reach (manipulation).** S1 chases visual error; S2 uses symbolic re‑aiming (“aim left by θ”) to be slow‑but‑correct.
- **Split‑belt treadmill (locomotion).** S2 imposes an asymmetric gait before automaticity returns.

**Robotics rhyme:** early in deployment, let S2 (the reasoner) output explicit **waypoints/orientations/gripper commands** that are tracked directly; later, those rollouts can **distill** into a faster S1‑like policy (RL fine‑tuning/BC), matching how skills re‑automatize in humans. CrossInstruct’s distribution over trajectories is expressly designed to seed TD3+BC for that distillation. (See §IV‑C and §V‑F.) :contentReference[oaicite:2]{index=2}
