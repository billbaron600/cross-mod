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

# Method

## Diagrammatic Sketches Over Camera Views
<p class="text">
We use human <em>cross‑modal instructions</em>—rough freehand sketches plus short textual notes—drawn directly over a pair of calibrated camera views. From these scribbles, the system first produces <strong>2D end‑effector trajectories</strong> in each view and then <strong>lifts them into 3D</strong>. Each per‑view curve is treated as a time‑indexed Gaussian corridor in pixel space; rays are cast through pixels in both cameras and intersected to localize a distribution of feasible <strong>3D waypoints</strong> per time step. The mean yields an executable centerline; sampling the distribution yields diverse rollouts for downstream learning. This preserves sketch geometry while honoring user‑indicated collision hints. See Fig. 4 (2D curves to 3D waypoints) in the paper. :contentReference[oaicite:3]{index=3}
</p>

<div class="grid grid-2 section">
  <img src="assets/img/play-jenga-sketched-demo.png" alt="Play Jenga sketched demo" loading="lazy">
  <img src="assets/img/rubish-in-bin-sketched-demo.png" alt="Rubbish in Bin sketched demo" loading="lazy">
</div>

## Task Identification, Precision Points, Sketching, and Action Commands
<p class="text">
Our pipeline couples a <strong>reasoning VLM</strong> with a fine‑tuned <strong>Molmo</strong> pointing model. The reasoning model performs task identification and high‑level planning, proposes semantic <em>keypoint descriptors</em> (e.g., “button center,” “rim edge”), and drafts the rough motion sketch. The pointing model converts each descriptor into precise pixel coordinates in both views. These precise keypoints are fed back to the reasoning model, which refines the per‑view sketch and outputs <strong>3D waypoints + end‑effector orientations + gripper open/close</strong> commands. See the hierarchical precision coupling in Fig. 3 and §IV‑A. :contentReference[oaicite:4]{index=4}
</p>

<div class="stack section">
  <img src="assets/img/systems-diagram.png" alt="System diagram" loading="lazy">
  <img src="assets/img/pointing-commands-and-sketching.png" alt="Pointing commands and sketching" loading="lazy">
</div>

## Hierarchical Precision Coupling
<p class="text">
We integrate a large reasoning VLM <em>R</em> with a small, fine‑tuned pointing model <em>G</em>. <em>R</em> proposes semantic keypoint descriptors and keeps the global task context; <em>G</em> resolves them to pixel‑accurate points in each view. The updated context then guides <em>R</em> to produce two per‑view 2D trajectories, which are lifted to a time‑indexed 3D waypoint distribution. Finally, <em>R</em> specifies EE orientations and gripper states along the path. (§IV‑A; Fig. 3.) :contentReference[oaicite:5]{index=5}
</p>

## Multi‑View Lifting via Ray Casting
<p class="text">
Each per‑view curve defines a Gaussian over pixels at time <em>t</em>; casting calibrated camera rays through both views and intersecting those density “tubes” yields a set of plausible 3D points <em>S<sub>t</sub></em>. We fit a Gaussian <em>N(μ<sub>t</sub>, Σ<sub>t</sub>)</em> to obtain the waypoint distribution at time <em>t</em>. The mean trajectory <em>E[τ]</em> can be tracked directly; samples from <em>p(τ)</em> seed robust policy learning. (§IV‑B; Fig. 5 illustrates intersecting rays.) :contentReference[oaicite:6]{index=6}
</p>

## Execution and RL Warm‑Start
<p class="text">
We track <em>E[τ]</em> with a lightweight planner (IK with singularity mitigation) or reactive motion primitives, and we also use samples from <em>p(τ)</em> to build a dataset for TD3+BC—both for pre‑training and persistent BC regularization during RL. (§IV‑C.) :contentReference[oaicite:7]{index=7}
</p>


