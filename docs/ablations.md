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

<h1>Ablations &amp; Comparisons</h1>

<p class="text">
This page isolates the contribution of <strong>hierarchical precision coupling</strong>—the mechanism that anchors high-level
reasoning to <em>pixel-accurate</em> spatial grounding—by removing it and measuring what breaks.
</p>

<h3>Main ablation: removing hierarchical precision coupling</h3>

<p class="text">
<strong>CrossInstruct (full system)</strong> separates “what to do” from “where exactly.” A large reasoning VLM interprets the
sketch/text intent, then a dedicated pointing model (Molmo) returns precise 2D keypoints. Those keypoints are fed back to
the reasoning model to anchor multi-view trajectory sketches before lifting them into 3D.
</p>

<ul class="text">
  <li>
    <strong>CrossInstruct:</strong> reasoning + Molmo keypoints → keypoint-anchored 2D sketches (multi-view) → 3D lifting → executable motion.
  </li>
  <li>
    <strong>VLM‑Reasoning (no precision coupling):</strong> remove Molmo; the reasoning model must both reason <em>and</em> draw the
    trajectories / output end-effector poses directly over images. Everything else is kept the same (same demonstrations,
    same evaluation seeds, same downstream pipeline).
  </li>
</ul>

<p class="text">
<strong>Why this is a clean ablation:</strong> the only change is whether trajectory sketches are anchored by pixel-level keypoints.
This isolates the effect of precision coupling from the rest of the pipeline.
</p>

<h3>Baselines for context (not ablations)</h3>

<p class="text">
To contextualize instruction-driven behavior synthesis, we also report pure RL baselines—<strong>TD3</strong> and <strong>SAC</strong>—
trained from scratch with <strong>sparse rewards</strong> and a fixed <strong>1M environment-step</strong> budget per task (no behavior-cloning priors).
These measure how far exploration-only learning gets without cross-modal supervision.
</p>

<h3>Evaluation protocol</h3>

<p class="text">
All methods are evaluated on <strong>RLBench</strong> with <strong>20 held‑out random seeds per task</strong>. The metric is
<strong>task success rate</strong> (fraction of successful rollouts). We additionally run qualitative real‑world tests under
domain + embodiment shift.
</p>

<h3>What changes when you remove coupling?</h3>

<p class="text">
Removing precision coupling produces trajectories that are often <em>nearly</em> correct, but spatially offset in ways that are
fatal for precision tasks (e.g., the robot under-reaches the target interaction point). Failures are most common in cluttered
or low‑contrast scenes and when multiple objects share similar visual cues (e.g., similarly-colored distractors).
</p>

<p class="text">
See <em>Fig. 9</em> for under-reach / misalignment failures (button, basketball, Jenga), and <em>Fig. 10</em> for distractor grounding
when colors are ambiguous (peg vs. similarly colored objects).
</p>

<h3>Quantitative snapshot</h3>

<p class="text">
RLBench success rate (higher is better). Values are averaged over held‑out seeds.
</p>

<div class="tables">
  <table class="results">
    <thead>
      <tr>
        <th>Method</th>
        <th>Basketball‑in‑Hoop</th>
        <th>Square Block on Peg</th>
        <th>Close Drawer</th>
        <th>Slide Block to Target</th>
      </tr>
    </thead>
    <tbody>
      <tr><td>CrossInstruct</td><td>0.90</td><td>0.25</td><td>0.90</td><td>0.90</td></tr>
      <tr><td>VLM‑Reasoning</td><td>0.00</td><td>0.20</td><td>0.45</td><td>0.20</td></tr>
      <tr><td>Pure RL (SAC)</td><td>0.00</td><td>0.00</td><td>0.95</td><td>0.10</td></tr>
      <tr><td>Pure RL (TD3)</td><td>0.00</td><td>0.00</td><td>0.40</td><td>0.00</td></tr>
    </tbody>
  </table>

  <table class="results">
    <thead>
      <tr>
        <th>Method</th>
        <th>Play Jenga</th>
        <th>Lift Numbered Block</th>
        <th>Put Rubbish in Bin</th>
        <th>Push Button</th>
      </tr>
    </thead>
    <tbody>
      <tr><td>CrossInstruct</td><td>0.55</td><td>0.95</td><td>1.00</td><td>0.95</td></tr>
      <tr><td>VLM‑Reasoning</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.30</td></tr>
      <tr><td>Pure RL (SAC)</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.05</td></tr>
      <tr><td>Pure RL (TD3)</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr>
    </tbody>
  </table>
</div>

<h3>What we did not ablate</h3>

<p class="text">
We keep the rest of the pipeline fixed and focus the ablation budget on the precision-coupling interface, because it is the
primary connection between high-level reasoning and pixel-level grounding.
</p>

