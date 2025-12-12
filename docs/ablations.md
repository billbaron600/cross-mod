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
<strong>(1) CrossInstruct (full system)</strong> separates “what to do” from “where exactly.” A large reasoning VLM interprets the
sketch/text intent, then a dedicated pointing model (Molmo) returns precise 2D keypoints. Those keypoints are fed back to
the reasoning model to anchor multi-view trajectory sketches before lifting them into 3D.
</p>

<div class="stack section method-media method-media--wide">
  <img src="assets/img/systems-diagram.png"
       alt="systems diagram"
       loading="lazy">
</div>

<p class="text">
<strong>(2) VLM‑Reasoning (no precision coupling):</strong> remove Molmo; the reasoning model must both reason <em>and</em> draw the
trajectories / output end-effector poses directly over images. Everything else is kept the same (same demonstrations,
same evaluation seeds, same downstream pipeline).
</p>
<div class="stack section method-media method-media--wide">
  <img src="assets/img/main-vlm-ablation.png"
       alt="Main vlm ablation"
       loading="lazy">
</div>



<p class="text">
<strong>Why this is a clean ablation:</strong> the only change is whether trajectory sketches are anchored by pixel-level keypoints.
This isolates the effect of precision coupling from the rest of the pipeline.
</p>

<h3>Baselines for context (not ablations)</h3>

<p class="text">
We include <strong>TD3</strong> and <strong>SAC</strong> to answer a simple question:
<strong>how far can standard, exploration-driven reinforcement learning get on these RLBench tasks without any cross-modal supervision?</strong>
These baselines are not meant to compete on the same supervision signal. They are a reference point for what “from-scratch RL” achieves under the same task and compute budget.
</p>

<ul class="text">
  <li>
    <strong>Why TD3:</strong> a widely used off-policy actor critic for continuous control.
    It uses a <em>deterministic</em> policy and stabilizes learning with techniques like clipped double Q-learning and delayed policy updates.
    This is a strong baseline for sample-efficient, exploitation-focused learning.
  </li>
  <li>
    <strong>Why SAC:</strong> another widely used off-policy actor critic, but with a <em>stochastic</em> policy and entropy regularization.
    This makes it a strong baseline for exploration and robustness, which matters when rewards are sparse.
  </li>
  <li>
    <strong>Why both:</strong> together they bracket two common RL regimes:
    deterministic policies (TD3) versus entropy-regularized stochastic policies (SAC).
  </li>
</ul>

<p class="text"><strong>Training setup (kept intentionally standard):</strong></p>
<ul class="text">
  <li><strong>Reward:</strong> sparse binary success signal.</li>
  <li><strong>Budget:</strong> 1M environment steps per task.</li>
  <li><strong>No priors:</strong> standard implementations, without behavior cloning or cross-modal initialization.</li>
</ul>


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

