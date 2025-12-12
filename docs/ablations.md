<link rel="stylesheet" href="assets/css/site.css">

<nav class="topbar">
  <div class="links">
    <a href="index.html">Background</a>
    <a href="motivation.html">Motivation</a>
    <a href="method.html">Method</a>
    <a href="video-rollouts.html">Video Rollouts</a>
    <a class="active" href="ablations.html">Experiments/Ablations</a>
    <a href="results-analysis.html">Results / Analysis</a>
    <a href="rl-refinement-future.html">RL Refinement / Future Work</a>
    <a href="https://arxiv.org/abs/2509.21107" target="_blank" rel="noopener">Paper</a>
    <a href="https://github.com/billbaron600/cross-mod/tree/main" target="_blank" rel="noopener">Code</a>
  </div>
</nav>

<h1>Experiments, Ablations &amp; Baselines</h1>

<p class="text">
This page isolates the contribution of <strong>hierarchical precision coupling</strong>—the mechanism that anchors high-level
reasoning to <em>pixel-accurate</em> spatial grounding—by removing it and measuring what breaks.
We also include standard RL baselines (TD3, SAC) as context for what “from-scratch RL” achieves under a comparable budget.
</p>

<ul class="text">
  <li><strong>Tasks:</strong> RLBench manipulation tasks.</li>
  <li><strong>Instruction-driven methods:</strong> <strong>1</strong> sketched demonstration per task, evaluated on <strong>20</strong> held-out seeds (no task fine-tuning).</li>
  <li><strong>Main ablation:</strong> remove pixel-level keypoint anchoring (precision coupling).</li>
  <li><strong>RL baselines:</strong> TD3/SAC trained with sparse reward for <strong>1M</strong> environment steps per task, then evaluated on <strong>20</strong> held-out seeds.</li>
  <li><strong>Metric:</strong> success rate over 20 held-out evaluation seeds.</li>
</ul>

<h2>RLBench tasks</h2>

<div class="stack section method-media method-media--wide">
  <img src="assets/img/rlbench_tasks.png"
       alt="RLBench tasks"
       loading="lazy">
</div>

<h2>Experiment setup (what we run)</h2>

<p class="text">
We evaluate <strong>generalization</strong> across randomized RLBench task instances. For each task, results are reported as
<strong>success rate</strong> over <strong>20 held-out evaluation seeds</strong> (one rollout per seed).
</p>

<h3>Instruction-driven methods (CrossInstruct and the ablated variant)</h3>
<ul class="text">
  <li><strong>No task fine-tuning:</strong> no training or gradient updates on the target task.</li>
  <li>
    <strong>Single demonstration seed:</strong> each task provides <strong>one</strong> sketched demonstration from a single seed.
    The demonstration exists only as a <strong>sketch over images</strong> (plus the instruction).
  </li>
  <li>
    <strong>Held-out testing:</strong> evaluate on <strong>20 unseen seeds</strong> for that task to test transfer under object pose and scene variation.
  </li>
</ul>

<h2>Main ablation: removing hierarchical precision coupling</h2>

<p class="text">
<strong>What “precision coupling” means here:</strong> high-level reasoning decides <em>what</em> to do, while a dedicated pointing model supplies
<em>where exactly</em> (pixel-accurate 2D keypoints). Those keypoints are then fed back to anchor trajectory sketches before lifting to 3D.
</p>

<h3>(1) CrossInstruct (full system)</h3>

<p class="text">
A large reasoning VLM interprets sketch/text intent, then a dedicated pointing model (Molmo) returns precise 2D keypoints.
Those keypoints are fed back to the reasoning model to anchor multi-view trajectory sketches before lifting them into 3D.
</p>

<div class="stack section method-media method-media--wide">
  <img src="assets/img/systems-diagram.png"
       alt="systems diagram"
       loading="lazy">
</div>

<h3>(2) VLM-Reasoning (no precision coupling)</h3>

<p class="text">
Remove Molmo; the reasoning model must both reason <em>and</em> directly draw trajectories / output end-effector poses over images.
Everything else is kept the same (same demonstrations, same evaluation seeds, same downstream pipeline).
</p>

<div class="stack section method-media method-media--wide">
  <img src="assets/img/main-vlm-ablation.png"
       alt="Main vlm ablation"
       loading="lazy">
</div>

<p class="text">
<strong>Purpose of the ablation:</strong> the only change is whether trajectory sketches are anchored by pixel-level keypoints.
This isolates the effect of precision coupling from the rest of the pipeline.
</p>

<ul class="text">
  <li><strong>Held fixed:</strong> demonstrations, evaluation seeds, downstream 3D lifting and execution pipeline.</li>
  <li><strong>Varied:</strong> whether we explicitly inject pixel-accurate keypoints to anchor the sketch/plan.</li>
</ul>

<h2>Baselines for context (not ablations): from-scratch RL</h2>

<p class="text">
We include <strong>TD3</strong> and <strong>SAC</strong> to answer:
<strong>how far can standard exploration-driven reinforcement learning get on these RLBench tasks without cross-modal supervision?</strong>
These baselines are not meant to compete on the same supervision signal. They are a reference point for what “from-scratch RL”
achieves under the same task and compute budget.
</p>

<ul class="text">
  <li>
    <strong>TD3:</strong> widely used off-policy actor-critic for continuous control (deterministic policy), stabilizing learning via
    clipped double Q-learning and delayed policy updates.
  </li>
  <li>
    <strong>SAC:</strong> widely used off-policy actor-critic (stochastic policy) with entropy regularization, typically improving exploration and robustness
    under sparse rewards.
  </li>
  <li>
    <strong>Why both:</strong> together they bracket two common RL regimes: deterministic policies (TD3) vs. entropy-regularized stochastic policies (SAC).
  </li>
</ul>

<p class="text"><strong>Training setup (kept intentionally standard):</strong></p>
<ul class="text">
  <li><strong>Reward:</strong> sparse binary success signal.</li>
  <li><strong>Budget:</strong> 1M environment steps per task.</li>
  <li><strong>No priors:</strong> standard implementations, without behavior cloning or cross-modal initialization.</li>
  <li><strong>Evaluation:</strong> success rate on 20 held-out seeds after training.</li>
</ul>

<h2>Metric &amp; qualitative checks</h2>

<ul class="text">
  <li>
    <strong>Primary metric:</strong> <strong>success rate</strong> = percentage of successful rollouts among the <strong>20 held-out evaluation seeds</strong>
    (one rollout per seed).
  </li>
</ul>

<p class="text">
We also include qualitative real-world tests under domain and embodiment shift (different arm morphology and real sensor noise)
to assess whether the behaviors remain plausible beyond simulation.
</p>
