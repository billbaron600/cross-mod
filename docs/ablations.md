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

# Ablations & Comparisons

<p class="text">
<strong>What we ablate:</strong> the <em>hierarchical precision coupling</em>. We compare CrossInstruct (reasoning VLM + Molmo pointing) against a <strong>VLM‑Reasoning (no precision coupling)</strong> variant that removes Molmo and asks the reasoning model to both reason and draw trajectories directly over images. This isolates the contribution of pixel‑accurate keypoints anchoring the sketches. (§V‑C; Fig. 8–10.) :contentReference[oaicite:8]{index=8}
</p>

<p class="text">
<strong>What we compare (not ablate):</strong> we also report pure RL baselines—<strong>TD3</strong> and <strong>SAC</strong>—trained from scratch with sparse rewards and a 1M‑step budget per task. These measure how far exploration‑driven policies can get without cross‑modal supervision. (§V‑C.) :contentReference[oaicite:9]{index=9}
</p>

<h3>Findings from the ablation</h3>
<p class="text">
Removing precision coupling leads to small but fatal spatial misalignments in precision tasks (Basketball‑in‑Hoop, Push Button, Jenga). With coupling, Molmo’s keypoints stabilize the sketches and improve spatial grounding, especially in cluttered or low‑contrast scenes where object boundaries are ambiguous. See <em>Fig. 9</em> for under‑reach on button/ball/Jenga without coupling, and <em>Fig. 10</em> for color confusion in Peg when the reasoning‑only variant grounds to a similarly colored distractor. (§V‑E.) :contentReference[oaicite:10]{index=10}
</p>

<h3>Evaluation protocol</h3>
<p class="text">
All methods are evaluated on <strong>RLBench</strong> tasks with 20 held‑out random seeds per task (Panda in sim), plus qualitative real‑world tests on a morphologically different arm. Performance metric: <em>task success rate</em> (fraction of successful rollouts). (§V‑A, §V‑D.) :contentReference[oaicite:11]{index=11}
</p>

<h3>What we do <em>not</em> ablate in the paper</h3>
<p class="text">
We do <em>not</em> report a single‑view vs. multi‑view lifting ablation, nor ablations on the distributional vs. deterministic trajectory representation. The core ablation focuses on the precision‑coupling module because it is the primary architectural lever connecting System‑2 reasoning to pixel‑level grounding. (See §IV for method components and §V for experiments.) :contentReference[oaicite:12]{index=12}
</p>
