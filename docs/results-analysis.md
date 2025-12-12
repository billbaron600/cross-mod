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

# Results / Analysis

**Tasks:** the 8 RLBench tasks we evaluate on

<div class="stack section method-media method-media--wide">
  <img src="assets/img/rlbench_tasks.png"
       alt="RLBench tasks"
       loading="lazy">
</div>

**Metric:** task success rate (fraction of successful rollouts)

<div class="section">
  <div class="table-wrap">
    <table class="metrics">
      <caption>Simulation Results on RLBench (transposed)</caption>
      <thead>
        <tr>
          <th>Task</th>
          <th>CrossInstruct</th>
          <th>VLM-Reasoning</th>
          <th>Pure RL (SAC)</th>
          <th>Pure RL (TD3)</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>basketball</td>
          <td>0.90</td>
          <td>0.00</td>
          <td>0.00</td>
          <td>0.00</td>
        </tr>
        <tr>
          <td>peg</td>
          <td>0.25</td>
          <td>0.20</td>
          <td>0.00</td>
          <td>0.00</td>
        </tr>
        <tr>
          <td>close drawer</td>
          <td>0.90</td>
          <td>0.45</td>
          <td>0.95</td>
          <td>0.40</td>
        </tr>
        <tr>
          <td>slide block</td>
          <td>0.90</td>
          <td>0.20</td>
          <td>0.10</td>
          <td>0.00</td>
        </tr>
        <tr>
          <td>jenga</td>
          <td>0.55</td>
          <td>0.00</td>
          <td>0.00</td>
          <td>0.00</td>
        </tr>
        <tr>
          <td>lift block</td>
          <td>0.95</td>
          <td>0.00</td>
          <td>0.00</td>
          <td>0.00</td>
        </tr>
        <tr>
          <td>rubbish</td>
          <td>1.00</td>
          <td>0.00</td>
          <td>0.00</td>
          <td>0.00</td>
        </tr>
        <tr>
          <td>push button</td>
          <td>0.95</td>
          <td>0.30</td>
          <td>0.05</td>
          <td>0.00</td>
        </tr>
      </tbody>
    </table>
  </div>
</div>


  <p class="text">
    <em>Table I</em> in the paper (p. 6) reports these success rates across eight RLBench tasks. CrossInstruct substantially outperforms the reasoning‑only variant and pure RL on precision‑sensitive tasks (e.g., Basketball‑in‑Hoop, Push Button), where pixel‑accurate keypoints prevent small misalignments that otherwise cause failure. (§V‑E; Fig. 8 for a spatially accurate basketball rollout.) :contentReference[oaicite:13]{index=13}
  </p>

  <p class="text">
    The one notable exception is Close Drawer, a short‑horizon task with modest precision needs, where exploration‑driven RL (SAC) can occasionally succeed from scratch. In scenes with bright, simple cues (e.g., square peg/hole with clear contrast), the reasoning‑only variant can be competitive, but it degrades in clutter or color ambiguity (see Fig. 9–10 for typical failure modes without precision coupling). (§V‑E.) :contentReference[oaicite:14]{index=14}
  </p>

  <p class="text">
    <strong>Generalization to the real world.</strong> CrossInstruct transfers from sketched instructions collected in different scenes to hardware with different embodiment/kinematics (Place Cups; Saw Block), honoring abstract intent like color‑matching and “repeat 3×” without closed‑loop replanning. See Fig. 11 (p. 7). (§V‑D, §V‑E.) :contentReference[oaicite:15]{index=15}
  </p>

  <p class="text">
    <strong>RL warm‑start.</strong> Sampling from the trajectory distribution <em>p(τ)</em> provides diverse, semantically consistent demonstrations for TD3+BC. On Jenga (sparse binary reward), policies initialized from CrossInstruct achieve ~90% success within 40k steps, whereas training from scratch fails to obtain non‑zero return; on Peg, learning is significantly accelerated compared to TD3/SAC/PPO from scratch (see Fig. 12, p. 7; §V‑F). :contentReference[oaicite:16]{index=16}
  </p>
</div>
