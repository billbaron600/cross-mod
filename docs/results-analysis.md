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

<H1> Results </H1>




  <div class="stack section method-media method-media--wide">
    <img src="assets/img/rlbench_tasks.png"
         alt="RLBench tasks"
         loading="lazy">
  </div>


 <div class="table-wrap" style="display:flex;justify-content:center;">
  <table class="metrics">
    <caption>Task Success Rates for the Different Methods</caption>
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
        <td style="background:#d1fae5;">0.90</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.00</td>
      </tr>
      <tr>
        <td>peg</td>
        <td style="background:#d1fae5;">0.25</td>
        <td>0.20</td>
        <td>0.00</td>
        <td>0.00</td>
      </tr>
      <tr>
        <td>close drawer</td>
        <td>0.90</td>
        <td>0.45</td>
        <td style="background:#d1fae5;">0.95</td>
        <td>0.40</td>
      </tr>
      <tr>
        <td>slide block</td>
        <td style="background:#d1fae5;">0.90</td>
        <td>0.20</td>
        <td>0.10</td>
        <td>0.00</td>
      </tr>
      <tr>
        <td>jenga</td>
        <td style="background:#d1fae5;">0.55</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.00</td>
      </tr>
      <tr>
        <td>lift block</td>
        <td style="background:#d1fae5;">0.95</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.00</td>
      </tr>
      <tr>
        <td>rubbish</td>
        <td style="background:#d1fae5;">1.00</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.00</td>
      </tr>
      <tr>
        <td>push button</td>
        <td style="background:#d1fae5;">0.95</td>
        <td>0.30</td>
        <td>0.05</td>
        <td>0.00</td>
      </tr>
      <tr>
        <td style="background:#111;color:#fff;font-weight:700;">Average</td>
        <td style="background:#d1fae5;font-weight:700;">0.80</td>
        <td style="background:#fee2e2;font-weight:700;">0.14</td>
        <td style="background:#fee2e2;font-weight:700;">0.14</td>
        <td style="background:#fee2e2;font-weight:700;">0.05</td>
      </tr>
    </tbody>
  </table>
</div>




<H1> Analysis </H1>

<h3>Results analysis</h3>

<p class="text">
CrossInstruct outperforms the VLM-Reasoning ablation on most tasks, and the gap is largest on tasks where small spatial
misalignments are fatal.
</p>

<p class="text"><strong>CrossInstruct vs. VLM-Reasoning (no precision coupling)</strong></p>
<ul class="text">
  <li>
    <strong>Average success rate:</strong> CrossInstruct = 0.80, VLM-Reasoning = 0.14
    (absolute +0.66, about 5.6× higher on average).
  </li>
  <li>
    <strong>Where the gap comes from:</strong> without precision coupling, trajectories are often “nearly right” but slightly offset.
    That is enough to miss the interaction point, under-reach, or fail to make contact in precision tasks.
  </li>
  <li>
    <strong>Most sensitive tasks:</strong> Basketball, Jenga, Lift Block, Put Rubbish in Bin, and Push Button all require accurate
    contact or alignment. In these settings, a few centimeters of error can turn a correct plan into a failed rollout.
  </li>
</ul>

<p class="text"><strong>Why pure RL does best on Close Drawer</strong></p>
<ul class="text">
  <li>
    Close Drawer is relatively forgiving: it can succeed with coarse positioning and does not require tight coordination between
    end-effector pose, orientation, and gripper timing.
  </li>
  <li>
    Because the task is short-horizon and less precision-constrained, exploration-driven RL can discover a workable strategy more
    reliably within a fixed interaction budget, even with sparse rewards.
  </li>
</ul>

<p class="text"><strong>Why VLM-Reasoning can look better on the “colorful object” tasks</strong></p>
<ul class="text">
  <li>
    VLM-Reasoning does relatively better on <strong>Square Block on Peg</strong>, <strong>Slide Block to Target</strong>, and
    <strong>Push Button</strong> compared to the pure RL baselines.
  </li>
  <li>
    These are the three tasks with highly saturated, high-contrast targets (bright red blocks or buttons, a green target).
    Strong color cues make it easier for a vision-based model to localize “the right thing” even without explicit keypoints.
  </li>
  <li>
    The same reliance on color can also be brittle: when multiple objects share similar colors, a reasoning-only model can ground to
    the wrong instance and commit to an incorrect trajectory.
  </li>
</ul>

<p class="text">
Overall, the table suggests a simple pattern: <strong>language-level understanding is not the bottleneck</strong>; the bottleneck is
<strong>pixel-level grounding</strong>. Precision coupling converts a good plan into a correctly anchored trajectory, which is what
matters most on contact-rich manipulation.
</p>


