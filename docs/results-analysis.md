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

<p class="text">
CrossInstruct outperforms the VLM-Reasoning ablation on <strong>all 8 tasks</strong>. The gap is not best explained by “precision demand”
alone. Instead, the consistent advantage comes from better <strong>visual grounding</strong> when the scene does not contain a single
bright, unambiguous target that a reasoning-only model can latch onto.
</p>

<p class="text"><strong>CrossInstruct vs. VLM-Reasoning (no precision coupling)</strong></p>
<ul class="text">
  <li>
    <strong>Average success rate:</strong> CrossInstruct = 0.80, VLM-Reasoning = 0.14
    (absolute +0.66, about 5.6× higher on average).
  </li>
  <li>
    <strong>Consistent win:</strong> CrossInstruct is higher on every task, including the highest-precision task
    (<strong>Square Block on Peg</strong>), where VLM-Reasoning can still perform reasonably well.
  </li>
</ul>

<p class="text"><strong>Why the gap tracks visual distinctiveness more than precision</strong></p>
<ul class="text">
  <li>
    A reasoning-only model can sometimes succeed on very high-precision tasks when the target is visually obvious, for example when
    the manipulated object is highly saturated and isolated from distractors. This helps explain why VLM-Reasoning can look
    comparatively strong on <strong>Square Block on Peg</strong>.
  </li>
  <li>
    The failure mode appears when the manipulated object is <strong>not</strong> visually distinct. In lower-contrast scenes or clutter,
    object boundaries are ambiguous and the reasoning model’s “good plan” can be anchored to the wrong pixels.
  </li>
  <li>
    Precision coupling addresses this by delegating localization to the pointing model, which is more reliable when there are
    weak color gradients or multiple plausible targets. The result is not just a better plan, but a plan that is grounded to the
    correct contact point.
  </li>
</ul>

<p class="text"><strong>Where CrossInstruct still struggles</strong></p>
<ul class="text">
  <li>
    CrossInstruct’s lowest-performing tasks are those that require <strong>tight coordination</strong> between end-effector
    <strong>position</strong>, <strong>orientation</strong>, and <strong>gripper timing</strong>, rather than simply reaching a visually
    identified object.
  </li>
  <li>
    Intuitively, these tasks impose a coupled constraint: being “near” the object is not enough. The approach direction, wrist
    orientation, and open-close schedule must all be correct in a narrow window for success.
  </li>
</ul>

<p class="text"><strong>Pure RL baselines (context)</strong></p>
<ul class="text">
  <li>
    The pure RL baselines are trained on each task with sparse rewards, then evaluated on the same held-out seeds. They serve as a
    reference for how far exploration-only learning can get without cross-modal supervision.
  </li>
  <li>
    Close Drawer is the one task where pure RL is competitive, which is consistent with it being relatively forgiving and not
    requiring precise multi-part coordination of pose and gripper events.
  </li>
</ul>



