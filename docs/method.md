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

<p class="text">
Cross‑Mod turns rough <strong>sketch + text</strong> instructions into an executable robot motion. A human annotates two calibrated RGB camera views with freehand strokes (paths, arrows, “avoid” regions) and short notes (e.g., “grasp here”, “close gripper”, “repeat 3x”). We treat these annotations as lightweight “demonstrations”: at runtime, the system transfers the intent to a new scene, produces a 3D waypoint distribution plus orientation/gripper commands, and rolls out the mean trajectory open loop.
</p>

<div class="stack section">
  <img src="assets/img/systems-diagram.png" alt="Cross‑Mod system overview" loading="lazy">
</div>

## Overview of the Loop
<div class="text">
  <ol>
    <li><strong>Inputs:</strong> (1) one or more cross‑modal instruction examples and (2) a fresh two‑view observation of the current scene.</li>
    <li><strong>Reasoning:</strong> a large vision‑language model infers the task and breaks it into subgoals (reach, align, interact, retreat, repeat).</li>
    <li><strong>Semantic keypoints:</strong> it proposes the task‑critical points it must hit (e.g., “button center”, “block edge”, “bin rim”).</li>
    <li><strong>Precision coupling:</strong> a lightweight pointing model converts each keypoint description into pixel‑accurate coordinates in both views.</li>
    <li><strong>2D trajectory sketching:</strong> with grounded pixels in context, the reasoning model draws a smooth 2D end‑effector path in each view that connects the keypoints and respects any “avoid/collision” hints.</li>
    <li><strong>Lift to 3D:</strong> calibrated ray casting fuses the two per‑view paths into a time‑aligned 3D waypoint distribution.</li>
    <li><strong>Execution:</strong> the system outputs (a) a 3D waypoint sequence, (b) end‑effector orientations, and (c) gripper open/close events. The robot tracks the mean path open loop; sampled paths provide diverse rollouts for learning.</li>
  </ol>
</div>

## Diagrammatic Sketches Over Camera Images
<p class="text">
Cross‑modal instructions are intentionally lightweight: they capture <strong>shape</strong> (“go around this object”), <strong>contacts</strong> (“grasp here”), and <strong>constraints</strong> (“don’t hit this wall”) without requiring teleoperation. Because the sketches live directly in image space, they’re quick to provide and naturally aligned with what the robot sees.
</p>

<div class="grid grid-2 section">
  <img src="assets/img/play-jenga-sketched-demo.png" alt="Play Jenga sketched demo" loading="lazy">
  <img src="assets/img/rubish-in-bin-sketched-demo.png" alt="Rubbish in Bin sketched demo" loading="lazy">
</div>

<p class="text">
In practice we use three kinds of marks: (1) a rough <strong>path line</strong> for the end‑effector, (2) <strong>arrows</strong> to indicate approach or motion direction, and (3) short <strong>text labels</strong> for key actions (“close”, “open”, “repeat”, “avoid”).
</p>

## Reasoning Model: Task, Subgoals, and Keypoints
<p class="text">
A large reasoning VLM reads the instruction(s) and the current two‑view observation to decide <strong>what the task is</strong> and <strong>what must be precise</strong>. It produces:
</p>

<div class="text">
  <ul>
    <li><strong>Task identification:</strong> what manipulation is being requested (press, pick‑and‑place, insertion, sliding, tool use).</li>
    <li><strong>Subgoal structure:</strong> reach → align → interact → retreat (and any repeats).</li>
    <li><strong>Semantic keypoint descriptors:</strong> short phrases for the exact points needed to execute (targets, contact points, alignment edges).</li>
    <li><strong>Initial motion intent:</strong> a rough per‑view plan consistent with the sketch and text.</li>
  </ul>
</div>

<p class="text">
These keypoints start as language (e.g., “rim edge”, “button center”). The next step is grounding them to pixel coordinates so the motion is actually executable.
</p>

## Precision Coupling: Pixel‑Accurate Keypoints
<p class="text">
We couple the reasoning VLM to a small pointing model that specializes in <strong>turning a keypoint description into a precise pixel coordinate</strong> in each view. Those pixel locations are injected back into the reasoning model’s context, so the final plan is both semantically correct and geometrically precise (critical for small targets like buttons, peg holes, and block edges).
</p>

<div class="stack section">
  <img src="assets/img/pointing-commands-and-sketching.png" alt="Pointing commands and sketch refinement" loading="lazy">
</div>

## Full Loop: From Grounded Pixels to 3D Motion
<p class="text">
After keypoints are grounded, the reasoning model redraws the final 2D path in each camera view as a smooth, time‑ordered curve. We treat each curve as a <strong>corridor</strong> rather than a single thin line: at each time step, we allow a small amount of uncertainty around the drawn pixel, which makes the resulting 3D path more robust to ambiguity and minor sketch variation.
</p>

<p class="text">
To lift the paths into 3D, we use the known calibration of the two cameras. For each time step, we cast rays from both cameras through pixels along the two corridors and keep the 3D points that are simultaneously consistent with both views. This produces a <strong>distribution of feasible 3D waypoints over time</strong>. The mean waypoint sequence is the executable centerline, and sampling produces diverse rollouts that remain faithful to the original sketch intent.
</p>

<p class="text">
Finally, the reasoning model assigns <strong>end‑effector orientations</strong> (to align the gripper with slots/surfaces or to pull along a specific direction) and <strong>gripper open/close events</strong> along the path. The robot tracks the resulting 6‑DoF motion open loop using an IK‑based controller or reactive primitives.
</p>

<p class="text">
If you want learning on top: sampling the waypoint distribution yields varied but intent‑consistent trajectories that can be logged as low‑cost “demonstrations” for downstream RL refinement.
</p>



