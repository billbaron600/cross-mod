<link rel="stylesheet" href="assets/css/site.css">


<!-- Centered paper header -->
<header class="hero-header">
  <h1 class="title">Cross-Mod: Cross-Modal Instructions for Robot Motion Generation</h1>

  <!-- Put your author names here (links optional) -->
  <p class="authors">
    <!-- Example: <a href="#">Author One</a>, <a href="#">Author Two</a> -->
    William Baron, Xiaoxiang Dong, Matthew Johnson-Roberson, Weiming Zhi
  </p>

  <p class="affil">The Robotics Institute, Carnegie Mellon University</p>

  <div class="cta">
    <a class="btn" href="https://arxiv.org/abs/2509.21107" target="_blank" rel="noopener">Paper</a>
    <a class="btn" href="https://github.com/billbaron600/cross-mod/tree/main" target="_blank" rel="noopener">Code</a>
  </div>
</header>

# Instructing Reasoning Models via Diagrammtic Sketches

<!-- Fig. 1: same width as other sections -->
<img src="assets/img/fig-1-cross-mod.png" alt="Figure 1: Cross-Mod teaser" class="hero" loading="lazy">

## Method

### Diagrammatic Sketches Over Camera Views
<p class="text">
We use human <em>cross-modal instructions</em>—rough freehand sketches plus short textual notes—drawn directly over a pair of calibrated camera views. From these scribbles, the system first generates <strong>2D end-effector trajectories</strong> in each view and then <strong>lifts them into 3D</strong>.
</p>
<p class="text">
Each per-view 2D curve is treated as a time-indexed Gaussian corridor in pixel space. Rays are cast through pixels in both cameras and intersected to localize a distribution of feasible <strong>3D waypoints</strong> per time step. The mean yields an executable centerline; sampling the distribution yields diverse rollouts for downstream learning. This multi-view lifting preserves the sketch’s shape while respecting collision hints the user draws.
</p>

<div class="grid grid-2 section">
  <img src="assets/img/play-jenga-sketched-demo.png" alt="Play Jenga sketched demo" loading="lazy">
  <img src="assets/img/rubish-in-bin-sketched-demo.png" alt="Rubbish in Bin sketched demo" loading="lazy">
</div>

### Task Identification, Precision Point Classification, Sketching, and Action Commands
<p class="text">
Our pipeline couples a <strong>reasoning VLM</strong> with a fine-tuned <strong>Molmo</strong> pointing model. The reasoning model performs task identification and high-level planning, proposes semantic <em>keypoint descriptors</em> (e.g., “button center,” “rim edge”), and drafts the rough motion sketch. The pointing model converts each descriptor into precise pixel coordinates in both views. These precise keypoints are fed back to the reasoning model, which refines the per-view sketch and outputs <strong>3D waypoints + end-effector orientations + gripper open/close</strong> commands.
</p>
<p class="text">
The 3D mean trajectory can be executed directly, or we can <strong>sample</strong> from the learned distribution to warm-start reinforcement learning (TD3+BC), improving robustness while staying faithful to the sketched intent.
</p>

<!-- Stack the two images vertically (2×1) -->
<div class="stack section">
  <img src="assets/img/systems-diagram.png" alt="System diagram" loading="lazy">
  <img src="assets/img/pointing-commands-and-sketching.png" alt="Pointing commands and sketching" loading="lazy">
</div>

## Video Examples

### Real-World Manipulator
<div class="grid grid-2 section">
  <video class="video" controls playsinline preload="metadata" aria-label="Sort cups—real robot demo">
    <source src="assets/video/sort-cups-video-use.mp4" type="video/mp4">
  </video>
  <video class="video" controls playsinline preload="metadata" aria-label="Saw block—real robot demo">
    <source src="assets/video/saw-block-video-use.mp4" type="video/mp4">
  </video>
</div>

### Simulation Rollouts
<!-- Force a 2×3 layout on desktop; stacks to 1×6 on small screens -->
<div class="grid grid-2 section sim-grid" data-layout="2x3">
  <video class="video" controls playsinline preload="metadata" aria-label="Insert block on peg">
    <source src="assets/video/insert-in-peg.mp4" type="video/mp4">
  </video>
  <video class="video" controls playsinline preload="metadata" aria-label="Put rubbish in bin">
    <source src="assets/video/rubish-in-bin.mp4" type="video/mp4">
  </video>
  <video class="video" controls playsinline preload="metadata" aria-label="Slide block to target">
    <source src="assets/video/slide-block-to-target.mp4" type="video/mp4">
  </video>
  <video class="video" controls playsinline preload="metadata" aria-label="Basketball in hoop">
    <source src="assets/video/basketball-in-hoop.mp4" type="video/mp4">
  </video>
  <video class="video" controls playsinline preload="metadata" aria-label="Close drawer">
    <source src="assets/video/close-drawer.mp4" type="video/mp4">
  </video>
  <video class="video" controls playsinline preload="metadata" aria-label="Play Jenga">
    <source src="assets/video/play-jenga.mp4" type="video/mp4">
  </video>
</div>

## Ablations & Analyses

<p class="text">
<strong>1) Hierarchical precision coupling (Reasoning VLM + Molmo) vs. Reasoning-only:</strong> Removing the pointing model (reasoning-only) often yields small spatial misalignments that cause failures on precision tasks (e.g., basketball, button, Jenga). With coupling, precise keypoints anchor the sketch and stabilize the full 3D trajectory generation.
</p>
<p class="text">
<strong>2) Cross-modal trajectories as RL initialization:</strong> Sampling from the trajectory distribution provides diverse, semantically correct demonstrations. TD3+BC initialized with these samples converges fast on difficult tasks such as Jenga and Peg, while training from scratch struggles.
</p>
<p class="text">
<strong>3) Multi-view lifting via ray-casting:</strong> Casting rays through both views reduces depth ambiguity, producing 3D waypoints that respect the sketched geometry across views. The mean can be executed directly, while the distribution captures uncertainty for robust rollouts.
</p>

## Results

**Metric:** task success rate (fraction of successful rollouts)

<div class="section">
  <div class="table-wrap">
    <table class="metrics">
      <caption>Simulation Results on RLBench (single merged table)</caption>
      <thead>
        <tr>
          <th>Method</th>
          <th>basketball</th>
          <th>peg</th>
          <th>close drawer</th>
          <th>slide block</th>
          <th>jenga</th>
          <th>lift block</th>
          <th>rubbish</th>
          <th>push button</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>CrossInstruct</td>
          <td>0.90</td>
          <td>0.25</td>
          <td>0.90</td>
          <td>0.90</td>
          <td>0.55</td>
          <td>0.95</td>
          <td>1.00</td>
          <td>0.95</td>
        </tr>
        <tr>
          <td>VLM-Reasoning</td>
          <td>0.00</td>
          <td>0.20</td>
          <td>0.45</td>
          <td>0.20</td>
          <td>0.00</td>
          <td>0.00</td>
          <td>0.00</td>
          <td>0.30</td>
        </tr>
        <tr>
          <td>Pure RL — SAC</td>
          <td>0.00</td>
          <td>0.00</td>
          <td>0.95</td>
          <td>0.10</td>
          <td>0.00</td>
          <td>0.00</td>
          <td>0.00</td>
          <td>0.05</td>
        </tr>
        <tr>
          <td>Pure RL — TD3</td>
          <td>0.00</td>
          <td>0.00</td>
          <td>0.40</td>
          <td>0.00</td>
          <td>0.00</td>
          <td>0.00</td>
          <td>0.00</td>
          <td>0.00</td>
        </tr>
      </tbody>
    </table>
  </div>

  <p class="text">
    CrossInstruct’s hierarchical precision coupling and multi-view lifting drive strong out-of-the-box performance on precision tasks, while the trajectory distribution provides an effective initialization for TD3+BC on hard rollouts.
  </p>
</div>










