<link rel="stylesheet" href="assets/css/site.css">

# Cross-Mod: Cross-Modal Instructions for Robot Motion Generation

<!-- Fig. 1: same width as other sections -->
<img src="assets/img/fig-1-cross-mod.png" alt="Figure 1: Cross-Mod teaser" class="hero" loading="lazy">

## Method

### Diagrammatic Sketches Over Camera Views
<p class="text">
We use human <em>cross‑modal instructions</em>—rough freehand sketches plus short textual notes—drawn directly over a pair of calibrated camera views of the scene. From these scribbles, the system first generates <strong>2D end‑effector trajectories</strong> in each view and then <strong>lifts them into 3D</strong>.
</p>
<p class="text">
Concretely, each per‑view 2D curve is treated as a time‑indexed Gaussian “tube” in pixel space. Rays are cast through the pixels of both cameras (pinhole model) and their intersections define a distribution over feasible 3D waypoints at each time step. Taking the mean yields an executable centerline; sampling the distribution yields diverse rollouts for downstream learning. This multi‑view lifting gives us a coherent 3D path while preserving the shape implied by the sketch and avoiding collisions indicated by the user’s markings. See the <em>multi‑view lifting via ray casting</em> description and Fig. 4 in the paper for details. :contentReference[oaicite:1]{index=1}
</p>

<div class="grid grid-2 section">
  <img src="assets/img/play-jenga-sketched-demo.png" alt="Play Jenga sketched demo" loading="lazy">
  <img src="assets/img/rubish-in-bin-sketched-demo.png" alt="Rubbish in Bin sketched demo" loading="lazy">
</div>

### Task Identification, Precision Point Classification, Sketching, and Action Commands
<p class="text">
Our pipeline uses a <strong>hierarchical precision coupling</strong> between two vision–language models. A large <em>reasoning VLM</em> performs task identification and high‑level planning, proposing semantic <em>keypoint descriptors</em> (e.g., “button center,” “basketball rim edge”) and drafting the rough motion sketch. A smaller, fine‑tuned <strong>Molmo</strong> pointing model then converts each descriptor into precise pixel coordinates in both views. These precise keypoints are fed back into the reasoning context, which refines the per‑view trajectory and emits <strong>3D waypoints + end‑effector orientations + gripper open/close</strong> commands that form an executable trajectory. :contentReference[oaicite:2]{index=2}
</p>
<p class="text">
Finally, the 3D mean trajectory can be tracked directly, or we can sample multiple trajectories from the learned distribution to <strong>warm‑start reinforcement learning</strong> (TD3+BC), improving robustness while remaining faithful to the sketch intent (see §IV‑C and Fig. 12 for the RL initialization effect). :contentReference[oaicite:3]{index=3}
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

## Results

**Metric:** task success rate (fraction of successful rollouts)

<div class="section">
  <div class="table-wrap">
    <table class="metrics">
      <caption>Basketball / Peg / Close Drawer / Slide Block (RLBench)</caption>
      <thead>
        <tr>
          <th>Method</th>
          <th>basketball</th>
          <th>peg</th>
          <th>close drawer</th>
          <th>slide block</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>CrossInstruct</td>
          <td>0.90</td>
          <td>0.25</td>
          <td>0.90</td>
          <td>0.90</td>
        </tr>
        <tr>
          <td>VLM‑Reasoning</td>
          <td>0.00</td>
          <td>0.20</td>
          <td>0.45</td>
          <td>0.20</td>
        </tr>
        <tr>
          <td>Pure RL — SAC</td>
          <td>0.00</td>
          <td>0.00</td>
          <td>0.95</td>
          <td>0.10</td>
        </tr>
        <tr>
          <td>Pure RL — TD3</td>
          <td>0.00</td>
          <td>0.00</td>
          <td>0.40</td>
          <td>0.00</td>
        </tr>
      </tbody>
    </table>
  </div>

  <div class="table-wrap">
    <table class="metrics">
      <caption>Jenga / Lift Block / Rubbish / Push Button (RLBench)</caption>
      <thead>
        <tr>
          <th>Method</th>
          <th>jenga</th>
          <th>lift block</th>
          <th>rubbish</th>
          <th>push button</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>CrossInstruct</td>
          <td>0.55</td>
          <td>0.95</td>
          <td>1.00</td>
          <td>0.95</td>
        </tr>
        <tr>
          <td>VLM‑Reasoning</td>
          <td>0.00</td>
          <td>0.00</td>
          <td>0.00</td>
          <td>0.30</td>
        </tr>
        <tr>
          <td>Pure RL — SAC</td>
          <td>0.00</td>
          <td>0.00</td>
          <td>0.00</td>
          <td>0.05</td>
        </tr>
        <tr>
          <td>Pure RL — TD3</td>
          <td>0.00</td>
          <td>0.00</td>
          <td>0.00</td>
          <td>0.00</td>
        </tr>
      </tbody>
    </table>
  </div>

  <p class="text">
  CrossInstruct’s hierarchical precision coupling and multi‑view lifting drive strong out‑of‑the‑box performance, especially on precision tasks like <em>basketball‑in‑hoop</em> and <em>push button</em>, and the generated trajectory distribution provides effective initialization for TD3+BC in harder settings (see Table I / Figs. 8–12 in the paper). :contentReference[oaicite:4]{index=4}
  </p>
</div>






