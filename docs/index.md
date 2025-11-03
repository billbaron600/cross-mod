<link rel="stylesheet" href="assets/css/site.css">

# Cross-Mod: Cross-Modal Instructions for Robot Motion Generation

<!-- Fig. 1: same width as other sections -->
<img src="assets/img/fig-1-cross-mod.png" alt="Figure 1: Cross-Mod teaser" class="hero">

## Method

### Diagrammatic Sketches Over Camera Views
<p class="text">
Briefly describe how multi-view sketches condition trajectory generation and how they’re used downstream (you can expand later).
</p>

<div class="grid grid-2 section">
  <img src="assets/img/play-jenga-sketched-demo.png" alt="Play Jenga sketched demo">
  <img src="assets/img/rubish-in-bin-sketched-demo.png" alt="Rubbish in Bin sketched demo">
</div>

### Task Identification, Precision Point Classification, Sketching, and Action Commands
<p class="text">
Reasoning model → task identification and high-level plan; <b>Molmo</b> for precision point classification; multi-view <b>ray casting</b> to lift 2D points to 3D; coupling to orientations + gripper actions from the reasoning outputs to yield executable trajectories.
</p>

<!-- Stack the two images vertically (2×1) -->
<div class="stack section">
  <img src="assets/img/systems-diagram.png" alt="System diagram">
  <img src="assets/img/pointing-commands-and-sketching.png" alt="Pointing commands and sketching">
</div>

## Video Examples

### Real-World Manipulator
<div class="grid grid-2 section">
  <video class="video" controls playsinline preload="metadata">
    <source src="assets/video/sort-cups-video-use.mp4" type="video/mp4">
  </video>
  <video class="video" controls playsinline preload="metadata">
    <source src="assets/video/saw-block-video-use.mp4" type="video/mp4">
  </video>
</div>

### Simulation Rollouts
<div class="grid grid-2 section">
  <video class="video" controls playsinline preload="metadata">
    <source src="assets/video/insert-in-peg.mp4" type="video/mp4">
  </video>
  <video class="video" controls playsinline preload="metadata">
    <source src="assets/video/rubish-in-bin.mp4" type="video/mp4">
  </video>
  <video class="video" controls playsinline preload="metadata">
    <source src="assets/video/slide-block-to-target.mp4" type="video/mp4">
  </video>
  <video class="video" controls playsinline preload="metadata">
    <source src="assets/video/basketball-in-hoop.mp4" type="video/mp4">
  </video>
  <video class="video" controls playsinline preload="metadata">
    <source src="assets/video/close-drawer.mp4" type="video/mp4">
  </video>
  <video class="video" controls playsinline preload="metadata">
    <source src="assets/video/play-jenga.mp4" type="video/mp4">
  </video>
</div>



## Results

**Metric:** task success rate (fraction of successful rollouts)

<div class="section">

**Basketball / Peg / Close Drawer / Slide Block**

| Method           | basketball | peg  | close drawer | slide block |
|------------------|-----------:|-----:|-------------:|------------:|
| CrossInstruct    | 0.90       | 0.25 | 0.90         | 0.90        |
| VLM-Reasoning    | 0.00       | 0.20 | 0.45         | 0.20        |
| Pure RL — SAC    | 0.00       | 0.00 | 0.95         | 0.10        |
| Pure RL — TD3    | 0.00       | 0.00 | 0.40         | 0.00        |

**Jenga / Lift Block / Rubbish / Push Button**

| Method           | jenga | lift block | rubbish | push button |
|------------------|------:|-----------:|--------:|------------:|
| CrossInstruct    | 0.55  | 0.95       | 1.00    | 0.95        |
| VLM-Reasoning    | 0.00  | 0.00       | 0.00    | 0.30        |
| Pure RL — SAC    | 0.00  | 0.00       | 0.00    | 0.05        |
| Pure RL — TD3    | 0.00  | 0.00       | 0.00    | 0.00        |

</div>






