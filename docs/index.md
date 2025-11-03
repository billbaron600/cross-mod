<link rel="stylesheet" href="assets/css/site.css">

<div class="page">

# Cross-Mod: Cross-Modal Instructions for Robot Motion Generation

<!-- ===== Figure 1 (full width) ===== -->
</div>
<div class="full-bleed">
  <img src="assets/img/fig-1-cross-mod.png" alt="Figure 1: Cross-Mod teaser" class="hero">
</div>
<div class="page">

## Method

### Diagrammatic Sketches Over Camera Views
Briefly describe how multi-view sketches condition the trajectory generation and how they’re used downstream (you can expand later).

<div class="grid grid-2">
  <img src="assets/img/play-jenga-sketched-demo.png" alt="Play Jenga sketched demo">
  <img src="assets/img/rubish-in-bin-sketched-demo.png" alt="Rubbish in Bin sketched demo">
</div>

### Task Identification, Precision Point Classification, Sketching, and Action Commands
Short paragraph on:
- Reasoning model → task identification and high-level plan  
- **Molmo** for precision point classification (pointing)  
- Multi-view **ray casting** to lift 2D points to 3D  
- Coupling to orientations + gripper actions from the reasoning outputs, yielding executable trajectories

<div class="grid grid-2">
  <img src="assets/img/systems-diagram.png" alt="System diagram">
  <img src="assets/img/pointing-commands-and-sketching.png" alt="Pointing commands and sketching">
</div>

## Video Examples

### Real-World Manipulator
<div class="grid grid-2">
  <video class="video" controls playsinline preload="metadata">
    <source src="assets/video/sort-cups-video-use.mp4" type="video/mp4">
  </video>
  <video class="video" controls playsinline preload="metadata">
    <source src="assets/video/saw-block-video-use.mp4" type="video/mp4">
  </video>
</div>

### Simulation Rollouts
<div class="grid grid-3">
  <video class="video" controls playsinline preload="metadata">
    <source src="assets/video/basketball-in-hoop.mp4" type="video/mp4">
  </video>
  <video class="video" controls playsinline preload="metadata">
    <source src="assets/video/close-drawer.mp4" type="video/mp4">
  </video>
  <video class="video" controls playsinline preload="metadata">
    <source src="assets/video/play-jenga.mp4" type="video/mp4">
  </video>
  <video class="video" controls playsinline preload="metadata">
    <source src="assets/video/insert-in-peg.mp4" type="video/mp4">
  </video>
  <video class="video" controls playsinline preload="metadata">
    <source src="assets/video/rubish-in-bin.mp4" type="video/mp4">
  </video>
  <video class="video" controls playsinline preload="metadata">
    <source src="assets/video/slide-block-to-target.mp4" type="video/mp4">
  </video>
</div>

<!-- You can add a Results section later if desired -->
</div>








