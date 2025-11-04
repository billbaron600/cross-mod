
import os, sys, argparse

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Import the refactored modules
from cross_mod.perception.drawing_io import create_combined_object as create_combined_object
from cross_mod.perception.density.gaussian import generate_waypoint_gaussians as generate_waypoint_gaussians
from cross_mod.perception.density.flow import run_generate_densities_for_all_images as run_generate_densities_for_all_images
from cross_mod.perception.density.stack import combine_density_views as combine_density_views
from cross_mod.perception.raycast import fit_NeRF_model as fit_NeRF_model
from cross_mod.perception.projection import project_ray_traced_mean as project_ray_traced_mean
from cross_mod.planning.ivk_runner import generate_IVK_trajectories as generate_IVK_trajectories
from utils.RLBenchFunctions.get_configuration_settings import Configuration  # Keep legacy Configuration class

try:
    import yaml  # optional
except Exception:
    yaml = None

def parse_args():
    p = argparse.ArgumentParser(description="cross-mod unified CLI (refactored)")
    p.add_argument("--task-name", type=str, required=True, help="RLBench task name (e.g., CloseDrawer)")
    p.add_argument("--phases", type=str, default="setup,densities,raycast,ivk", help="Comma list: setup,densities,raycast,project,ivk")
    p.add_argument("--seeds", type=str, default="0", help="Comma list of integer seeds")
    p.add_argument("--densities-source", type=str, default="gaussian", help="gaussian|flow")
    p.add_argument("--legacy-json", type=str, required=True, help="Path to the legacy JSON config (we reuse Configuration)")
    p.add_argument("--flow-iters", type=int, default=5000)
    return p.parse_args()

def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    phases = [s.strip() for s in args.phases.split(",") if s.strip()]

    # Build the original Configuration (this preserves camera tar & working dirs)
    cfg = Configuration(json_path=args.legacy_json, task=args.task_name, real_demo=False)
    cfg.create_working_dirs()

    if "setup" in phases:
        pass  # create_working_dirs already materializes dirs & camera tar via generate_camera_view when configured

    if "densities" in phases:
        create_combined_object(cfg, limit_to_correction_indices=seeds, segment_idx=0, render_trajectories=False)
        if args.densities_source == "gaussian":
            generate_waypoint_gaussians(cfg, limit_to_correction_indices=seeds, traj_idx=0, segment_idx=0)
        elif args.densities_source == "flow":
            run_generate_densities_for_all_images(cfg, limit_to_correction_indices=seeds, segment_idx=0, training_iterations=args.flow_iters)
        else:
            raise ValueError(f"Unknown densities source: {args.densities_source}")

    if "raycast" in phases:
        fit_NeRF_model(cfg, use_mean=True, segment_idx=0, limit_to_correction_indices=seeds)

    if "project" in phases:
        project_ray_traced_mean(cfg, limit_to_correction_indices=seeds, segment_idx=0)

    if "ivk" in phases:
        generate_IVK_trajectories(cfg, limit_to_correction_indices=seeds, segment_idx=0,
                                  discrete_gripper=True,
                                  use_gripper_orientation_file=True,
                                  use_gripper_action_file=True)

if __name__ == "__main__":
    main()