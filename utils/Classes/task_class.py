from utils.Classes.PreferenceModel import PreferenceModel
from rlbench.tasks import SlideBlockToTarget
from stable_baselines3 import SAC
from utils.RLBenchFunctions.template_sensor_views import generate_camera_view, compute_camera_pose
from typing import List, Optional, Dict
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import copy
import numpy as np

class SketchCollector:
    def __init__(self, max_y=1200):
        self.max_y = max_y
        self.drawing = False
        self.ix, self.iy = -1, -1

    def draw_sketches(self, frames, n_trajs=1, traj_len_est=6, ts=20, save_drawings=True, cursor_size=3):
        """
        Allows a user to draw trajectories on a list of image arrays (each of shape 1200x1200x3).

        Args:
            frames: List of image arrays (numpy ndarray or torch tensor) with shape HxWx3, values in [0,1] or [0,255].
            n_trajs: Number of trajectories to draw per image.
            traj_len_est: Unused currently.
            ts: Unused currently.
            save_drawings: If True, saves modified images with drawings.
            cursor_size: Size of the drawing cursor.

        Returns:
            A nested list of torch tensors representing normalized trajectories: List[n_images][n_trajs][T, 2]
        """

        def draw_curve(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = not self.drawing
            elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
                cv2.circle(self.img, (x, y), cursor_size, (0, 0, 255), -1)
                self.list_xy.append([x, y])
            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
                cv2.circle(self.img, (x, y), cursor_size, (0, 0, 255), -1)

        grand_traj_l = []

        for im_idx, frame in enumerate(frames):
            # Ensure numpy format
            if isinstance(frame, torch.Tensor):
                frame_np = frame.cpu().detach().numpy()
            else:
                frame_np = frame

            # Convert to uint8
            if frame_np.max() <= 1.0:
                frame_np = (frame_np * 255).astype(np.uint8)
            else:
                frame_np = frame_np.astype(np.uint8)

            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

            traj_list = []
            for jj in range(n_trajs):
                self.list_xy = []
                self.img = frame_bgr.copy()
                cv2.namedWindow("Curve Window")
                cv2.setMouseCallback("Curve Window", draw_curve)

                while True:
                    cv2.imshow("Curve Window", self.img)
                    cv2.moveWindow("Curve Window", 500, 100)
                    if cv2.waitKey(10) == 27:
                        break
                cv2.destroyAllWindows()

                if save_drawings:
                    save_path = f"sketch_view_{im_idx}_traj_{jj}.png"
                    cv2.imwrite(save_path, self.img)

                xy_tor = torch.tensor(self.list_xy, dtype=torch.float32)
                xy_tor[:, 1] = self.max_y - xy_tor[:, 1]
                traj_list.append(xy_tor.detach().clone())

            grand_traj_l.append(copy.deepcopy(traj_list))

        # Normalize and visualize
        grand_traj_out = []
        for im in range(len(grand_traj_l)):
            traj_list_s = []
            plt.figure()
            for traj in grand_traj_l[im]:
                traj_all = traj / self.max_y
                plt.scatter(traj_all[:, 0], traj_all[:, 1])
                traj_list_s.append(traj_all[None])
            grand_traj_out.append(copy.deepcopy(traj_list_s))
            plt.xlim([0, 1])
            plt.ylim([0, 1])

        return grand_traj_out


class IterationClass:
    def __init__(self):
        self.trajectory_correction_images = []
        self.template_sensor_views = []
        self.template_sensor_extrinsics = []
    
    def add_template_view(self,frames,extrinsics,add_correction_now=False):
        self.template_sensor_views.append(frames)
        self.template_sensor_extrinsics.append(extrinsics)
        if add_correction_now==True:
            self.elicit_human_correction(frames)
            print("x")

    def elicit_human_correction(frames,current_trajectory=None):
        sketch_collector = SketchCollector()
        human_corrections = sketch_collector.draw_sketches(frames)




class Task:
    def __init__(
        self,
        task_class=SlideBlockToTarget,
        reward_model=None,
        preference_database=None,
        SAC_kwargs=None,
        task_seeds: List[int] = list(range(16)),
        camera_view_config: Optional[List[Dict]] = None
    ):
        """
        Task class for managing the human-in-the-loop RLBench training loop.

        Args:
            task_class: RLBench task class to be used (not a string).
            reward_model: Instance of the reward model. If None, use PreferenceModel.
            preference_database: Existing list of preference pairs to initialize with.
            SAC_kwargs: Keyword arguments to initialize SAC policy later.
            task_seeds: List of seeds to use for RLBench task variations.
            camera_view_config: Camera view configuration. If None, defaults to main_user_preferences.ipynb config.
        """
        self.task_class = task_class

        self.reward_model = reward_model if reward_model is not None else PreferenceModel()

        self.preference_database = preference_database if preference_database is not None else []

        self.SAC_kwargs = SAC_kwargs if SAC_kwargs is not None else {}

        self.policy = None  # Will be initialized later when SAC training starts

        self.iterations = []  # List to store Iteration objects

        self.task_seeds = task_seeds

        # Default camera views if none provided
        if camera_view_config is None:
            # Define the input parameters in a single keyword dictionary
            self.camera_view_config = {
                "task": self.task_class,
                "resolution": [1200, 1200],
                "camera_positions": [
                    compute_camera_pose(np.array((1.25, 0.75, 1.5))),
                    compute_camera_pose(np.array((1.25, -0.75, 1.5)), center_point=np.array([0.3, 0, 1])),
                    compute_camera_pose(np.array((-0.5, -0.75, 1.25)), center_point=np.array([0.3, 0, 1])),
                    compute_camera_pose(np.array((-0.5, 0.75, 1.25)))
                ],
                "seed": None,
                "environment_information": {
                    "observation_mode": 'state',
                    "render_mode": "rgb_array"
                },
                "robot_starting_position": None,
                "save_folder": "run_results/"}
        else:
            self.camera_view_config = camera_view_config

        self.elicit_demonstrations()

    def elicit_demonstrations(self):
        """
        Placeholder for the function that gathers human demonstrations via sketches and saves them.
        Implementation should handle camera view generation, user interaction, and NeRF reconstruction.
        """

        iteration_data = IterationClass()
        for i in self.task_seeds:
            camera_view_config = self.camera_view_config
            camera_view_config["seed"] = i
            frames,extrinsics,working_folder = generate_camera_view(**camera_view_config)
            iteration_data.add_template_view(frames, extrinsics,add_correction_now=True)
        


    