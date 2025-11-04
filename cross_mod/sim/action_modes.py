# refactored module (auto-generated)


# ---- imports from original modules ----
from abc import abstractmethod

from enum import Enum

from gymnasium import Wrapper

from gymnasium.wrappers import TimeLimit

from pyquaternion import Quaternion

from pyrep.const import ConfigurationPathAlgorithms as Algos, ObjectType

from pyrep.errors import ConfigurationPathError, IKError

from pyrep.objects import Object, Dummy

from pyrep.objects.joint import Joint

from rlbench.action_modes.action_mode import MoveArmThenGripper,JointPositionActionMode, ActionMode

from rlbench.action_modes.arm_action_modes import *

from rlbench.action_modes.arm_action_modes import JointPosition

from rlbench.action_modes.arm_action_modes import JointVelocity,JointPosition, ArmActionMode, RelativeFrame

from rlbench.action_modes.gripper_action_modes import Discrete

from rlbench.backend.exceptions import InvalidActionError

from rlbench.backend.robot import Robot

from rlbench.backend.scene import Scene

from rlbench.const import SUPPORTED_ROBOTS

from rlbench.environment import Environment

from rlbench.gym import RLBenchEnv

from rlbench.tasks import SlideBlockToTarget

from scipy.spatial.transform import Rotation

from shimmy.openai_gym_compatibility import GymV21CompatibilityV0

from stable_baselines3 import PPO

from stable_baselines3 import SAC

from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines3.common.callbacks import EvalCallback

from typing import List, Union

import numpy as np

import random



def obs_to_dict(obs):
    """
    Robustly convert an RLBench Observation → dict, regardless of
    whether the Observation class is a dataclass, has __dict__, or
    only exposes attributes via @property.
    """
    # 1) Fast path: instance has a real __dict__
    try:
        return {k: v for k, v in vars(obs).items() if not k.startswith("_")}
    except TypeError:
        pass                       # no __dict__ on this build

    # 2) Fallback: inspect dir(), keep non-callable public attrs
    out = {}
    for name in dir(obs):
        if name.startswith("_"):
            continue                # skip dunder / private names
        value = getattr(obs, name)
        if callable(value):
            continue                # skip methods
        out[name] = value
    return out

def assert_action_shape(action: np.ndarray, expected_shape: tuple):
    if np.shape(action) != expected_shape:
        raise InvalidActionError(
            'Expected the action shape to be: %s, but was shape: %s' % (
                str(expected_shape), str(np.shape(action))))

def assert_unit_quaternion(quat):
    if not np.isclose(np.linalg.norm(quat), 1.0):
        raise InvalidActionError('Action contained non unit quaternion!')

class IVKPlanningBounds_Delta(MoveArmThenGripper):
    """Same as MoveArmThenGripper, but with fixed action bounds.

    This method clamps the first 7 values (x, y, z, qx, qy, qz, qw)
    to ±0.1, and the 8th (gripper) to [0.0, 0.04].
    """

    def __init__(self, action_limits=None,*args, **kwargs):
        # call parent constructor
        self.specified_action_bounds = action_limits 
        super().__init__(*args, **kwargs)
        # any additional initialization can go here
        # e.g. self.some_setting = True
        

    def action_bounds(self):
        if self.specified_action_bounds is None:
            PANDA_LOWER = np.array([
                -0.1,   # dx
                -0.1,   # dy
                -0.1,   # dz
                -1,   # dx quat
                -1,   # dy quat
                -1,   # dz quat
                0,   # dw quat
                0.0       # gripper closed
            ], dtype=np.float32)

            PANDA_UPPER = np.array([
                0.1,
                0.1,
                0.1,
                1.0,
                1.0,
                1.0,
                1.0,
                0.04      # gripper fully open (RL‑Bench convention)
            ], dtype=np.float32)
            return PANDA_LOWER,PANDA_UPPER
        else:
            return self.specified_action_bounds["PANDA_LOWER"], self.specified_action_bounds["PANDA_UPPER"]

class IVKPlanningBounds_NonDiscrete(MoveArmThenGripper):
    """Same as MoveArmThenGripper, but with fixed action bounds.

    This method clamps the first 7 values (x, y, z, qx, qy, qz, qw)
    to ±0.1, and the 8th (gripper) to [0.0, 0.04].
    """

    def __init__(self, action_limits=None,*args, **kwargs):
        # call parent constructor
        self.specified_action_bounds = action_limits 
        super().__init__(*args, **kwargs)
        # any additional initialization can go here
        # e.g. self.some_setting = True
        

    def action_bounds(self):
        # Cartesian bounds (metres) – tune to your workspace
        x_min, x_max = -0.3, 1.0
        y_min, y_max = -1.0, 1.0
        z_min, z_max =  0.6, 1.8      # table-top to ~40 cm above

        PANDA_LOWER = np.array([
            x_min,y_min,z_min,
            -1.0, -1.0, -1.0, -1.0,  # quaternion (qx, qy, qz, qw)
            0.0                                                 # gripper closed
        ], dtype=np.float32)

        PANDA_UPPER = np.array([
            x_max,y_max,z_max,
            1.0, 1.0, 1.0, 1.0,      # quaternion (qx, qy, qz, qw)
            0.04                                                 # gripper fully open
        ], dtype=np.float32)

        """
        PANDA_LOWER = np.array([
            x_min, y_min, z_min,        # position
            -1.0, -1.0, -1.0, -1.0,     # quaternion (qx, qy, qz, qw)
            0.0                       # gripper closed
        ], dtype=np.float32)

        PANDA_UPPER = np.array([
            x_max, y_max, z_max,        # position
            1.0,  1.0,  1.0,  1.0,     # quaternion (qx, qy, qz, qw)
            1.0                       # gripper fully open (RL-Bench)
        ], dtype=np.float32)
        """

        return PANDA_LOWER, PANDA_UPPER

class IVKPlanningBounds(MoveArmThenGripper):
    """Same as MoveArmThenGripper, but with fixed action bounds.

    This method clamps the first 7 values (x, y, z, qx, qy, qz, qw)
    to ±0.1, and the 8th (gripper) to [0.0, 0.04].
    """

    def __init__(self, action_limits=None,*args, **kwargs):
        # call parent constructor
        self.specified_action_bounds = action_limits 
        super().__init__(*args, **kwargs)
        # any additional initialization can go here
        # e.g. self.some_setting = True
        

    def action_bounds(self):
        # Cartesian bounds (metres) – tune to your workspace
        x_min, x_max = -0.3, 1.0
        y_min, y_max = -1.0, 1.0
        z_min, z_max =  0.6, 1.8      # table-top to ~40 cm above

        PANDA_LOWER = np.array([
            x_min,y_min,z_min,
            -1.0, -1.0, -1.0, -1.0,  # quaternion (qx, qy, qz, qw)
            0.0                                                 # gripper closed
        ], dtype=np.float32)

        PANDA_UPPER = np.array([
            x_max,y_max,z_max,
            1.0, 1.0, 1.0, 1.0,      # quaternion (qx, qy, qz, qw)
            1.0                                                 # gripper fully open
        ], dtype=np.float32)

        """
        PANDA_LOWER = np.array([
            x_min, y_min, z_min,        # position
            -1.0, -1.0, -1.0, -1.0,     # quaternion (qx, qy, qz, qw)
            0.0                       # gripper closed
        ], dtype=np.float32)

        PANDA_UPPER = np.array([
            x_max, y_max, z_max,        # position
            1.0,  1.0,  1.0,  1.0,     # quaternion (qx, qy, qz, qw)
            1.0                       # gripper fully open (RL-Bench)
        ], dtype=np.float32)
        """

        return PANDA_LOWER, PANDA_UPPER

class IVKPlanningBounds_General(MoveArmThenGripper):


    def __init__(self, action_limits=None,*args, **kwargs):
        # call parent constructor
        self.specified_action_bounds = action_limits 
        super().__init__(*args, **kwargs)
        # any additional initialization can go here
        # e.g. self.some_setting = True
        

    def action_bounds(self):
        # Cartesian bounds (metres) – tune to your workspace
        x_min, x_max = -0.5, 1.5
        y_min, y_max = -1.5, 1.5
        z_min, z_max =  0.6, 1.7      # table-top to ~40 cm above

        PANDA_LOWER = np.array([
            x_min,y_min,z_min,
            -1.0, -1.0, -1.0, -1.0,  # quaternion (qx, qy, qz, qw)
            0.0                                                 # gripper closed
        ], dtype=np.float32)

        PANDA_UPPER = np.array([
            x_max,y_max,z_max,
            1.0, 1.0, 1.0, 1.0,      # quaternion (qx, qy, qz, qw)
            1.0                                                 # gripper fully open
        ], dtype=np.float32)

        """
        PANDA_LOWER = np.array([
            x_min, y_min, z_min,        # position
            -1.0, -1.0, -1.0, -1.0,     # quaternion (qx, qy, qz, qw)
            0.0                       # gripper closed
        ], dtype=np.float32)

        PANDA_UPPER = np.array([
            x_max, y_max, z_max,        # position
            1.0,  1.0,  1.0,  1.0,     # quaternion (qx, qy, qz, qw)
            1.0                       # gripper fully open (RL-Bench)
        ], dtype=np.float32)
        """

        return PANDA_LOWER, PANDA_UPPER

class IVKPlanningBounds_JENGA(MoveArmThenGripper):
    """Same as MoveArmThenGripper, but with fixed action bounds.

    This method clamps the first 7 values (x, y, z, qx, qy, qz, qw)
    to ±0.1, and the 8th (gripper) to [0.0, 0.04].
    """

    def __init__(self, action_limits=None,*args, **kwargs):
        # call parent constructor
        self.specified_action_bounds = action_limits 
        super().__init__(*args, **kwargs)
        # any additional initialization can go here
        # e.g. self.some_setting = True
        

    def action_bounds(self):
        # Cartesian bounds (metres) – tune to your workspace
        x_min, x_max = -0.3249, 0.3249
        y_min, y_max = -.4550, 0.4550
        z_min, z_max =  0.65, 1.3      # table-top to ~40 cm above

        PANDA_LOWER = np.array([
            x_min, y_min, z_min,        # position
            -1.0, -1.0, -1.0, -1.0,     # quaternion (qx, qy, qz, qw)
            0.0                       # gripper closed
        ], dtype=np.float32)

        PANDA_UPPER = np.array([
            x_max, y_max, z_max,        # position
            1.0,  1.0,  1.0,  1.0,     # quaternion (qx, qy, qz, qw)
            1.0                       # gripper fully open (RL-Bench)
        ], dtype=np.float32)

        return PANDA_LOWER, PANDA_UPPER

class EndEffectorPoseViaPlanning_Extended(ArmActionMode):
    """High-level action where target pose is given and reached via planning.

    Given a target pose, a linear path is first planned (via IK). If that fails,
    sample-based planning will be used. The decision to apply collision
    checking is a crucial trade off! With collision checking enabled, you
    are guaranteed collision free paths, but this may not be applicable for task
    that do require some collision. E.g. using this mode on pushing object will
    mean that the generated path will actively avoid not pushing the object.

    Note that path planning can be slow, often taking a few seconds in the worst
    case.

    This was the action mode used in:
    James, Stephen, and Andrew J. Davison. "Q-attention: Enabling Efficient
    Learning for Vision-based Robotic Manipulation."
    arXiv preprint arXiv:2105.14829 (2021).
    """

    def __init__(self,
                 absolute_mode: bool = True,
                 frame: RelativeFrame = RelativeFrame.WORLD,
                 collision_checking: bool = False):
        """
        If collision check is enbled, and an object is grasped, then we

        Args:
            absolute_mode: If we should opperate in 'absolute', or 'delta' mode.
            frame: Either WORLD or EE.
            collision_checking: IF collision checking is enabled.
        """
        self._absolute_mode = absolute_mode
        self._frame = frame
        self._collision_checking = collision_checking
        self._robot_shapes = None

    def _quick_boundary_check(self, scene: Scene, action: np.ndarray):
        pos_to_check = action[:3]
        relative_to = None if self._frame == RelativeFrame.WORLD else scene.robot.arm.get_tip()
        if relative_to is not None:
            scene.target_workspace_check.set_position(pos_to_check, relative_to)
            pos_to_check = scene.target_workspace_check.get_position()
        if not scene.check_target_in_workspace(pos_to_check):
            raise InvalidActionError('A path could not be found because the '
                                     'target is outside of workspace.')

    def _pose_in_end_effector_frame(self, robot: Robot, action: np.ndarray):
        a_x, a_y, a_z, a_qx, a_qy, a_qz, a_qw = action
        x, y, z, qx, qy, qz, qw = robot.arm.get_tip().get_pose()
        new_rot = Quaternion(
            a_qw, a_qx, a_qy, a_qz) * Quaternion(qw, qx, qy, qz)
        qw, qx, qy, qz = list(new_rot)
        pose = [a_x + x, a_y + y, a_z + z] + [qx, qy, qz, qw]
        return pose

    def action(self, scene: Scene, action: np.ndarray):
        assert_action_shape(action, (7,))
        assert_unit_quaternion(action[3:])
        if not self._absolute_mode and self._frame != RelativeFrame.EE:
            action = calculate_delta_pose(scene.robot, action)
        relative_to = None if self._frame == RelativeFrame.WORLD else scene.robot.arm.get_tip()
        self._quick_boundary_check(scene, action)

        colliding_shapes = []
        if self._collision_checking:
            if self._robot_shapes is None:
                self._robot_shapes = scene.robot.arm.get_objects_in_tree(
                    object_type=ObjectType.SHAPE)
            # First check if we are colliding with anything
            colliding = scene.robot.arm.check_arm_collision()
            if colliding:
                # Disable collisions with the objects that we are colliding with
                grasped_objects = scene.robot.gripper.get_grasped_objects()
                colliding_shapes = [
                    s for s in scene.pyrep.get_objects_in_tree(
                        object_type=ObjectType.SHAPE) if (
                            s.is_collidable() and
                            s not in self._robot_shapes and
                            s not in grasped_objects and
                            scene.robot.arm.check_arm_collision(
                                s))]
                [s.set_collidable(False) for s in colliding_shapes]

        try:
            path = scene.robot.arm.get_path(
                action[:3],
                quaternion=action[3:],
                ignore_collisions=not self._collision_checking,
                relative_to=relative_to,
                trials=100,
                max_configs=10,
                max_time_ms=10,
                trials_per_goal=5,
                algorithm=Algos.RRTConnect
            )
            [s.set_collidable(True) for s in colliding_shapes]
        except ConfigurationPathError as e:
            [s.set_collidable(True) for s in colliding_shapes]
            raise InvalidActionError(
                'A path could not be found. Most likely due to the target '
                'being inaccessible or a collison was detected.') from e
        done = False
        while not done:
            done = path.step()
            scene.step()
            success, terminate = scene.task.success()
            # If the task succeeds while traversing path, then break early
            if success:
                break

    def action_shape(self, scene: Scene) -> tuple:
        return 7,

class EndEffectorPoseViaPlanning_Record(ArmActionMode):
    """High-level action where target pose is given and reached via planning.

    Given a target pose, a linear path is first planned (via IK). If that fails,
    sample-based planning will be used. The decision to apply collision
    checking is a crucial trade off! With collision checking enabled, you
    are guaranteed collision free paths, but this may not be applicable for task
    that do require some collision. E.g. using this mode on pushing object will
    mean that the generated path will actively avoid not pushing the object.

    Note that path planning can be slow, often taking a few seconds in the worst
    case.

    This was the action mode used in:
    James, Stephen, and Andrew J. Davison. "Q-attention: Enabling Efficient
    Learning for Vision-based Robotic Manipulation."
    arXiv preprint arXiv:2105.14829 (2021).
    """

    def __init__(self,
                 absolute_mode: bool = True,
                 frame: RelativeFrame = RelativeFrame.WORLD,
                 collision_checking: bool = False):
        """
        If collision check is enbled, and an object is grasped, then we

        Args:
            absolute_mode: If we should opperate in 'absolute', or 'delta' mode.
            frame: Either WORLD or EE.
            collision_checking: IF collision checking is enabled.
        """
        self._absolute_mode = absolute_mode
        self._frame = frame
        self._collision_checking = collision_checking
        self._robot_shapes = None
        self.action_commands = []
        self.dones = []
        self.current_obs_list = []
        self.next_obs_list = []
        self.dones = []
        # If the task succeeds while traversing path, then break early
        self.termination = []
        self.success_status = []

    def _quick_boundary_check(self, scene: Scene, action: np.ndarray):
        pos_to_check = action[:3]
        relative_to = None if self._frame == RelativeFrame.WORLD else scene.robot.arm.get_tip()
        if relative_to is not None:
            scene.target_workspace_check.set_position(pos_to_check, relative_to)
            pos_to_check = scene.target_workspace_check.get_position()
        if not scene.check_target_in_workspace(pos_to_check):
            raise InvalidActionError('A path could not be found because the '
                                     'target is outside of workspace.')

    def _pose_in_end_effector_frame(self, robot: Robot, action: np.ndarray):
        a_x, a_y, a_z, a_qx, a_qy, a_qz, a_qw = action
        x, y, z, qx, qy, qz, qw = robot.arm.get_tip().get_pose()
        new_rot = Quaternion(
            a_qw, a_qx, a_qy, a_qz) * Quaternion(qw, qx, qy, qz)
        qw, qx, qy, qz = list(new_rot)
        pose = [a_x + x, a_y + y, a_z + z] + [qx, qy, qz, qw]
        return pose

    def action(self, scene: Scene, action: np.ndarray):
        assert_action_shape(action, (7,))
        assert_unit_quaternion(action[3:])
        if not self._absolute_mode and self._frame != RelativeFrame.EE:
            action = calculate_delta_pose(scene.robot, action)
        relative_to = None if self._frame == RelativeFrame.WORLD else scene.robot.arm.get_tip()
        self._quick_boundary_check(scene, action)

        colliding_shapes = []
        if self._collision_checking:
            if self._robot_shapes is None:
                self._robot_shapes = scene.robot.arm.get_objects_in_tree(
                    object_type=ObjectType.SHAPE)
            # First check if we are colliding with anything
            colliding = scene.robot.arm.check_arm_collision()
            if colliding:
                # Disable collisions with the objects that we are colliding with
                grasped_objects = scene.robot.gripper.get_grasped_objects()
                colliding_shapes = [
                    s for s in scene.pyrep.get_objects_in_tree(
                        object_type=ObjectType.SHAPE) if (
                            s.is_collidable() and
                            s not in self._robot_shapes and
                            s not in grasped_objects and
                            scene.robot.arm.check_arm_collision(
                                s))]
                [s.set_collidable(False) for s in colliding_shapes]

        try:
            path = scene.robot.arm.get_path(
                action[:3],
                quaternion=action[3:],
                ignore_collisions=not self._collision_checking,
                relative_to=relative_to,
                trials=50,
                max_configs=5,
                max_time_ms=5,
                trials_per_goal=3,
                algorithm=Algos.RRTConnect
            )
            [s.set_collidable(True) for s in colliding_shapes]
        except ConfigurationPathError as e:
            [s.set_collidable(True) for s in colliding_shapes]
            raise InvalidActionError(
                'A path could not be found. Most likely due to the target '
                'being inaccessible or a collison was detected.') from e
        done = False
        
        while not done:
            done = path.step()
            current_obs = obs_to_dict(scene.get_observation())
            this_action = np.array(scene.robot.arm.get_joint_target_positions())-np.array(scene.robot.arm.get_joint_positions())
            scene.step()
            next_obs = obs_to_dict(scene.get_observation())
            
            self.action_commands.append(this_action)
            #self.action_commands.append(path._joint_position_action)
            self.current_obs_list.append(current_obs)
            self.next_obs_list.append(next_obs)
            self.dones.append(done)
            success, terminate = scene.task.success()
            # If the task succeeds while traversing path, then break early
            self.termination.append(terminate)
            self.success_status.append(success)

            if success:
                break

    def action_shape(self, scene: Scene) -> tuple:
        return 7,

class EndEffectorPoseViaPlanning_Custom(ArmActionMode):
    """High-level action where target pose is given and reached via planning.

    Given a target pose, a linear path is first planned (via IK). If that fails,
    sample-based planning will be used. The decision to apply collision
    checking is a crucial trade off! With collision checking enabled, you
    are guaranteed collision free paths, but this may not be applicable for task
    that do require some collision. E.g. using this mode on pushing object will
    mean that the generated path will actively avoid not pushing the object.

    Note that path planning can be slow, often taking a few seconds in the worst
    case.

    This was the action mode used in:
    James, Stephen, and Andrew J. Davison. "Q-attention: Enabling Efficient
    Learning for Vision-based Robotic Manipulation."
    arXiv preprint arXiv:2105.14829 (2021).
    """

    def __init__(self,
                 absolute_mode: bool = True,
                 frame: RelativeFrame = RelativeFrame.WORLD,
                 collision_checking: bool = False):
        """
        If collision check is enbled, and an object is grasped, then we

        Args:
            absolute_mode: If we should opperate in 'absolute', or 'delta' mode.
            frame: Either WORLD or EE.
            collision_checking: IF collision checking is enabled.
        """
        self._absolute_mode = absolute_mode
        self._frame = frame
        self._collision_checking = collision_checking
        self._robot_shapes = None

    def _quick_boundary_check(self, scene: Scene, action: np.ndarray):
        pos_to_check = action[:3]
        relative_to = None if self._frame == RelativeFrame.WORLD else scene.robot.arm.get_tip()
        if relative_to is not None:
            scene.target_workspace_check.set_position(pos_to_check, relative_to)
            pos_to_check = scene.target_workspace_check.get_position()
        if not scene.check_target_in_workspace(pos_to_check):
            raise InvalidActionError('A path could not be found because the '
                                     'target is outside of workspace.')

    def _pose_in_end_effector_frame(self, robot: Robot, action: np.ndarray):
        a_x, a_y, a_z, a_qx, a_qy, a_qz, a_qw = action
        x, y, z, qx, qy, qz, qw = robot.arm.get_tip().get_pose()
        new_rot = Quaternion(
            a_qw, a_qx, a_qy, a_qz) * Quaternion(qw, qx, qy, qz)
        qw, qx, qy, qz = list(new_rot)
        pose = [a_x + x, a_y + y, a_z + z] + [qx, qy, qz, qw]
        return pose

    def action(self, scene: Scene, action: np.ndarray):
        assert_action_shape(action, (7,))
        assert_unit_quaternion(action[3:])
        if not self._absolute_mode and self._frame != RelativeFrame.EE:
            action = calculate_delta_pose(scene.robot, action)
        relative_to = None if self._frame == RelativeFrame.WORLD else scene.robot.arm.get_tip()
        self._quick_boundary_check(scene, action)

        colliding_shapes = []
        if self._collision_checking:
            if self._robot_shapes is None:
                self._robot_shapes = scene.robot.arm.get_objects_in_tree(
                    object_type=ObjectType.SHAPE)
            # First check if we are colliding with anything
            colliding = scene.robot.arm.check_arm_collision()
            if colliding:
                # Disable collisions with the objects that we are colliding with
                grasped_objects = scene.robot.gripper.get_grasped_objects()
                colliding_shapes = [
                    s for s in scene.pyrep.get_objects_in_tree(
                        object_type=ObjectType.SHAPE) if (
                            s.is_collidable() and
                            s not in self._robot_shapes and
                            s not in grasped_objects and
                            scene.robot.arm.check_arm_collision(
                                s))]
                [s.set_collidable(False) for s in colliding_shapes]

        try:
            path = scene.robot.arm.get_path(
                action[:3],
                quaternion=action[3:],
                ignore_collisions=not self._collision_checking,
                relative_to=relative_to,
                trials=500,
                max_configs=20,
                max_time_ms=10, # was initially 10
                trials_per_goal=10,
                algorithm=Algos.RRTConnect
            )
            [s.set_collidable(True) for s in colliding_shapes]
        except ConfigurationPathError as e:
            [s.set_collidable(True) for s in colliding_shapes]
            raise InvalidActionError(
                'A path could not be found. Most likely due to the target '
                'being inaccessible or a collison was detected.') from e
        done = False
        while not done:
            done = path.step()
            scene.step()
            success, terminate = scene.task.success()
            # If the task succeeds while traversing path, then break early
            if success:
                break

    def action_shape(self, scene: Scene) -> tuple:
        return 7,

class MoveArmThenGripperWithBounds(MoveArmThenGripper):
    """Same as MoveArmThenGripper, but with fixed action bounds.

    This method clamps the first 7 values (x, y, z, qx, qy, qz, qw)
    to ±0.1, and the 8th (gripper) to [0.0, 0.04].
    """

    def __init__(self, action_limits=None,*args, **kwargs):
        # call parent constructor
        self.specified_action_bounds = action_limits 
        super().__init__(*args, **kwargs)
        # any additional initialization can go here
        # e.g. self.some_setting = True
        

    def action_bounds(self):
        if self.specified_action_bounds is None:
            PANDA_LOWER = np.array([
                -2.8973,   # J0
                -1.7628,   # J1
                -2.8973,   # J2
                -3.0718,   # J3
                -2.8973,   # J4
                -0.0175,   # J5
                -2.8973,   # J6
                0.0       # gripper closed
            ], dtype=np.float32)

            PANDA_UPPER = np.array([
                2.8973,   # J0
                1.7628,   # J1
                2.8973,   # J2
                -0.0698,   # J3  (still negative!)
                2.8973,   # J4
                3.7525,   # J5
                2.8973,   # J6
                0.04      # gripper fully open (RL‑Bench convention)
            ], dtype=np.float32)
            return PANDA_LOWER,PANDA_UPPER
        else:
            return self.specified_action_bounds["PANDA_LOWER"], self.specified_action_bounds["PANDA_UPPER"]

class MoveArmThenGripperWithBoundsDelta(MoveArmThenGripper):
    """Same as MoveArmThenGripper, but with fixed action bounds.

    This method clamps the first 7 values (x, y, z, qx, qy, qz, qw)
    to ±0.1, and the 8th (gripper) to [0.0, 0.04].
    """

    def __init__(self, action_limits=None,*args, **kwargs):
        # call parent constructor
        self.specified_action_bounds = action_limits 
        super().__init__(*args, **kwargs)
        # any additional initialization can go here
        # e.g. self.some_setting = True

    def action_bounds(self):

        # ─── per-step joint delta (radians) ───
        #  ±0.1 rad ≈ ±5.7° per control step
        
        # ─── per-step gripper delta (meters) ───
        #  Panda gripper open range is [0.0, 0.04], so a ±0.02 step
        DELTA_JOINT = 0.15
        DELTA_GRIP   = 0.02

        # Build the lower/upper arrays
        PANDA_LOWER = np.array(
            [-DELTA_JOINT] * 7 + [0.0],
            dtype=np.float32,
        )
        PANDA_UPPER = np.array(
            [ DELTA_JOINT] * 7 + [ 0.04],
            dtype=np.float32,
        )

        return PANDA_LOWER, PANDA_UPPER
    """
    def action_bounds(self):

        # ─── per-step joint delta (radians) ───
        #  ±0.1 rad ≈ ±5.7° per control step
        #DELTA_JOINT_LOWER = [-0.15,-0.1,-0.1,-0.15,-0.3,-0.1,-0.3,0.0]
        #DELTA_JOINT_UPPER = [0.15, 0.1, 0.1, 0.15, 0.3,0.1,0.3,0.04]
        
        
        
        DELTA_JOINT_LOWER = [-0.06,-0.06,-0.06,-0.15,-0.1,-0.1,-0.3,0.0]
        DELTA_JOINT_UPPER = [0.06, 0.06, 0.06, 0.15, 0.1,0.1,0.3,0.04]
        
        # Build the lower/upper arrays
        PANDA_LOWER = np.array(
            DELTA_JOINT_LOWER,
            dtype=np.float32,
        )
        PANDA_UPPER = np.array(
            DELTA_JOINT_UPPER,
            dtype=np.float32,
        )

        return PANDA_LOWER, PANDA_UPPER
    """

class MoveArmThenGripperWithBoundsDelta_IVK(MoveArmThenGripper):
    """Same as MoveArmThenGripper, but with fixed action bounds.

    This method clamps the first 7 values (x, y, z, qx, qy, qz, qw)
    to ±0.1, and the 8th (gripper) to [0.0, 0.04].
    """

    def __init__(self, action_limits=None,*args, **kwargs):
        # call parent constructor
        self.specified_action_bounds = action_limits 
        super().__init__(*args, **kwargs)
        # any additional initialization can go here
        # e.g. self.some_setting = True

    def action_bounds(self):

        # ─── per-step joint delta (radians) ───
        #  ±0.1 rad ≈ ±5.7° per control step
        
        # ─── per-step gripper delta (meters) ───
        #  Panda gripper open range is [0.0, 0.04], so a ±0.02 step
        DELTA_JOINT = 0.5
        DELTA_GRIP   = 0.02

        # Build the lower/upper arrays
        PANDA_LOWER = np.array(
            [-DELTA_JOINT] * 7 + [-DELTA_GRIP],
            dtype=np.float32,
        )
        PANDA_UPPER = np.array(
            [ DELTA_JOINT] * 7 + [ DELTA_GRIP],
            dtype=np.float32,
        )

        return PANDA_LOWER, PANDA_UPPER

class MoveArmThenGripperWithBounds_DEPRECATED(ActionMode):
    """Same as MoveArmThenGripper, but defines action_bounds()."""

    def __init__(self, arm_action_mode, gripper_action_mode):
        super().__init__(arm_action_mode, gripper_action_mode)

    def action(self, scene: Scene, action: np.ndarray):
        arm_act_size = np.prod(self.arm_action_mode.action_shape(scene))
        arm_action = np.array(action[:arm_act_size])
        gripper_action = np.array(action[arm_act_size:])
        self.arm_action_mode.action(scene, arm_action)
        gripper_action[0]=1.0
        self.gripper_action_mode.action(scene, gripper_action)

    def action_shape(self, scene: Scene):
        return (np.prod(self.arm_action_mode.action_shape(scene)) 
                + np.prod(self.gripper_action_mode.action_shape(scene)))

    def action_bounds(self):
        joint_bounds = 0.5
        return np.array(7 * [-joint_bounds] + [0.0]), np.array(7 * [joint_bounds] + [0.04])
