"""
Gymnasium environment for the planar Push-T task in Drake, using IiwaHardwareStation.

The environment wraps a full Drake simulation of the Kuka Iiwa arm pushing the
small_t_pusher slider, matching the setup used in run_sim_sim_eval.py.

Observation (10D):
  [pusher_x, pusher_y, slider_x, slider_y, cos(slider_theta), sin(slider_theta),
   goal_x, goal_y, cos(goal_theta), sin(goal_theta)]

Action (2D):
  [delta_x, delta_y] in [-1, 1], scaled by action_scale meters/step
  Integrated into an absolute position command sent to the robot's DiffIK.

Episode structure:
  - reset() advances through delay_before_execution (robot goes to home) automatically
  - step() advances control_dt seconds each call
  - Terminated on overlap >= success_overlap_thresh, truncated at max_episode_steps

Usage:
  env = PushTDrakeEnv(cfg_path="rl_push_t/configs/rl_env.yaml")
"""

import hydra
import numpy as np
from gymnasium import Env, spaces
from omegaconf import OmegaConf

from planning_through_contact.geometry.collision_checker import CollisionChecker
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.simulation.controllers.iiwa_hardware_station import (
    IiwaHardwareStation,
)
from planning_through_contact.simulation.environments.simulated_real_table_environment import (
    SimulatedRealTableEnvironment,
)
from planning_through_contact.simulation.planar_pushing_sim_config import (
    PlanarPushingSimConfig,
)
from planning_through_contact.simulation.sim_utils import (
    BoxWorkspace,
    PlanarPushingWorkspace,
    get_slider_initial_pose_within_workspace,
)
from rl_push_t.envs.overlap import compute_tee_overlap
from rl_push_t.envs.rl_action_source import RLActionSource


class PushTDrakeEnv(Env):
    """
    Gymnasium environment for the Drake Push-T task using IiwaHardwareStation.

    This environment is compatible with gymnasium.vector.AsyncVectorEnv for
    parallel RL training across multiple CPU processes.

    Args:
        cfg_path: Path to a yaml config file (e.g. "rl_push_t/configs/rl_env.yaml").
            All sim parameters (goal pose, pusher start, timestep, etc.) and RL
            parameters (action_scale, control_dt, etc.) are loaded from the yaml,
            matching the format used by run_sim_sim_eval.py.
        render_mode: Gymnasium render mode (currently unused).
    """

    def __init__(self, cfg_path: str, meshcat=None):
        super().__init__()
        cfg = OmegaConf.load(cfg_path)
        self._slider_goal_pose = hydra.utils.instantiate(cfg.slider_goal_pose)
        self._pusher_start_pose = hydra.utils.instantiate(cfg.pusher_start_pose)
        self._rl_cfg = cfg.get("rl", {})

        # Gym spaces
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Build the Drake simulation (expensive, done once per process)
        self._setup_simulation(cfg_path, meshcat)

        # Episode tracking
        self._current_sim_time = 0.0
        self._step_count = 0
        self._current_position = np.array([self._pusher_start_pose.x, self._pusher_start_pose.y], dtype=np.float64)

    def _setup_simulation(self, cfg_path: str, meshcat=None):
        """Build the Drake diagram. Called once in __init__."""
        cfg = OmegaConf.load(cfg_path)
        self._sim_config = PlanarPushingSimConfig.from_yaml(cfg)
        # from_yaml leaves camera_configs=None when yaml has an empty list
        if self._sim_config.camera_configs is None:
            self._sim_config.camera_configs = []
        if meshcat is not None:
            self._sim_config.use_realtime = True

        # RLActionSource replaces DiffusionPolicySource / GcsPlannerSource
        self._rl_source = RLActionSource(self._pusher_start_pose)

        station = IiwaHardwareStation(self._sim_config, meshcat=meshcat)

        self._environment = SimulatedRealTableEnvironment(
            desired_position_source=self._rl_source,
            robot_system=station,
            sim_config=self._sim_config,
            station_meshcat=meshcat,
        )

        # Cache frequently used references
        self._plant = self._environment._plant
        self._mbp_context = self._environment.mbp_context
        self._slider_model_instance = self._environment._slider_model_instance
        self._robot_model_instance = self._environment._robot_model_instance
        self._simulator = self._environment._simulator
        self._pusher_body = self._plant.GetBodyByName("pusher")

        # Workspace and collision checker for slider pose randomization (matches run_sim_sim_eval.py)
        self._workspace = PlanarPushingWorkspace(
            slider=BoxWorkspace(
                width=self._rl_cfg.get("workspace_width", 0.4),
                height=self._rl_cfg.get("workspace_height", 0.3),
                center=np.array([self._slider_goal_pose.x, self._slider_goal_pose.y]),
                buffer=0,
            )
        )
        self._collision_checker = CollisionChecker(
            cfg.arbitrary_shape_pickle_path,
            cfg.pusher_radius,
            meshcat=None,
        )

    def _get_obs(self) -> np.ndarray:
        """Read current observation from the Drake MultibodyPlant context."""
        # Slider pose (from generalized coords: [qw,qx,qy,qz, x,y,z])
        slider_q = self._plant.GetPositions(self._mbp_context, self._slider_model_instance)
        slider_pose = PlanarPose.from_generalized_coords(slider_q)

        # Pusher end-effector position in world frame
        pusher_tf = self._plant.EvalBodyPoseInWorld(self._mbp_context, self._pusher_body)
        pusher_x = pusher_tf.translation()[0]
        pusher_y = pusher_tf.translation()[1]

        return np.array(
            [
                pusher_x,
                pusher_y,
                slider_pose.x,
                slider_pose.y,
                np.cos(slider_pose.theta),
                np.sin(slider_pose.theta),
                self._slider_goal_pose.x,
                self._slider_goal_pose.y,
                np.cos(self._slider_goal_pose.theta),
                np.sin(self._slider_goal_pose.theta),
            ],
            dtype=np.float32,
        )

    def _compute_reward(self, obs: np.ndarray):
        """
        Dense reward ported from ManiSkill planar_push_t.py:554-598.
        Normalized to [0, 1] (max reward = 1.0 on success).
        """
        pusher_x, pusher_y = obs[0], obs[1]
        slider_x, slider_y = obs[2], obs[3]
        slider_theta = np.arctan2(obs[5], obs[4])  # sin/cos → theta

        # Rotation alignment reward
        rot_rew = np.cos(slider_theta - self._slider_goal_pose.theta)
        reward = ((rot_rew + 1.0) / 2.0) ** 2 / 2.0

        # Translation reward (T center → goal center)
        dist_T = np.hypot(slider_x - self._slider_goal_pose.x, slider_y - self._slider_goal_pose.y)
        reward += ((1.0 - np.tanh(5.0 * dist_T)) ** 2) / 2.0

        # EE proximity reward (pusher → T center)
        dist_push = np.hypot(pusher_x - slider_x, pusher_y - slider_y)
        reward += np.sqrt(max(1.0 - np.tanh(5.0 * dist_push), 0.0)) / 20.0

        # Overlap-based success check
        overlap = compute_tee_overlap(
            slider_x,
            slider_y,
            slider_theta,
            self._slider_goal_pose.x,
            self._slider_goal_pose.y,
            self._slider_goal_pose.theta,
        )
        success = overlap >= self._rl_cfg.get("success_overlap_thresh", 0.9)
        if success:
            reward = 3.0

        return reward / 3.0, success, overlap

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        # Randomize slider initial pose within workspace, avoiding collision with pusher
        # Matches run_sim_sim_eval.py reset_environment() logic
        slider_pose = get_slider_initial_pose_within_workspace(
            self._workspace,
            self._sim_config.slider,
            self._pusher_start_pose,
            self._slider_goal_pose,
            self._collision_checker,
            rng=rng,
        )

        # Reset pusher position tracker
        self._current_position = np.array([self._pusher_start_pose.x, self._pusher_start_pose.y], dtype=np.float64)
        self._rl_source.set_action(self._current_position)

        # Reset sim time and step count
        context = self._simulator.get_mutable_context()
        context.SetTime(0.0)

        # Set initial robot + slider positions BEFORE Initialize so that:
        # (a) IiwaPlanner.Initialize reads q0 = default_joint_positions
        # (b) The GCS trajectory plan is near-trivial (start ≈ goal)
        self._environment.reset(
            self._sim_config.default_joint_positions,
            slider_pose,
            self._pusher_start_pose,
        )

        self._simulator.Initialize()

        # Skip the IiwaPlanner startup delay by forcing PUSHING mode directly.
        # The robot is already at default_joint_positions (set above), so no
        # trajectory planning is needed.
        self._environment._robot_system._planner.force_pushing_mode(self._simulator.get_mutable_context())

        # Advance a single timestep so Drake initializes all state properly
        self._simulator.AdvanceTo(self._sim_config.time_step)
        self._current_sim_time = self._sim_config.time_step
        self._step_count = 0

        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray):
        # Integrate delta action into absolute position command, clipped to a
        # circle of radius CLIP_RADIUS around the goal pose to keep the robot within IK reach
        candidate = self._current_position + action * self._rl_cfg.get("action_scale", 0.01)
        center = np.array([self._slider_goal_pose.x, self._slider_goal_pose.y])
        displacement = candidate - center
        dist = np.linalg.norm(displacement)
        CLIP_RADIUS = 0.2
        if dist > CLIP_RADIUS:
            candidate = center + displacement * (CLIP_RADIUS / dist)
        self._current_position = candidate
        self._rl_source.set_action(self._current_position)

        # Advance Drake simulation by one control step
        self._current_sim_time += self._rl_cfg.get("control_dt", 0.1)
        self._simulator.AdvanceTo(self._current_sim_time)
        self._step_count += 1

        obs = self._get_obs()
        reward, success, overlap = self._compute_reward(obs)
        terminated = success
        truncated = self._step_count >= self._rl_cfg.get("max_episode_steps", 200)
        info = {"overlap": overlap, "success": success}

        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass
