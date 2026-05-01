"""
Gymnasium environment for the planar Push-T task in Drake, using IiwaHardwareStation.

The environment wraps a full Drake simulation of the Kuka Iiwa arm pushing the
small_t_pusher slider, matching the setup used in run_sim_sim_eval.py.

Observation (6D), all expressed relative to the goal pose:
  [pusher_x - goal_x, pusher_y - goal_y,
   slider_x - goal_x, slider_y - goal_y,
   cos(slider_theta - goal_theta), sin(slider_theta - goal_theta)]

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
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Build the Drake simulation (expensive, done once per process)
        self._setup_simulation(cfg_path, meshcat)

        # Episode tracking
        self._current_sim_time = 0.0
        self._step_count = 0

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

        # Visualize the target slider pose in Meshcat when available
        if meshcat is not None:
            self._environment.visualize_desired_slider_pose()

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
        """Read current observation from the Drake MultibodyPlant context.

        All positions and orientations are expressed relative to the goal pose so
        that inputs are centered near zero with comparable x/y scales.
        """
        # Slider pose (from generalized coords: [qw,qx,qy,qz, x,y,z])
        slider_q = self._plant.GetPositions(self._mbp_context, self._slider_model_instance)
        slider_pose = PlanarPose.from_generalized_coords(slider_q)

        # Pusher end-effector position in world frame
        pusher_tf = self._plant.EvalBodyPoseInWorld(self._mbp_context, self._pusher_body)
        pusher_x = pusher_tf.translation()[0]
        pusher_y = pusher_tf.translation()[1]

        rel_theta = slider_pose.theta - self._slider_goal_pose.theta

        return np.array(
            [
                pusher_x - self._slider_goal_pose.x,
                pusher_y - self._slider_goal_pose.y,
                slider_pose.x - self._slider_goal_pose.x,
                slider_pose.y - self._slider_goal_pose.y,
                np.cos(rel_theta),
                np.sin(rel_theta),
            ],
            dtype=np.float32,
        )

    def _compute_reward(self, obs: np.ndarray, debug: bool = False):
        """
        Dense reward ported from ManiSkill planar_push_t.py:554-598.
        Normalized to [0, 1] (max reward = 1.0 on success).

        obs is the relative observation:
          [pusher_rel_x, pusher_rel_y, slider_rel_x, slider_rel_y, cos(rel_theta), sin(rel_theta)]

        If debug=True, prints all reward components on a single overwritten line.
        """
        pusher_rel_x, pusher_rel_y = obs[0], obs[1]
        slider_rel_x, slider_rel_y = obs[2], obs[3]

        # Rotational alignment: peaks at 1/2 when slider matches goal orientation, 0 when 180° off
        # obs[4] = cos(slider_theta - goal_theta), which is exactly what rot_rew needs
        rot_component = ((obs[4] + 1.0) / 2.0) ** 2 / 2.0

        # Translational alignment: peaks at 1/2 when slider is at goal position, decays with distance
        dist_T = np.hypot(slider_rel_x, slider_rel_y)
        trans_component = ((1.0 - np.tanh(5.0 * dist_T)) ** 2) / 2.0

        # End-effector proximity: small bonus encouraging the pusher to stay close to the slider
        dist_push = np.hypot(pusher_rel_x - slider_rel_x, pusher_rel_y - slider_rel_y)
        ee_component = max(1.0 - np.tanh(5.0 * dist_push), 0.0) / 1.0

        reward = rot_component + trans_component + ee_component

        # Overlap-based success check (needs absolute coordinates)
        slider_x = slider_rel_x + self._slider_goal_pose.x
        slider_y = slider_rel_y + self._slider_goal_pose.y
        slider_theta = np.arctan2(obs[5], obs[4]) + self._slider_goal_pose.theta
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
            reward = 10.0

        if debug:
            print(
                f"\r  rot={rot_component:.3f}  trans={trans_component:.3f}"
                f"  ee={ee_component:.3f}  total={reward / 3.0:.3f}"
                f"  overlap={overlap:.3f}  success={success}     ",
                end="",
                flush=True,
            )

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

        self._rl_source.set_action(np.array([self._pusher_start_pose.x, self._pusher_start_pose.y]))

        # time_offset lets callers (e.g. multi-episode eval recording) keep Drake's
        # sim clock monotonically increasing so Meshcat frames don't overwrite each other.
        time_offset = float((options or {}).get("time_offset", 0.0))

        context = self._simulator.get_mutable_context()
        context.SetTime(time_offset)

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
        self._simulator.AdvanceTo(time_offset + self._sim_config.time_step)
        self._current_sim_time = time_offset + self._sim_config.time_step
        self._step_count = 0

        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray):
        # Use actual EE position as integration base to prevent wind-up when DiffIK is stuck.
        # If the robot didn't move (IK failed), the next delta is still relative to where it actually is.
        pusher_tf = self._plant.EvalBodyPoseInWorld(self._mbp_context, self._pusher_body)
        actual_xy = pusher_tf.translation()[:2].copy()

        # Integrate delta action into absolute position command, clipped to a
        # circle of radius CLIP_RADIUS around the goal pose to keep the robot within IK reach
        candidate = actual_xy + action * self._rl_cfg.get("action_scale", 0.05)
        center = np.array([self._slider_goal_pose.x, self._slider_goal_pose.y])
        displacement = candidate - center
        dist = np.linalg.norm(displacement)
        CLIP_RADIUS = 0.2
        if dist > CLIP_RADIUS:
            candidate = center + displacement * (CLIP_RADIUS / dist)
        self._rl_source.set_action(candidate)

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
