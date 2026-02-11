import logging
import os
import time

import matplotlib

matplotlib.use("WebAgg")

import numpy as np
from gcs_planar_pushing.experiments.utils import get_default_plan_config, get_default_solver_params
from gcs_planar_pushing.geometry.planar.planar_pushing_trajectory import PlanarPushingTrajectory

# GCS Planner imports
from gcs_planar_pushing.planning.planar.mpc import PlanarPushingMPC
from gcs_planar_pushing.planning.planar.planar_plan_config import PlanarPushingStartAndGoal
from pydrake.all import (
    AbstractValue,
    Context,
    LeafSystem,
    Meshcat,
    Rgba,
    RigidTransform,
)

from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.simulation.planar_pushing_sim_config import PlanarPushingSimConfig

logger = logging.getLogger(__name__)

# Set the print precision to 4 decimal places
np.set_printoptions(precision=4)


class GcsPlannerController(LeafSystem):
    """
    Generate actions using GCS-based MPC planner as described in https://arxiv.org/pdf/2402.10312
    and implemented in https://github.com/Michaelszeng/planning-through-contact.

    Note: this planner uses ground truth state information of both slider and pusher.
    """

    def __init__(
        self,
        sim_config: PlanarPushingSimConfig,
        meshcat: Meshcat = None,
        delay: float = 1.0,
        freq: float = 2.0,
        slow_down_factor: float = 1.0,
        debug: bool = False,
    ):
        super().__init__()
        self._sim_config = sim_config
        self._delay = delay
        self._freq = freq
        self._slow_down_factor = slow_down_factor
        self._debug = debug
        self._meshcat = meshcat

        # Initialize state tracking variables
        self._traj_start_time = None  # Time when the GCS trajectory execution actually starts (when run_flag goes high)
        self._current_action = np.array([0.0, 0.0])
        self._time = 0.0
        self._last_plan_step = -1  # Integer step index

        # Parameters for GCS Planner
        config = get_default_plan_config(
            slider_type="arbitrary",
            arbitrary_shape_pickle_path=self._sim_config.arbitrary_shape_pickle_path,
            slider_physical_properties=self._sim_config.slider_physical_properties,
            pusher_radius=self._sim_config.pusher_radius,
            use_case="drake_iiwa",
        )
        self.planner_config = config
        solver_params = get_default_solver_params()
        self.solver_params = solver_params
        self.traj = None

        # Input port for pusher pose
        self.pusher_pose = self.DeclareAbstractInputPort(
            "pusher_pose_measured",
            AbstractValue.Make(RigidTransform()),
        )

        # Input port for pusher velocity
        self.pusher_velocity = self.DeclareVectorInputPort(
            "pusher_velocity_measured",
            2,
        )

        # Input port for Slider pose
        self.slider_pose = self.DeclareAbstractInputPort(
            "slider_pose_measured",
            AbstractValue.Make(RigidTransform()),
        )

        # Input port for run flag
        self.run_flag = self.DeclareVectorInputPort(
            "run_flag",
            1,
        )

        self.output = self.DeclareVectorOutputPort("planar_position_command", 2, self.DoCalcOutput)

        # Debug ports for logging
        self.debug_action = self.DeclareVectorOutputPort("debug_action", 2, self.DoCalcDebugAction)
        self.debug_pusher_pose = self.DeclareVectorOutputPort("debug_pusher_pose", 2, self.DoCalcDebugPusherPose)

    def DoCalcOutput(self, context: Context, output):
        _time = context.get_time()
        self._time = _time

        # Check run flag; only start planning when run_flag is 1
        if self.run_flag.Eval(context)[0] == 0:
            output.set_value(self._current_action)
            self._traj_start_time = None
            return

        if self._traj_start_time is None:  # If we haven't started yet, mark start time
            self._traj_start_time = _time

        # Calculate time based on when run_flag went high
        time_since_traj_start = (_time - self._traj_start_time) / self._slow_down_factor + 1e-3

        # Get planar poses for Slider and Pusher and velocity of pusher
        current_slider_rigid_transform = self.slider_pose.Eval(context)
        current_pusher_rigid_transform = self.pusher_pose.Eval(context)
        current_slider_pose = PlanarPose.from_pose(current_slider_rigid_transform)
        current_pusher_pose = PlanarPose.from_pose(current_pusher_rigid_transform)
        current_pusher_vel = self.pusher_velocity.Eval(context)
        # current_pusher_vel = None
        # current_slider_pose = self._gcs_planner.original_traj.get_slider_planar_pose(time_in_traj_to_retrieve_action)
        # current_pusher_pose = self._gcs_planner.original_traj.get_pusher_planar_pose(time_in_traj_to_retrieve_action)
        # current_pusher_vel = self._gcs_planner.original_traj.get_pusher_velocity(time_in_traj_to_retrieve_action)

        # At next update cycle, generate new trajectory prediction with the fixed mode sequence
        current_step = int(_time * self._freq)
        if current_step > self._last_plan_step:
            self._last_plan_time = _time
            self._last_plan_step = current_step
            print(f"Sim time: {_time:.4f}s | Current step: {current_step}")
            print(f"time_since_traj_start: {time_since_traj_start:.4f}")
            print(f"    current_slider_pose: {current_slider_pose}")
            print(f"    current_pusher_pose: {current_pusher_pose}")
            if current_pusher_vel is not None:
                print(f"    current_pusher_vel: {current_pusher_vel} (magnitude: {np.linalg.norm(current_pusher_vel)})")
            start = time.time()
            path = self._gcs_planner.plan(
                t=time_since_traj_start,
                current_slider_pose=current_slider_pose,
                current_pusher_pose=current_pusher_pose,
                current_pusher_velocity=current_pusher_vel,
                # save_video=True,
                # output_folder="temp_videos",
                # output_name=f"traj_{current_step}.mp4",
            )
            print(f"    GCS Planner planning time: {time.time() - start:.3f}s")
            # TODO: converting the PlanarPushingPath to a PlanarPushingTrajectory with path.to_traj() takes ~0.05 sec
            # (which is a lot), so we should try to avoid doing this.
            self.traj = path.to_traj(rounded=True)
            self.traj.plot_velocity_profile(save_plot=f"temp_vel_profiles/{current_step}")

        # Output Action from trajectory prediction
        time_in_traj_to_retrieve_action = (_time - self._last_plan_time) / self._slow_down_factor + 1e-3
        self._current_action = self.traj.get_pusher_planar_pose(time_in_traj_to_retrieve_action).vector()[:2]
        formatter = {'float_kind': lambda x: f"{x:.10f}"}
        print(f"self._current_action: {np.array2string(self._current_action, formatter=formatter)}")
        print(f"current_pusher_pose: {np.array2string(current_pusher_pose.vector()[:2], formatter=formatter)}")

        # Visualize new predicted trajectory using fixed mode sequence in meshcat
        self._visualize_trajectories(self.traj, Rgba(178 / 255, 34 / 255, 34 / 255, 1.0), _time)

        # Obtain and output current action from trajectory prediction
        output.set_value(self._current_action)

        # debug print statements
        if self._debug:
            print(f"Time: {_time:.3f}, action: {self._current_action}")

    def DoCalcDebugAction(self, context, output):
        output.set_value(self._current_action)

    def DoCalcDebugPusherPose(self, context, output):
        # Re-evaluate pusher pose from input port
        current_pusher_rigid_transform = self.pusher_pose.Eval(context)
        current_pusher_pose = PlanarPose.from_pose(current_pusher_rigid_transform)
        output.set_value(current_pusher_pose.vector()[:2])

    def reset(self, pusher_reset_position: np.ndarray = None, new_slider_start_pose: PlanarPose = None):
        """
        Upon trial reset, set current action to the reset position and generate new GCS plan.
        """
        if pusher_reset_position is not None:
            self._current_action = pusher_reset_position

        # Create GCS planner and do a full plan, generating the mode sequence
        start_and_goal = PlanarPushingStartAndGoal(
            slider_initial_pose=new_slider_start_pose,
            slider_target_pose=self._sim_config.slider_goal_pose,
            pusher_initial_pose=self._sim_config.pusher_start_pose,
            pusher_target_pose=self._sim_config.pusher_start_pose,  # Pusher has same start and target pose
        )
        print(f"slider_initial_pose: {new_slider_start_pose}")
        print(f"pusher_initial_pose: {self._sim_config.pusher_start_pose}")
        print(f"slider_target_pose: {self._sim_config.slider_goal_pose}")
        print(f"pusher_target_pose: {self._sim_config.pusher_start_pose}")
        print("... Creating GCS Plan...", flush=True)

        # For ease of testing, load cached path if it exists (else compute fresh path and cache it)
        CACHE_PATH = (
            f"mpc_path_cache_{new_slider_start_pose.x:.2f}_"
            f"{new_slider_start_pose.y:.2f}_{new_slider_start_pose.theta:.2f}.pkl"
        )
        if os.path.exists(CACHE_PATH):
            print(f"Loading cached path from {CACHE_PATH}")
            self._gcs_planner = PlanarPushingMPC(
                self.planner_config,
                start_and_goal,
                self.solver_params,
                plan=False,
            )
            self._gcs_planner.load_original_path(CACHE_PATH)
        else:
            print("Computing fresh path and caching it...")
            self._gcs_planner = PlanarPushingMPC(
                self.planner_config,
                start_and_goal,
                self.solver_params,
                plan=True,
            )
            self._gcs_planner.original_path.save(CACHE_PATH)

        print("✓ GCS Plan created successfully", flush=True)

        # Visualize original GCS plan in meshcat
        # Since this doesn't change throughout a trial, we only need to visualize once here
        self._visualize_trajectories(self._gcs_planner.original_traj, Rgba(0.0, 0.0, 0.0, 1.0), self._time)

    def _visualize_trajectories(self, trajectory: PlanarPushingTrajectory, color: Rgba, time_in_recording: float):
        """Visualize multiple predicted trajectories in meshcat with points at each timestep."""
        if self._meshcat is None:
            return

        # Build matrix of 3D positions by sampling the trajectory
        NUM_STEPS = 50
        pos_3d_matrix = np.zeros((3, NUM_STEPS))
        for idx, vis_t in enumerate(np.linspace(trajectory.start_time, trajectory.end_time, NUM_STEPS)):
            p_WP = trajectory.get_value(vis_t, "p_WP")  # pusher position in world frame
            pos_3d_matrix[0, idx] = p_WP[0, 0]  # x
            pos_3d_matrix[1, idx] = p_WP[1, 0]  # y
            pos_3d_matrix[2, idx] = 0.0  # z (=0 for planar pushing)

        # Use a stable name per color (reuse the same line object)
        name = f"trajectory_{color.r():.2f}_{color.g():.2f}_{color.b():.2f}"

        # Update the line geometry (this updates in place for the same path)
        self._meshcat.SetLine(f"{name}/line", pos_3d_matrix, line_width=5.0, rgba=color)

        # # Visualize segment endpoints as spheres
        # for i, t in enumerate(trajectory.end_times):
        #     p_WP = trajectory.get_value(t, "p_WP")
        #     sphere_path = f"{name}/segment_end_{i}"
        #     self._meshcat.SetObject(sphere_path, Sphere(0.002), rgba=color)
        #     self._meshcat.SetTransform(sphere_path, RigidTransform(np.array([p_WP[0, 0], p_WP[1, 0], 0.0])))

        # For recording: ensure visibility at the current time by setting a property change
        self._meshcat.SetProperty(name, "visible", True, time_in_recording)


def plot_gcs_controller_logs(action_log, pusher_log, root_context):
    """
    Helper to plot the logged data.

    Args:
        action_log: VectorLogSink for action
        pusher_log: VectorLogSink for pusher pose
        root_context: Context of the Diagram (root context)
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    # Retrieve logs
    action_log_data = action_log.FindLog(root_context)
    pusher_log_data = pusher_log.FindLog(root_context)

    action_times = action_log_data.sample_times()
    action_values = action_log_data.data()
    pusher_times = pusher_log_data.sample_times()
    pusher_values = pusher_log_data.data()

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.25)

    # Plot full trajectories
    ax.plot(
        action_values[0, :],
        action_values[1, :],
        label="Action Path",
        color="blue",
        alpha=0.3,
    )
    ax.plot(
        pusher_values[0, :],
        pusher_values[1, :],
        "--",
        label="Pusher Path",
        color="green",
        alpha=0.3,
    )

    # Current points markers
    (action_point,) = ax.plot([], [], "bo", markersize=10, label="Action")
    (pusher_point,) = ax.plot([], [], "gx", markersize=10, label="Pusher")

    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_title("Trajectory (Time Scrollable)")
    ax.legend()
    ax.grid(True)
    ax.axis("equal")

    # Time Slider
    t_min, t_max = min(action_times[0], pusher_times[0]), max(action_times[-1], pusher_times[-1])
    ax_time = plt.axes([0.2, 0.1, 0.65, 0.03])
    # Use action_times for discrete steps, assuming action updates are the replan events
    time_slider = Slider(ax_time, "Time", t_min, t_max, valinit=t_min, valstep=action_times)

    def update(val):
        t = time_slider.val
        # Find closest indices
        idx_a = np.searchsorted(action_times, t)
        idx_a = min(idx_a, len(action_times) - 1)
        idx_p = np.searchsorted(pusher_times, t)
        idx_p = min(idx_p, len(pusher_times) - 1)

        action_point.set_data([action_values[0, idx_a]], [action_values[1, idx_a]])
        pusher_point.set_data([pusher_values[0, idx_p]], [pusher_values[1, idx_p]])
        fig.canvas.draw_idle()

    time_slider.on_changed(update)
    update(t_min)
    plt.show()
