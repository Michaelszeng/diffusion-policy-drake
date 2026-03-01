import importlib
import math
import os
import pathlib
import random
import shutil
import traceback

import cv2
import hydra
import numpy as np
from omegaconf import OmegaConf
from pydrake.all import StartMeshcat

from planning_through_contact.geometry.collision_checker import CollisionChecker
from planning_through_contact.simulation.controllers.gcs_planner_source import (
    GcsPlannerSource,
)
from planning_through_contact.simulation.controllers.iiwa_hardware_station import (
    IiwaHardwareStation,
)
from planning_through_contact.simulation.controllers.robot_system_base import (
    RobotSystemBase,
)
from planning_through_contact.simulation.environments.simulated_real_table_environment import (
    SimulatedRealTableEnvironment,
)
from planning_through_contact.simulation.planar_pushing_sim_config import (
    PlanarPushingSimConfig,
)
from planning_through_contact.simulation.sim_eval_utils import (
    MeshcatRecordingManager,
    SimEvaluator,
)
from planning_through_contact.simulation.sim_utils import (
    get_slider_initial_pose_within_workspace,
)
from planning_through_contact.utils import file_lock
from planning_through_contact.visualize.analysis import (
    PlanarPushingLog,
)

TRIALS_TO_SKIP = []
# TRIALS_TO_SKIP = [0]

ONLY_1_TRIAL = False


class SimSimGcsPlanner:
    def __init__(self, cfg: OmegaConf, collect_data: bool = False):
        station_meshcat = StartMeshcat()

        # load sim_config
        self.cfg = cfg

        # Hold system-wide lock during writing and reading of small_table_hydroelastic.urdf and arbitrary_shape.sdf.
        # After this locked code block is finished, other processes are free to modify these files without affecting
        # this process.
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        table_urdf_lock = os.path.join(
            project_root, "planning_through_contact/simulation/models/small_table_hydroelastic.urdf"
        )
        slider_sdf_lock = os.path.join(project_root, "planning_through_contact/simulation/models/arbitrary_shape.sdf")
        with file_lock(table_urdf_lock):
            with file_lock(slider_sdf_lock):
                self.sim_config = PlanarPushingSimConfig.from_yaml(cfg)
                # Set up position controller (i.e. IiwaHardwareStation)
                module_name, class_name = cfg.robot_station._target_.rsplit(".", 1)
                robot_system_class = getattr(importlib.import_module(module_name), class_name)
                position_controller: RobotSystemBase = robot_system_class(
                    sim_config=self.sim_config, meshcat=station_meshcat
                )

        self.multi_run_config = self.sim_config.multi_run_config
        self.pusher_start_pose = self.sim_config.pusher_start_pose
        self.slider_goal_pose = self.sim_config.slider_goal_pose
        print(f"Initial pusher pose: {self.pusher_start_pose}")
        print(f"Target slider pose: {self.slider_goal_pose}")

        self.workspace = self.multi_run_config.workspace

        self.collision_checker = CollisionChecker(cfg.arbitrary_shape_pickle_path, cfg.pusher_radius, station_meshcat)

        # Set up random seeds
        random.seed(self.multi_run_config.seed)
        np.random.seed(self.multi_run_config.seed)

        # GCS Planner position source
        position_source = GcsPlannerSource(self.sim_config, station_meshcat)

        # Remove existing temporary image writer directory
        image_writer_dir = "trajectories_rendered/temp"
        if os.path.exists(image_writer_dir):
            shutil.rmtree(image_writer_dir)

        # Set up environment
        self.environment = SimulatedRealTableEnvironment(
            desired_position_source=position_source,
            robot_system=position_controller,
            sim_config=self.sim_config,
            station_meshcat=station_meshcat,
            arbitrary_shape_pickle_path=cfg.arbitrary_shape_pickle_path,
        )
        self.environment.export_diagram("sim_sim_environment.svg")

        if isinstance(self.environment._robot_system, IiwaHardwareStation):
            self.run_flag_port = self.environment._robot_system.GetOutputPort("run_flag")
        self.robot_system_context = self.environment.robot_system_context

        self.evaluator = SimEvaluator(self.environment, self.sim_config, self.multi_run_config)

        self.collect_data = collect_data
        self.only_one_trial = ONLY_1_TRIAL and not collect_data
        self.traj_start_time = 0.0
        self.num_saved_trajectories = 0
        self.num_completed_trials = 0
        self._collected_episodes = []
        self._camera_info = {}
        if self.collect_data:
            for key, meta in cfg.data_collection_config.shape_meta.obs.items():
                if meta.get("type") == "rgb":
                    self._camera_info[key] = list(meta.shape)

    def run(
        self,
    ):
        num_runs = len(TRIALS_TO_SKIP) + 1 if self.only_one_trial else self.multi_run_config.num_runs
        max_duration = self.multi_run_config.max_attempt_duration

        success_count = 0
        failure_count = 0
        error_count = 0

        # Get current time from simulator
        current_time = self.environment._simulator.get_context().get_time()
        time_step = self.sim_config.time_step * 10

        output_dir = self.cfg.get("output_dir", ".")
        recorder = MeshcatRecordingManager(
            meshcat=self.environment._meshcat,
            environment=self.environment,
            num_trials_to_record=self.multi_run_config.num_trials_to_record,
            total_runs=num_runs,
            output_dir=output_dir,
            file_prefix="gcs_planner",
        )
        recorder.start()

        if self.collect_data:
            capture_dt = 1.0 / self.cfg.data_collection_config.policy_freq

        try:
            for trial_idx in range(num_runs):
                if trial_idx in TRIALS_TO_SKIP:
                    print(f"\n{'=' * 20} Skipping Trial {trial_idx} (Hardcoded in TRIALS_TO_SKIP) {'=' * 20}")
                    continue

                print(f"\n{'=' * 20} Trial {trial_idx} {'=' * 20}")
                trial_success = False
                if self.collect_data:
                    trial_images = {name: [] for name in self._camera_info}

                try:
                    slider_pose = self.get_slider_pose_for_trial(trial_idx)

                    if trial_idx > 0 and isinstance(self.environment._robot_system, IiwaHardwareStation):
                        print("Returning to start...")
                        self.environment._desired_position_source._gcs_planner._ready = False
                        self.environment.set_slider_planar_pose(slider_pose)
                        self.environment._robot_system._planner.reset()

                        # Simulate until run_flag is True (slider settles during this time)
                        while True:
                            current_time += time_step
                            current_time = round(current_time / time_step) * time_step
                            self.environment._simulator.AdvanceTo(current_time)

                            run_flag = self.run_flag_port.Eval(self.robot_system_context)[0]
                            if run_flag:
                                break

                    self.traj_start_time = current_time
                    self.reset_environment(trial_idx, slider_pose)
                    self.environment.visualize_desired_slider_pose(current_time)
                    self.environment.visualize_desired_pusher_pose(current_time)

                    # Capture initial image (t=0) before simulation advances
                    if self.collect_data:
                        for cam_name in self._camera_info:
                            trial_images[cam_name].append(self._capture_camera_image(cam_name))
                        next_capture_time = current_time + capture_dt

                    # Run trial
                    trial_end_time = current_time + max_duration
                    trial_done = False
                    while current_time < trial_end_time:
                        # Advance simulation
                        current_time += time_step
                        # Rounding to avoid precision issues
                        current_time = round(current_time / time_step) * time_step

                        self.environment._simulator.AdvanceTo(current_time)

                        if self.collect_data and current_time >= next_capture_time - 1e-6:
                            for cam_name in self._camera_info:
                                trial_images[cam_name].append(self._capture_camera_image(cam_name))
                            next_capture_time += capture_dt

                        # Check success
                        if self.evaluator.check_success():
                            success_count += 1
                            trial_success = True
                            print(f"Trial {trial_idx} Result: SUCCESS")
                            trial_done = True
                            break

                        # Check failure
                        failure, mode = self.evaluator.check_failure(current_time, self.traj_start_time)
                        if failure:
                            failure_count += 1
                            print(f"Trial {trial_idx} Result: {mode.value}")
                            trial_done = True
                            break

                    if not trial_done:
                        # If we exited loop due to timeout (current_time >= trial_end_time)
                        # and check_failure didn't catch it yet (or we want to be explicit)
                        failure_count += 1
                        print(f"Trial {trial_idx} Result: TIMEOUT")

                except Exception as e:
                    # Check if we succeeded despite the error (e.g. planner failed because we are at goal)
                    # This often happens with GCS planner when the start and goal are very close
                    if self.evaluator.check_success():
                        success_count += 1
                        trial_success = True
                        print(f"Trial {trial_idx} Result: SUCCESS (ended with error: {e})")
                    else:
                        error_count += 1
                        print(f"Trial {trial_idx} Result: ERROR - {e}")
                        traceback.print_exc()
                    # Continue to next trial
                    pass

                if self.collect_data and trial_success:
                    episode = self._extract_trial_data(trial_images)
                    if episode is not None:
                        self._collected_episodes.append(episode)

                self.num_completed_trials += 1
                print(f"Running Stats: Success={success_count}, Failure={failure_count}, Error={error_count}")

                recorder.on_trial_complete()

        finally:
            recorder.finalize()

            if self.collect_data:
                self._save_to_zarr()

            print("Finished running trials.")

    def get_slider_pose_for_trial(self, trial_idx: int):
        slider = self.sim_config.slider
        ss = np.random.SeedSequence([self.multi_run_config.seed, trial_idx])
        trial_rng = np.random.default_rng(ss)
        return get_slider_initial_pose_within_workspace(
            self.workspace, slider, self.pusher_start_pose, self.slider_goal_pose, self.collision_checker, rng=trial_rng
        )

    def reset_environment(self, trial_idx: int, slider_pose=None):
        """
        Reset environment with new initial slider pose.
        Use seed sequence to ensure deterministic sequence of initial slider poses is produced every run.
        """
        print("=" * 100 + f"\nResetting environment for trial {trial_idx}.")
        if slider_pose is None:
            slider_pose = self.get_slider_pose_for_trial(trial_idx)

        self.environment.reset(
            self.sim_config.default_joint_positions,
            slider_pose,
            self.pusher_start_pose,
        )

    def get_planar_pushing_log(self, vector_log, traj_start_time):
        start_idx = 0
        sample_times = vector_log.sample_times()
        while sample_times[start_idx] < traj_start_time:
            start_idx += 1

        t = sample_times[start_idx:] - sample_times[start_idx]
        nan_array = np.array([float("nan") for _ in t])
        return PlanarPushingLog(
            t=t,
            x=vector_log.data()[0, start_idx:],
            y=vector_log.data()[1, start_idx:],
            theta=vector_log.data()[2, start_idx:],
            lam=nan_array,
            c_n=nan_array,
            c_f=nan_array,
            lam_dot=nan_array,
        )

    def _capture_camera_image(self, camera_name):
        """Capture an RGB image from the named camera, resized to shape_meta dimensions."""
        port = self.environment._robot_system.GetOutputPort(f"rgbd_sensor_{camera_name}")
        rgba = port.Eval(self.environment.robot_system_context).data.copy()
        rgb = rgba[:, :, :3]
        target_h, target_w = self._camera_info[camera_name][1], self._camera_info[camera_name][2]
        if rgb.shape[0] != target_h or rgb.shape[1] != target_w:
            rgb = cv2.resize(rgb, (target_w, target_h))
        return rgb

    def _extract_trial_data(self, trial_images=None):
        """Extract state/action/target/image arrays from the current trial's logs, resampled at policy_freq."""
        dc_cfg = self.cfg.data_collection_config
        freq = dc_cfg.policy_freq
        dt = 1.0 / freq

        pusher_log = self.environment.get_pusher_pose_log()
        pusher = self.get_planar_pushing_log(pusher_log, self.traj_start_time)
        slider_log = self.environment.get_slider_pose_log()
        slider = self.get_planar_pushing_log(slider_log, self.traj_start_time)

        t = pusher.t
        if len(t) < 2:
            return None

        total_time = math.floor(t[-1] * freq) / freq
        states, slider_states = [], []
        current_time = 0.0
        idx = 0

        while current_time < total_time:
            idx = _get_closest_index(t, current_time, idx)
            states.append([pusher.x[idx], pusher.y[idx], pusher.theta[idx]])
            slider_states.append([slider.x[idx], slider.y[idx], slider.theta[idx]])
            current_time = round((current_time + dt) * freq) / freq

        # Align state and image counts (may differ by 1 due to timing edge cases)
        n = len(states)
        if trial_images:
            n = min(n, min(len(imgs) for imgs in trial_images.values()))
        if n < 2:
            return None

        states = np.array(states[:n])
        slider_states = np.array(slider_states[:n])
        actions = np.concatenate([states[1:, :2], states[-1:, :2]], axis=0)
        goal = self.slider_goal_pose.vector()
        targets = np.tile(goal, (n, 1))

        result = {
            "state": states,
            "slider_state": slider_states,
            "action": actions,
            "target": targets,
        }
        if trial_images:
            for cam_name, imgs in trial_images.items():
                result[cam_name] = np.array(imgs[:n], dtype=np.uint8)
        return result

    def _save_to_zarr(self):
        """Save collected episodes to zarr format."""
        import zarr

        if not self._collected_episodes:
            print("No episodes collected, skipping zarr save.")
            return

        dc_cfg = self.cfg.data_collection_config
        zarr_path = dc_cfg.zarr_path

        states = np.concatenate([ep["state"] for ep in self._collected_episodes])
        slider_states = np.concatenate([ep["slider_state"] for ep in self._collected_episodes])
        actions = np.concatenate([ep["action"] for ep in self._collected_episodes])
        targets = np.concatenate([ep["target"] for ep in self._collected_episodes])
        episode_ends = np.cumsum([len(ep["state"]) for ep in self._collected_episodes])

        root = zarr.open_group(zarr_path, mode="w")
        data_group = root.create_group("data")
        meta_group = root.create_group("meta")

        data_group.create_dataset("state", data=states, chunks=(dc_cfg.state_chunk_length, states.shape[1]))
        slider_chunks = (dc_cfg.state_chunk_length, slider_states.shape[1])
        data_group.create_dataset("slider_state", data=slider_states, chunks=slider_chunks)
        data_group.create_dataset("action", data=actions, chunks=(dc_cfg.action_chunk_length, actions.shape[1]))
        data_group.create_dataset("target", data=targets, chunks=(dc_cfg.target_chunk_length, targets.shape[1]))
        meta_group.create_dataset("episode_ends", data=episode_ends)

        for cam_name in self._camera_info:
            all_images = np.concatenate([ep[cam_name] for ep in self._collected_episodes])
            image_chunks = (dc_cfg.image_chunk_length, *all_images.shape[1:])
            data_group.create_dataset(cam_name, data=all_images, chunks=image_chunks, dtype="u1")
            del all_images

        print(f"\nSaved {len(self._collected_episodes)} episodes ({len(states)} total steps) to {zarr_path}")


def _get_closest_index(arr, t, start_idx=0):
    """Returns index of arr closest to t, searching forward from start_idx."""
    min_diff = float("inf")
    min_idx = start_idx
    for i in range(start_idx, len(arr)):
        diff = abs(arr[i] - t)
        if diff > min_diff:
            return min_idx
        if diff < 1e-4:
            return i
        if diff < min_diff:
            min_diff = diff
            min_idx = i
    return min_idx


def print_blue(text, end="\n"):
    print(f"\033[94m{text}\033[0m", end=end)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parents[1].joinpath("config")),
    config_name="sim_config/sim_sim/gcs_planner",  # specify the full path to your config
)
def main(cfg: OmegaConf):
    collect_data = cfg.get("collect_data", False)
    sim_sim_gcs_planner = SimSimGcsPlanner(cfg, collect_data=collect_data)
    sim_sim_gcs_planner.run()


if __name__ == "__main__":
    """
    Configure sim config through hydra yaml file
    Ex: python scripts/planar_pushing/run_sim_sim_gcs_planner.py --config-dir <dir> --config-name <file>
    """
    main()
