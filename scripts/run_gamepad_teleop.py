"""
Gamepad teleoperation data collection script.

Usage:
    python scripts/run_gamepad_teleop.py --config-name sim_sim/gamepad_teleop_carbon

Controls:
    A button:  Start recording / save trajectory
    B button:  Discard recording / reset environment
    X button:  Quit
    RT:        Half-speed movement
    LT:        Triple-speed movement

Set data_collection_config.convert_to_zarr=true in the config (or as a CLI override)
to automatically convert the collected pkl+image data to zarr after the session ends.
"""

import importlib
import math
import os
import pathlib
import pickle
import shutil
import time
from enum import Enum

import cv2
import hydra
import numpy as np
import zarr
from omegaconf import OmegaConf
from PIL import Image
from pydrake.all import StartMeshcat
from tqdm import tqdm

from planning_through_contact.geometry.collision_checker import CollisionChecker
from planning_through_contact.simulation.controllers.gamepad_controller_source import (
    GamepadControllerSource,
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
from planning_through_contact.simulation.sim_utils import (
    get_slider_initial_pose_within_workspace,
)
from planning_through_contact.utils import file_lock
from planning_through_contact.visualize.analysis import (
    CombinedPlanarPushingLogs,
    PlanarPushingLog,
)


class FSMState(Enum):
    REGULAR = "regular"
    DATA_COLLECTION = "data collection"
    TERMINATE = "terminate"


def _print_blue(text, end="\n"):
    print(f"\033[94m{text}\033[0m", end=end)


class GamepadDataCollection:
    def __init__(self, cfg: OmegaConf):
        np.random.seed(int(1e6 * time.time() % 1e6))

        station_meshcat = StartMeshcat()

        self.cfg = cfg

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        table_urdf_lock = os.path.join(
            project_root, "planning_through_contact/simulation/models/small_table_hydroelastic.urdf"
        )
        slider_sdf_lock = os.path.join(project_root, "planning_through_contact/simulation/models/arbitrary_shape.sdf")
        with file_lock(table_urdf_lock):
            with file_lock(slider_sdf_lock):
                self.sim_config = PlanarPushingSimConfig.from_yaml(cfg)
                module_name, class_name = cfg.robot_station._target_.rsplit(".", 1)
                robot_system_class = getattr(importlib.import_module(module_name), class_name)
                position_controller: RobotSystemBase = robot_system_class(
                    sim_config=self.sim_config, meshcat=station_meshcat
                )

        self.pusher_start_pose = self.sim_config.pusher_start_pose
        self.slider_goal_pose = self.sim_config.slider_goal_pose
        _print_blue(f"Initial pusher pose: {self.pusher_start_pose}")
        _print_blue(f"Target slider pose:  {self.slider_goal_pose}")

        self.collision_checker = CollisionChecker(cfg.arbitrary_shape_pickle_path, cfg.pusher_radius, station_meshcat)
        self.workspace = self.sim_config.multi_run_config.workspace

        position_source = GamepadControllerSource(
            station_meshcat,
            translation_scale=cfg.gamepad.translation_scale,
            deadzone=cfg.gamepad.deadzone,
            gamepad_orientation=np.array(cfg.gamepad.gamepad_orientation),
        )

        image_writer_dir = "trajectories_rendered/temp"
        if os.path.exists(image_writer_dir):
            shutil.rmtree(image_writer_dir)

        self.environment = SimulatedRealTableEnvironment(
            desired_position_source=position_source,
            robot_system=position_controller,
            sim_config=self.sim_config,
            station_meshcat=station_meshcat,
            arbitrary_shape_pickle_path=cfg.arbitrary_shape_pickle_path,
        )
        self.environment.export_diagram("gamepad_teleop_environment.svg")

        self.fsm_state = FSMState.REGULAR
        self.traj_start_time = 0.0
        self.num_saved_trajectories = 0

    def simulate_environment(self, end_time: float = float("inf")):
        prev_button_values = self.environment.get_button_values()
        base_translation_scale = self.environment._desired_position_source.get_translation_scale()
        time_step = self.sim_config.time_step * 10
        t = time_step
        validated_image_writer = False

        self.environment.visualize_desired_slider_pose()
        self.environment.visualize_desired_pusher_pose()

        while t < end_time and self.fsm_state != FSMState.TERMINATE:
            self.environment._simulator.AdvanceTo(t)

            if t > 0.2 and not validated_image_writer:
                self._validate_image_writer_dir()
                validated_image_writer = True

            button_values = self.environment.get_button_values()
            pressed_buttons = self._get_pressed_buttons(prev_button_values, button_values)

            self.fsm_state, self.traj_start_time = self._fsm_logic(
                self.fsm_state, pressed_buttons, t, self.traj_start_time
            )

            if button_values["RT"]:
                self.environment._desired_position_source.set_translation_scale(0.5 * base_translation_scale)
            elif button_values["LT"]:
                self.environment._desired_position_source.set_translation_scale(3.0 * base_translation_scale)
            else:
                self.environment._desired_position_source.set_translation_scale(base_translation_scale)

            t += time_step
            t = round(t / time_step) * time_step
            prev_button_values = button_values

        if os.path.exists("trajectories_rendered/temp"):
            shutil.rmtree("trajectories_rendered/temp")

        if self.cfg.data_collection_config.convert_to_zarr:
            convert_to_zarr(self.sim_config, self.cfg.data_collection_config)

    def _fsm_logic(self, fsm_state, pressed_buttons, curr_time, traj_start_time):
        pressed_A = pressed_buttons["A"]
        pressed_B = pressed_buttons["B"]
        pressed_X = pressed_buttons["X"]

        if pressed_X:
            _print_blue(f"Terminating. Collected {self.num_saved_trajectories} trajectories.")
            return FSMState.TERMINATE, traj_start_time

        if fsm_state == FSMState.REGULAR:
            if pressed_A:
                _print_blue(f"Recording started at t={curr_time:.2f}s")
                return FSMState.DATA_COLLECTION, curr_time
            if pressed_B:
                self._reset_environment()
                _print_blue("Environment reset.")
                return FSMState.REGULAR, traj_start_time

        elif fsm_state == FSMState.DATA_COLLECTION:
            if pressed_A:
                self._save_trajectory()
                _print_blue(f"Trajectory saved ({self.num_saved_trajectories} total). Entering regular mode.")
                return FSMState.REGULAR, traj_start_time
            elif pressed_B:
                self._delete_trajectory()
                _print_blue("Recording discarded. Entering regular mode.")
                return FSMState.REGULAR, traj_start_time

        return fsm_state, traj_start_time

    def _reset_environment(self):
        slider_pose = get_slider_initial_pose_within_workspace(
            self.workspace,
            self.sim_config.slider,
            self.pusher_start_pose,
            self.slider_goal_pose,
            self.collision_checker,
        )
        self.environment.reset(
            self.sim_config.default_joint_positions,
            slider_pose,
            self.pusher_start_pose,
        )

    def _save_trajectory(self):
        traj_dir = self._create_trajectory_dir()

        # Move images from temp dir to traj_dir, re-indexed relative to traj_start_time
        initial_image_id = int(round(self.traj_start_time, 2) * 1000)
        for camera in os.listdir("trajectories_rendered/temp"):
            camera_dir = f"trajectories_rendered/temp/{camera}"
            for file in os.listdir(camera_dir):
                image_id = int(file.split(".")[0])
                if image_id >= initial_image_id:
                    new_name = f"{image_id - initial_image_id}.png"
                    shutil.move(
                        f"{camera_dir}/{file}",
                        f"{traj_dir}/{camera}/{new_name}",
                    )

        pusher_log = self.environment.get_pusher_pose_log()
        pusher_desired = self._extract_planar_pushing_log(pusher_log, self.traj_start_time)
        slider_log = self.environment.get_slider_pose_log()
        slider_desired = self._extract_planar_pushing_log(slider_log, self.traj_start_time)

        combined_logs = CombinedPlanarPushingLogs(
            pusher_actual=None,
            slider_actual=None,
            pusher_desired=pusher_desired,
            slider_desired=slider_desired,
        )

        with open(f"{traj_dir}/combined_logs.pkl", "wb") as f:
            pickle.dump(combined_logs, f)

        self._save_trajectory_plot(combined_logs, traj_dir)
        self._clear_image_writer_dir()
        self._reset_environment()
        self.num_saved_trajectories += 1

    def _delete_trajectory(self):
        self._clear_image_writer_dir()
        self._reset_environment()

    def _create_trajectory_dir(self) -> str:
        rendered_plans_dir = self.cfg.data_collection_config.rendered_plans_dir
        os.makedirs(rendered_plans_dir, exist_ok=True)

        traj_idx = sum(1 for p in os.listdir(rendered_plans_dir) if os.path.isdir(os.path.join(rendered_plans_dir, p)))
        traj_dir = f"{rendered_plans_dir}/{traj_idx}"
        os.makedirs(traj_dir)
        for camera in os.listdir("trajectories_rendered/temp"):
            os.makedirs(f"{traj_dir}/{camera}")
        open(f"{traj_dir}/log.txt", "w").close()
        return traj_dir

    def _clear_image_writer_dir(self):
        for camera in os.listdir("trajectories_rendered/temp"):
            camera_dir = f"trajectories_rendered/temp/{camera}"
            for file in os.listdir(camera_dir):
                os.remove(f"{camera_dir}/{file}")

    def _validate_image_writer_dir(self):
        valid = all(
            os.path.exists(f"trajectories_rendered/temp/{camera}/0.png")
            for camera in os.listdir("trajectories_rendered/temp")
        )
        if not valid:
            _print_blue("ERROR: image writers not aligned to t=0. Please restart.")
            exit(1)

    def _extract_planar_pushing_log(self, vector_log, traj_start_time) -> PlanarPushingLog:
        sample_times = vector_log.sample_times()
        start_idx = 0
        while start_idx < len(sample_times) - 1 and sample_times[start_idx] < traj_start_time:
            start_idx += 1

        t = sample_times[start_idx:] - sample_times[start_idx]
        nan_array = np.full(len(t), float("nan"))
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

    def _save_trajectory_plot(self, combined_logs: CombinedPlanarPushingLogs, traj_dir: str):
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        for ax, log, title in [
            (axs[0], combined_logs.pusher_desired, "Pusher"),
            (axs[1], combined_logs.slider_desired, "Slider"),
        ]:
            ax.plot(log.t, log.x, label="x")
            ax.plot(log.t, log.y, label="y")
            ax.plot(log.t, log.theta, label="theta")
            ax.set_title(title)
            ax.legend()

        plt.tight_layout()
        plt.savefig(f"{traj_dir}/plot.png")
        plt.close()

    def _get_pressed_buttons(self, prev, curr) -> dict:
        return {btn: (curr[btn] and not prev[btn]) for btn in curr}


# ──────────────────────────────────────────────────────────────────────────────
# Zarr conversion (copied from planning-through-contact-adam/scripts/planar_pushing/run_data_generation.py)
# ──────────────────────────────────────────────────────────────────────────────


def convert_to_zarr(sim_config, data_collection_config, debug: bool = False):
    """
    Converts the rendered plans to zarr format.

    Assumes the rendered plans directory has the following structure:

    rendered_plans_dir
    ├── 0
    ├──├── combined_logs.pkl
    ├──├── log.txt
    ├──├── overhead_camera/
    ├──├── wrist_camera/
    ├── 1
    ...
    """

    print("\nConverting data to zarr...")

    rendered_plans_dir = pathlib.Path(data_collection_config.rendered_plans_dir)
    zarr_path = data_collection_config.zarr_path

    traj_dir_list = []
    for plan in os.listdir(rendered_plans_dir):
        traj_dir = rendered_plans_dir.joinpath(plan)
        if not os.path.isdir(traj_dir):
            continue
        traj_dir_list.append(traj_dir)

    concatenated_states = []
    concatenated_slider_states = []
    concatenated_actions = []
    concatenated_targets = []
    episode_ends = []
    current_end = 0

    freq = data_collection_config.policy_freq
    dt = 1 / freq

    num_ik_fails = 0
    num_angular_speed_violations = 0

    for traj_dir in tqdm(traj_dir_list):
        traj_log_path = traj_dir.joinpath("combined_logs.pkl")
        log_path = traj_dir.joinpath("log.txt")

        # If too many IK fails, skip this rollout
        if _is_ik_fail(log_path):
            num_ik_fails += 1
            continue

        # load pickle file and timing variables
        combined_logs = pickle.load(open(traj_log_path, "rb"))
        pusher_desired = combined_logs.pusher_desired
        slider_desired = combined_logs.slider_desired

        if _has_high_angular_speed(
            slider_desired,
            data_collection_config.angular_speed_threshold,
            data_collection_config.angular_speed_window_size,
        ):
            num_angular_speed_violations += 1
            continue

        t = pusher_desired.t
        total_time = math.floor(t[-1] * freq) / freq

        # get start time
        start_idx = _get_start_idx(pusher_desired)
        start_time = math.ceil(t[start_idx] * freq) / freq

        # get state, action, images
        current_time = start_time
        idx = start_idx
        state = []
        slider_state = []

        while current_time < total_time:
            # state and action
            idx = _get_closest_index(t, current_time, idx)
            current_state = np.array(
                [
                    pusher_desired.x[idx],
                    pusher_desired.y[idx],
                    pusher_desired.theta[idx],
                ]
            )
            current_slider_state = np.array(
                [
                    slider_desired.x[idx],
                    slider_desired.y[idx],
                    slider_desired.theta[idx],
                ]
            )
            state.append(current_state)
            slider_state.append(current_slider_state)

            # update current time
            current_time = round((current_time + dt) * freq) / freq

        state = np.array(state)  # T x 3
        slider_state = np.array(slider_state)  # T x 3
        action = np.array(state)[:, :2]  # T x 2
        action = np.concatenate([action[1:, :], action[-1:, :]], axis=0)  # shift action

        # get target
        target = np.array([slider_state[-1] for _ in range(len(state))])

        # update concatenated arrays
        concatenated_states.append(state)
        concatenated_slider_states.append(slider_state)
        concatenated_actions.append(action)
        concatenated_targets.append(target)
        episode_ends.append(current_end + len(state))
        current_end += len(state)

    assert num_ik_fails + num_angular_speed_violations + len(episode_ends) == len(traj_dir_list)
    print(f"{num_ik_fails} of {len(traj_dir_list)} rollouts were skipped due to IK fails.")
    print(f"{num_angular_speed_violations} of {len(traj_dir_list)} rollouts were skipped due to high angular speeds.")
    print(f"Total number of converted rollouts: {len(episode_ends)}\n")

    # save to zarr
    root = zarr.open_group(zarr_path, mode="w")
    data_group = root.create_group("data")
    meta_group = root.create_group("meta")

    # Chunk sizes optimized for read
    state_chunk_size = (data_collection_config.state_chunk_length, state.shape[1])
    slider_state_chunk_size = (
        data_collection_config.state_chunk_length,
        state.shape[1],
    )
    action_chunk_size = (data_collection_config.action_chunk_length, action.shape[1])
    target_chunk_size = (data_collection_config.target_chunk_length, target.shape[1])

    # convert to numpy
    concatenated_states = np.concatenate(concatenated_states, axis=0)
    concatenated_slider_states = np.concatenate(concatenated_slider_states, axis=0)
    concatenated_actions = np.concatenate(concatenated_actions, axis=0)
    concatenated_targets = np.concatenate(concatenated_targets, axis=0)
    episode_ends = np.array(episode_ends)
    last_episode_end = episode_ends[-1]

    assert last_episode_end == concatenated_states.shape[0]
    assert concatenated_states.shape[0] == concatenated_slider_states.shape[0]
    assert concatenated_states.shape[0] == concatenated_actions.shape[0]
    assert concatenated_states.shape[0] == concatenated_targets.shape[0]

    data_group.create_dataset("state", data=concatenated_states, chunks=state_chunk_size)
    data_group.create_dataset("slider_state", data=concatenated_slider_states, chunks=slider_state_chunk_size)
    data_group.create_dataset("action", data=concatenated_actions, chunks=action_chunk_size)
    data_group.create_dataset("target", data=concatenated_targets, chunks=target_chunk_size)
    meta_group.create_dataset("episode_ends", data=episode_ends)

    # Delete arrays to save memory
    del concatenated_states
    del concatenated_slider_states
    del concatenated_actions
    del concatenated_targets
    del episode_ends

    # Save images separately and one at a time to save RAM
    camera_names = [camera_config.name for camera_config in sim_config.camera_configs]
    desired_image_shape = np.array([data_collection_config.image_height, data_collection_config.image_width, 3])
    image_chunk_size = [
        data_collection_config.image_chunk_length,
        *desired_image_shape,
    ]

    for camera_name in camera_names:
        print(f"Converting images from {camera_name} to zarr...")
        concatenated_images = zarr.zeros(
            (last_episode_end, *desired_image_shape),
            chunks=image_chunk_size,
            dtype="u1",
        )
        sequence_idx = 0

        for traj_dir in tqdm(traj_dir_list):
            traj_log_path = traj_dir.joinpath("combined_logs.pkl")
            log_path = traj_dir.joinpath("log.txt")

            # If too many IK fails, skip this rollout
            if _is_ik_fail(log_path):
                continue

            # load pickle file and timing variables
            combined_logs = pickle.load(open(traj_log_path, "rb"))
            pusher_desired = combined_logs.pusher_desired
            total_time = pusher_desired.t[-1]
            total_time = math.floor(total_time * freq) / freq

            if _has_high_angular_speed(
                combined_logs.slider_desired,
                data_collection_config.angular_speed_threshold,
                data_collection_config.angular_speed_window_size,
            ):
                continue

            # get start time
            start_idx = _get_start_idx(pusher_desired)
            start_time = math.ceil(t[start_idx] * freq) / freq
            del pusher_desired

            # Get timestamp of initial image
            image_dir = traj_dir.joinpath(camera_name)
            timestamps = [int(f.split(".")[0]) for f in os.listdir(image_dir) if f.endswith(".png")]
            first_image_time = min(timestamps) / 1000
            assert first_image_time >= 0.0

            # get state, action, images
            current_time = start_time
            idx = start_idx

            while current_time < total_time:
                idx = _get_closest_index(t, current_time, idx)

                # Image names are "{time in ms}" rounded to the nearest 100th
                image_name = round(((current_time - first_image_time) * 1000) / 100) * 100 + first_image_time * 1000
                image_path = traj_dir.joinpath(camera_name, f"{int(image_name)}.png")
                img = Image.open(image_path).convert("RGB")
                img = np.asarray(img)
                if not np.allclose(img.shape, desired_image_shape):
                    # Image size for cv2 is (width, height) instead of (height, width)
                    img = cv2.resize(img, (desired_image_shape[1], desired_image_shape[0]))

                concatenated_images[sequence_idx] = img
                sequence_idx += 1

                if debug:
                    from matplotlib import pyplot as plt

                    print(f"\nCurrent time: {current_time}")
                    print(f"Current index: {idx}")
                    print(f"Image path: {image_path}")
                    plt.imshow(img[6:-6, 6:-6, :])
                    plt.show()

                current_time = round((current_time + dt) * freq) / freq
            # End episode time step loop
        # End episode loop

        # Save images to zarr
        assert len(concatenated_images) == last_episode_end
        assert sequence_idx == last_episode_end
        data_group.create_dataset(
            camera_name,
            data=concatenated_images,
            chunks=image_chunk_size,
        )

    # End camera loop


def _get_start_idx(pusher_desired):
    """
    Finds the index of the first "non-stationary" command.
    This is the index of the start of the trajectory.
    """

    length = len(pusher_desired.t)
    first_non_zero_idx = 0
    for i in range(length):
        if pusher_desired.x[i] != 0 or pusher_desired.y[i] != 0 or pusher_desired.theta[i] != 0:
            first_non_zero_idx = i
            break

    initial_state = np.array(
        [
            pusher_desired.x[first_non_zero_idx],
            pusher_desired.y[first_non_zero_idx],
            pusher_desired.theta[first_non_zero_idx],
        ]
    )
    assert not np.allclose(initial_state, np.array([0.0, 0.0, 0.0]))

    for i in range(first_non_zero_idx + 1, length):
        state = np.array([pusher_desired.x[i], pusher_desired.y[i], pusher_desired.theta[i]])
        if not np.allclose(state, initial_state):
            return i

    return None


def _is_ik_fail(log_path, max_failures=5):
    with open(log_path, "r") as f:
        line = f.readline()
        if len(line) != 0:
            ik_fails = int(line.rsplit(" ", 1)[-1])
            if ik_fails > max_failures:
                return True
    return False


def _get_closest_index(arr, t, start_idx=None, end_idx=None):
    """Returns index of arr that is closest to t."""

    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = len(arr)

    min_diff = float("inf")
    min_idx = -1
    eps = 1e-4
    for i in range(start_idx, end_idx):
        diff = abs(arr[i] - t)
        if diff > min_diff:
            return min_idx
        if diff < eps:
            return i
        if diff < min_diff:
            min_diff = diff
            min_idx = i


def _compute_angular_speed(time, orientation):
    dt = np.diff(time)
    dtheta = np.diff(orientation)
    angular_speed = abs(dtheta / dt)

    # Remove sharp angular velocity at beginning
    first_zero_idx = -1
    for i in range(len(angular_speed)):
        if np.allclose(angular_speed[i], 0.0):
            first_zero_idx = i
            break

    return angular_speed[first_zero_idx:]


def _has_high_angular_speed(slider_desired, threshold, window_size):
    if threshold is None:
        return False

    angular_speed = _compute_angular_speed(slider_desired.t, slider_desired.theta)
    angular_speed_cumsum = np.cumsum(angular_speed)
    max_window_avg = -1
    ret = False
    for i in range(len(angular_speed_cumsum) - window_size):
        window_avg = (angular_speed_cumsum[i + window_size] - angular_speed_cumsum[i]) / window_size
        max_window_avg = max(max_window_avg, window_avg)
        if window_avg > threshold:
            return True
    return False


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parents[1].joinpath("config")),
    config_name="sim_config/sim_sim/gamepad_teleop_carbon",
)
def main(cfg: OmegaConf):
    collector = GamepadDataCollection(cfg)
    collector.simulate_environment()


if __name__ == "__main__":
    main()
