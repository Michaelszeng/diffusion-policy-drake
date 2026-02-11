import importlib
import os
import pathlib
import pickle
import random
import shutil
import traceback
from typing import Optional

import hydra
import numpy as np
from omegaconf import OmegaConf
from pydrake.all import StartMeshcat

from planning_through_contact.geometry.collision_checker import CollisionChecker
from planning_through_contact.simulation.controllers.gcs_planner_source import (
    GcsPlannerSource,
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


class SimSimGcsPlanner:
    def __init__(self, cfg: OmegaConf):
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

        self.traj_start_time = 0.0
        self.num_saved_trajectories = 0
        self.num_completed_trials = 0

        self.reset_environment(0)

    def simulate_environment(
        self,
        end_time: float = float("inf"),
        recording_file: Optional[str] = None,
    ):
        # Loop variables
        time_step = self.sim_config.time_step * 10
        t = time_step
        meshcat = self.environment._meshcat
        recording_stopped = False

        validated_image_writer = False

        # Start meshcat recording
        if self.multi_run_config.num_trials_to_record > 0:
            meshcat.StartRecording(frames_per_second=10)

        try:
            # Simulate
            self.environment.visualize_desired_slider_pose()
            self.environment.visualize_desired_pusher_pose()
            while t < end_time:
                self.environment._simulator.AdvanceTo(t)

                # # Validate image writer directory
                # if t > 0.2 and not validated_image_writer:
                #     self.validate_image_writer_dir()
                #     validated_image_writer = True

                # Check if we should stop recording after num_trials_to_record
                if (
                    self.num_completed_trials >= self.multi_run_config.num_trials_to_record
                    and self.multi_run_config.num_trials_to_record > 0
                    and not recording_stopped
                ):
                    meshcat.StopRecording()
                    recording_stopped = True
                    if recording_file:
                        self.environment.save_recording(recording_file, os.path.dirname(recording_file))
                    else:
                        # Default recording file name
                        output_dir = self.cfg.get("output_dir", ".")
                        self.environment.save_recording("gcs_planner.html", output_dir)

                # Loop updates
                t += time_step
                t = round(t / time_step) * time_step

        except Exception as e:
            print(f"Error during simulation: {e}")
            traceback.print_exc()
            raise
        finally:
            # Ensure recording is saved even if there's an error
            if self.multi_run_config.num_trials_to_record > 0 and not recording_stopped:
                try:
                    meshcat.StopRecording()
                    if recording_file:
                        self.environment.save_recording(recording_file, os.path.dirname(recording_file))
                    else:
                        # Default recording file name
                        output_dir = self.cfg.get("meshcat_recording_output_dir", ".")
                        self.environment.save_recording("gcs_planner.html", output_dir)
                        print(f"Saved recording to {output_dir}/gcs_planner.html")
                except Exception as save_error:
                    print(f"Error saving recording: {save_error}")
                    traceback.print_exc()

            # # Delete temporary image writer directory
            # if os.path.exists("trajectories_rendered/temp"):
            #     shutil.rmtree("trajectories_rendered/temp")

    def reset_environment(self, trial_idx: int):
        """
        Reset environment with new initial slider pose.
        Use seed sequence to ensure deterministic sequence of initial slider poses is produced every run.
        """
        print("=" * 100 + f"\nResetting environment for trial {trial_idx}.")
        slider = self.sim_config.slider
        ss = np.random.SeedSequence([self.multi_run_config.seed, trial_idx])
        trial_rng = np.random.default_rng(ss)
        slider_pose = get_slider_initial_pose_within_workspace(
            self.workspace, slider, self.pusher_start_pose, self.slider_goal_pose, self.collision_checker, rng=trial_rng
        )

        self.environment.reset(
            self.sim_config.default_joint_positions,
            slider_pose,
            self.pusher_start_pose,
        )

    def save_trajectory(self):
        traj_dir = self.create_trajectory_dir()

        # Move images from temp to traj_dir
        initial_image_id = int(round(self.traj_start_time, 2) * 1000)
        for camera in os.listdir("trajectories_rendered/temp"):
            camera_dir = f"trajectories_rendered/temp/{camera}"
            for file in os.listdir(camera_dir):
                image_id = int(file.split(".")[0])
                if image_id >= initial_image_id:
                    new_image_name = f"{image_id - initial_image_id}.png"
                    shutil.move(f"{camera_dir}/{file}", f"{traj_dir}/{camera}/{new_image_name}")

        # Create combined_logs.pkl file
        pusher_log = self.environment.get_pusher_pose_log()
        pusher_desired = self.get_planar_pushing_log(pusher_log, self.traj_start_time)
        slider_log = self.environment.get_slider_pose_log()
        slider_desired = self.get_planar_pushing_log(slider_log, self.traj_start_time)

        combined_logs = CombinedPlanarPushingLogs(
            pusher_desired=pusher_desired,
            slider_desired=slider_desired,
            pusher_actual=None,
            slider_actual=None,
        )

        # Save combined_logs.pkl
        with open(f"{traj_dir}/combined_logs.pkl", "wb") as f:
            pickle.dump(combined_logs, f)

        self.clear_image_writer_dir()
        self.reset_environment(self.num_saved_trajectories)
        self.num_saved_trajectories += 1
        self.num_completed_trials += 1

    def delete_trajectory(self):
        self.clear_image_writer_dir()
        self.reset_environment(self.num_saved_trajectories)
        self.num_completed_trials += 1

    def create_trajectory_dir(self):
        rendered_plans_dir = self.cfg.data_collection_config.rendered_plans_dir
        if not os.path.exists(rendered_plans_dir):
            os.makedirs(rendered_plans_dir)

        # Find the next available trajectory index
        traj_idx = 0
        for path in os.listdir(rendered_plans_dir):
            if os.path.isdir(os.path.join(rendered_plans_dir, path)):
                traj_idx += 1

        # Setup the current directory
        os.makedirs(f"{rendered_plans_dir}/{traj_idx}")
        for camera in os.listdir("trajectories_rendered/temp"):
            os.makedirs(f"{rendered_plans_dir}/{traj_idx}/{camera}")
        open(f"{rendered_plans_dir}/{traj_idx}/log.txt", "w").close()
        return f"{rendered_plans_dir}/{traj_idx}"

    def clear_image_writer_dir(self):
        # remove all files in trajectories_temp/{camera}
        for camera in os.listdir("trajectories_rendered/temp"):
            camera_dir = f"trajectories_rendered/temp/{camera}"
            for file in os.listdir(camera_dir):
                os.remove(f"{camera_dir}/{file}")

    def validate_image_writer_dir(self):
        # Asserts that image writers are aligned to context time 0.0
        valid = True
        for camera in os.listdir("trajectories_rendered/temp"):
            if not os.path.exists(f"trajectories_rendered/temp/{camera}/0.png"):
                valid = False
                break

        if not valid:
            print_blue("Exiting: image writer directory not aligned to context time 0.0.")
            print_blue("Please restart the script.")
            exit(1)

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


def print_blue(text, end="\n"):
    print(f"\033[94m{text}\033[0m", end=end)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parents[1].joinpath("config")),
    config_name="sim_config/sim_sim/gcs_planner",  # specify the full path to your config
)
def main(cfg: OmegaConf):
    sim_sim_gcs_planner = SimSimGcsPlanner(cfg)
    sim_sim_gcs_planner.simulate_environment()


if __name__ == "__main__":
    """
    Configure sim config through hydra yaml file
    Ex: python scripts/planar_pushing/run_sim_sim_gcs_planner.py --config-dir <dir> --config-name <file>
    """
    main()
