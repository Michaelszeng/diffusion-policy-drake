import os
import sys
from contextlib import contextmanager
from enum import Enum

import hydra
import numpy as np

from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.simulation.systems.success_checker import (
    check_success_convex_hull,
    check_success_tolerance,
)


class Result(Enum):
    NONE = "none"
    SLIDER_FELL_OFF_TABLE = "slider fell"
    TIMEOUT = "timeout"
    MISSED_GOAL = "missed goal"
    ELBOW_DOWN = "elbow down"
    SUCCESS = "success"


def print_blue(text, end="\n"):
    print(f"\033[94m{text}\033[0m", end=end)


class _Tee:
    """Mirrors writes to both an original stream and a log file."""

    def __init__(self, original_stream, log_file):
        self._original = original_stream
        self._log = log_file

    def write(self, data):
        self._original.write(data)
        self._log.write(data)

    def flush(self):
        self._original.flush()
        self._log.flush()

    def isatty(self):
        return self._original.isatty()

    def fileno(self):
        return self._original.fileno()

    def __getattr__(self, name):
        return getattr(self._original, name)


@contextmanager
def tee_to_log(output_dir: str, log_filename: str):
    """Context manager that mirrors stdout and stderr to a log file in output_dir."""
    log_path = os.path.join(output_dir, log_filename)
    with open(log_path, "a") as log_file:
        original_stdout, original_stderr = sys.stdout, sys.stderr
        sys.stdout = _Tee(original_stdout, log_file)
        sys.stderr = _Tee(original_stderr, log_file)
        try:
            yield
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def create_hydra_output_dir():
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    if not os.path.exists(f"{output_dir}/analysis"):
        os.makedirs(f"{output_dir}/analysis")
    return output_dir


def append_to_summary_file(output_dir, trial_idx, result, trial_time, initial_conditions, final_error):
    with open(os.path.join(output_dir, "summary.txt"), "a") as f:
        f.write(f"Trial {trial_idx + 1}\n")
        f.write("--------------------\n")
        f.write(f"Result: {result.value}\n")
        f.write(f"Trial time: {trial_time:.2f}\n")
        f.write(f"Initial slider pose: {initial_conditions}\n")
        f.write(f"Final pusher error: {final_error['pusher_error']}\n")
        f.write(f"Final slider error: {final_error['slider_error']}\n")
        f.write("\n")


def generate_analysis_summary(output_dir, summary, multi_run_config, cfg):
    if len(summary["successful_trials"]) == 0:
        average_successful_trans_error = "N/A"
        average_successful_rot_error = "N/A"
    else:
        successful_translation_errors = []
        successful_rotation_errors = []
        for trial_idx in summary["successful_trials"]:
            successful_translation_errors.append(np.linalg.norm(summary["final_error"][trial_idx]["slider_error"][:2]))
            successful_rotation_errors.append(np.abs(summary["final_error"][trial_idx]["slider_error"][2]))

        average_succesful_trans_error = np.mean(successful_translation_errors)
        average_succesful_rot_error = np.mean(successful_rotation_errors)
        average_successful_trans_error = f"{100 * average_succesful_trans_error:.2f}cm"
        average_successful_rot_error = f"{np.rad2deg(average_succesful_rot_error):.2f}°"

    summary_path = os.path.join(output_dir, "summary.pkl")
    import pickle

    with open(summary_path, "wb") as f:
        pickle.dump(summary, f)

    # Read the current content
    existing_content = ""
    if os.path.exists(os.path.join(output_dir, "summary.txt")):
        with open(os.path.join(output_dir, "summary.txt"), "r") as f:
            existing_content = f.read()

    # Write the new content
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        num_runs = len(summary["trial_times"])
        f.write("Evaluation Summary\n")
        f.write("====================================\n")
        f.write("Units: seconds, meters, radians\n\n")
        f.write(f"Total trials: {num_runs}\n")
        f.write(f"Total successful trials: {len(summary['successful_trials'])}\n")
        f.write(f"Success rate: {len(summary['successful_trials']) / num_runs:.6f}\n")
        f.write(f"Average successful translation error: {average_successful_trans_error}\n")
        f.write(f"Average successful rotation error: {average_successful_rot_error}\n")
        f.write(f"Total time (sim): {summary['total_eval_sim_time']:.2f}\n")
        f.write(f"Total time (wall): {summary['total_eval_wall_time']:.2f}\n\n")

        success_criteria = multi_run_config.success_criteria
        f.write(f"Success criteria: {success_criteria}\n")
        if success_criteria == "tolerance":
            f.write(f"Translation tolerance: {multi_run_config.trans_tol}\n")
            f.write(f"Rotation tolerance: {np.deg2rad(multi_run_config.rot_tol):.6f}\n")
            f.write(f"Evaluate final slider rotation: {multi_run_config.evaluate_final_slider_rotation}\n")
            f.write(f"Evaluate final pusher position: {multi_run_config.evaluate_final_pusher_position}\n")
        f.write(f"Max attempt duration: {multi_run_config.max_attempt_duration}\n\n")
        f.write(f"Workspace width: {cfg.multi_run_config.workspace_width}\n")
        f.write(f"Workspace height: {cfg.multi_run_config.workspace_height}\n")
        f.write("====================================\n\n")

        # Append the existing content
        f.write(existing_content)


class SimEvaluator:
    def __init__(self, environment, sim_config, multi_run_config):
        self.env = environment
        self.sim_config = sim_config
        self.multi_run_config = multi_run_config

        # Cache lookups
        self.plant = environment._plant
        self.mbp_context = environment.mbp_context
        self.slider_model_instance = environment._slider_model_instance
        self.robot_model_instance = environment._robot_model_instance
        self.pusher_body = self.plant.GetBodyByName("pusher")

        self.pusher_start_pose = sim_config.pusher_start_pose
        self.slider_goal_pose = sim_config.slider_goal_pose

        # Success criteria
        self.success_criteria = multi_run_config.success_criteria

    def get_pusher_pose(self):
        pusher_position = self.plant.EvalBodyPoseInWorld(self.mbp_context, self.pusher_body).translation()
        return PlanarPose(pusher_position[0], pusher_position[1], 0.0)

    def get_slider_pose(self):
        slider_pose = self.plant.GetPositions(self.mbp_context, self.slider_model_instance)
        return PlanarPose.from_generalized_coords(slider_pose)

    def get_robot_joint_angles(self):
        return self.plant.GetPositions(self.mbp_context, self.robot_model_instance)

    def check_success(self):
        if self.success_criteria == "tolerance":
            return self._check_success_tolerance(self.multi_run_config.trans_tol, self.multi_run_config.rot_tol)
        elif self.success_criteria == "convex_hull":
            return self._check_success_convex_hull()
        else:
            raise ValueError(f"Invalid success criteria: {self.success_criteria}")

    def _check_success_tolerance(self, trans_tol, rot_tol):
        return check_success_tolerance(
            self.get_slider_pose(),
            self.slider_goal_pose,
            self.get_pusher_pose(),
            self.pusher_start_pose,
            trans_tol,
            rot_tol,
            self.multi_run_config.evaluate_final_slider_rotation,
            self.multi_run_config.evaluate_final_pusher_position,
            pusher_pos_tol=self.multi_run_config.pusher_pos_tol,
        )

    def _check_success_convex_hull(self):
        return check_success_convex_hull(
            slider_pose=self.get_slider_pose(),
            pusher_pose=self.get_pusher_pose(),
            dataset_path=self.multi_run_config.dataset_path,
            pusher_start_pose=self.pusher_start_pose,
            slider_goal_pose=self.slider_goal_pose,
            convex_hull_scale=self.multi_run_config.convex_hull_scale,
        )

    def check_close_to_goal(self):
        return self._check_success_tolerance(2 * self.multi_run_config.trans_tol, 2 * self.multi_run_config.rot_tol)

    def get_trial_duration(self, t, last_reset_time):
        # Handle case where diffusion_policy_config might not exist or have delay
        delay = 0.0
        if hasattr(self.sim_config, "diffusion_policy_config") and self.sim_config.diffusion_policy_config is not None:
            delay = self.sim_config.diffusion_policy_config.delay
        elif hasattr(self.sim_config, "delay_before_execution"):
            delay = self.sim_config.delay_before_execution

        return t - last_reset_time - delay

    def check_failure(self, t, last_reset_time):
        # Check timeout
        duration = self.get_trial_duration(t, last_reset_time)
        if duration > self.multi_run_config.max_attempt_duration:
            if self.check_close_to_goal():
                return True, Result.MISSED_GOAL
            else:
                return True, Result.TIMEOUT

        # Check if slider is on table
        slider_pose = self.plant.GetPositions(self.mbp_context, self.slider_model_instance)
        if slider_pose[-1] < 0.0:  # z value
            return True, Result.SLIDER_FELL_OFF_TABLE

        q = self.get_robot_joint_angles()
        if len(q) == 7:
            ELBOW_INDEX = 3
            ELBOW_THRESHOLD = np.deg2rad(5)
            elbow_angle = q[ELBOW_INDEX]
            if elbow_angle > ELBOW_THRESHOLD:
                return True, Result.ELBOW_DOWN

        # No immediate failures
        return False, Result.NONE

    def get_final_error(self):
        pusher_pose = self.get_pusher_pose()
        pusher_goal_pose = self.pusher_start_pose
        pusher_error = pusher_goal_pose.vector() - pusher_pose.vector()

        slider_pose = self.get_slider_pose()
        slider_error = self.slider_goal_pose.vector() - slider_pose.vector()

        return {"pusher_error": pusher_error[:2], "slider_error": slider_error}


TRIALS_PER_RECORDING_FILE = 20


class MeshcatRecordingManager:
    """
    Manages meshcat recording lifecycle: starting, splitting into files of
    at most TRIALS_PER_RECORDING_FILE trials each, and saving.

    Accepts num_trials_to_record as an int or the string "all".
    """

    def __init__(self, meshcat, environment, num_trials_to_record, total_runs, output_dir, file_prefix="recording"):
        self._meshcat = meshcat
        self._environment = environment
        self._output_dir = output_dir
        self._file_prefix = file_prefix

        if isinstance(num_trials_to_record, str) and num_trials_to_record.lower() == "all":
            self._target = total_runs
        else:
            self._target = int(num_trials_to_record)

        self._trials_in_chunk = 0
        self._total_recorded = 0
        self._chunk_index = 0
        self._active = False

    @property
    def active(self):
        return self._active

    def start(self):
        """Start recording if there are trials to record."""
        if self._target > 0:
            self._meshcat.StartRecording(frames_per_second=10)
            self._active = True
            self._trials_in_chunk = 0

    def on_trial_complete(self):
        """Call after each completed trial. Saves and rotates files as needed."""
        if not self._active:
            return

        self._trials_in_chunk += 1
        self._total_recorded += 1

        if self._trials_in_chunk >= TRIALS_PER_RECORDING_FILE or self._total_recorded >= self._target:
            self._save_chunk()

    def finalize(self):
        """Save any remaining recorded trials."""
        if self._active:
            # If we're finalizing but haven't completed a trial yet (e.g. interrupted),
            # we should still save the partial recording.
            if self._trials_in_chunk == 0:
                self._trials_in_chunk = 1

            filename = self._chunk_filename()
            self._environment.save_recording(filename, self._output_dir)
        self._active = False

    def _save_chunk(self):
        filename = self._chunk_filename()
        self._environment.save_recording(filename, self._output_dir)
        self._active = False
        self._chunk_index += 1
        self._trials_in_chunk = 0

        if self._total_recorded < self._target:
            self._meshcat.StartRecording(frames_per_second=10)
            self._active = True

    def _chunk_filename(self):
        if self._target <= TRIALS_PER_RECORDING_FILE:
            return f"{self._file_prefix}.html"
        start = self._chunk_index * TRIALS_PER_RECORDING_FILE
        end = start + self._trials_in_chunk - 1
        return f"{self._file_prefix}_{start:03d}-{end:03d}.html"
