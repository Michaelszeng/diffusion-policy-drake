from enum import Enum

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
        if self._active and self._trials_in_chunk > 0:
            self._save_chunk()
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
