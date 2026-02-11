"""
Alternative source of desired positions for the robot, using GCS Planner
(https://arxiv.org/pdf/2402.10312)
"""

import numpy as np
from pydrake.all import Diagram, DiagramBuilder, Meshcat

from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.simulation.controllers.gcs_planner_controller import GcsPlannerController
from planning_through_contact.simulation.planar_pushing_sim_config import PlanarPushingSimConfig


class GcsPlannerSource(Diagram):
    """
    Just export Inputs and Outputs for the GCS Planner Controller for unified interface.
    """

    def __init__(self, sim_config: PlanarPushingSimConfig, meshcat: Meshcat):
        super().__init__()

        builder = DiagramBuilder()

        ## Add Leaf systems

        # GCS Planner Controller
        self._gcs_planner = builder.AddNamedSystem("GcsPlanner", GcsPlannerController(sim_config, meshcat))

        ## Internal connections
        ## Export inputs and outputs (external)

        builder.ExportInput(
            self._gcs_planner.GetInputPort("pusher_pose_measured"),
            "pusher_pose_measured",
        )

        builder.ExportInput(
            self._gcs_planner.GetInputPort("slider_pose_measured"),
            "slider_pose_measured",
        )

        builder.ExportInput(
            self._gcs_planner.GetInputPort("pusher_velocity_measured"),
            "pusher_velocity_measured",
        )

        builder.ExportInput(
            self._gcs_planner.GetInputPort("run_flag"),
            "run_flag",
        )

        builder.ExportOutput(self._gcs_planner.get_output_port(), "planar_position_command")

        builder.BuildInto(self)

    def reset(self, pusher_position: np.ndarray = None, new_slider_start_pose: PlanarPose = None):
        self._gcs_planner.reset(pusher_position, new_slider_start_pose)
