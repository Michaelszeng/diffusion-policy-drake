"""
RLActionSource: A Drake LeafSystem whose planar_position_command output can be set
directly from Python. Used as the desired_position_source in SimulatedRealTableEnvironment
for RL training, replacing DiffusionPolicySource / GcsPlannerSource.
"""

import numpy as np
from pydrake.all import AbstractValue, LeafSystem, RigidTransform

from planning_through_contact.geometry.planar.planar_pose import PlanarPose


class RLActionSource(LeafSystem):
    """
    Drake LeafSystem that serves as a pass-through action source for RL training.

    The RL agent sets the desired planar position via set_action(), and this system
    outputs it as planar_position_command to the robot station.

    Input ports (declared so SimulatedRealTableEnvironment can connect them):
      - pusher_pose_measured: RigidTransform (connected but not used internally)
      - slider_pose_measured: RigidTransform (connected but not used internally)

    Output port:
      - planar_position_command: [x, y] absolute position in meters
    """

    def __init__(self, initial_pusher_pose: PlanarPose):
        super().__init__()
        self._current_command = np.array([initial_pusher_pose.x, initial_pusher_pose.y], dtype=np.float64)

        # Input ports (required for SimulatedRealTableEnvironment to connect)
        self.DeclareAbstractInputPort("pusher_pose_measured", AbstractValue.Make(RigidTransform()))
        self.DeclareAbstractInputPort("slider_pose_measured", AbstractValue.Make(RigidTransform()))

        # Output port
        self.DeclareVectorOutputPort("planar_position_command", 2, self._calc_output)

    def _calc_output(self, context, output):
        output.set_value(self._current_command)

    def set_action(self, command: np.ndarray) -> None:
        """Set the planar position command. Called from outside the Drake sim loop."""
        self._current_command = np.asarray(command, dtype=np.float64).copy()

    def get_current_command(self) -> np.ndarray:
        return self._current_command.copy()

    def reset(self, pusher_position: np.ndarray = None) -> None:
        """Called by SimulatedRealTableEnvironment.reset()."""
        if pusher_position is not None:
            self._current_command = np.asarray(pusher_position, dtype=np.float64).copy()
