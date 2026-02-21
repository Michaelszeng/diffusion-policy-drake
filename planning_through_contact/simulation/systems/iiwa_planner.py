import logging
from copy import copy
from enum import Enum

import numpy as np
import numpy.typing as npt
from pydrake.all import (
    AbstractValue,
    GcsTrajectoryOptimization,
    HPolyhedron,
    InputPortIndex,
    InverseKinematics,
    LeafSystem,
    MultibodyPlant,
    PathParameterizedTrajectory,
    PiecewisePolynomial,
    Point,
    RigidTransform,
    RotationMatrix,
    Solve,
    Toppra,
)

from planning_through_contact.simulation.planar_pushing_sim_config import (
    PlanarPushingSimConfig,
)

logger = logging.getLogger(__name__)


class IiwaPlannerMode(Enum):
    PLAN_GO_PUSH_START = 0
    GO_PUSH_START = 1
    WAIT_PUSH = 2
    PUSHING = 3


class IiwaPlanner(LeafSystem):
    """
    Finite State Machine (FSM) planner that manages the iiwa actions.

    FSM Overview:
    - States (IiwaPlannerMode):
        - PLAN_GO_PUSH_START: Plan a trajectory from current iiwa configuration to the desired push start configuration.
        - GO_PUSH_START: Execute the planned trajectory to reach the push start configuration.
        - WAIT_PUSH: Hold at the push start configuration for a brief delay before handing off to pushing.
        - PUSHING: Handoff state where the planner yields control to DiffIK/policy (no joint command output here).

    - Timing and state bookkeeping (context abstract state):
        - _mode_index: Leafsystem state index for urrent FSM mode (IiwaPlannerMode).
        - _times_index: Leafsystem state index for Dict[str, float] with keys:
            - "initial": initial delay start time threshold before first planning.
            - "go_push_start_initial": time when the GO_PUSH_START trajectory begins execution.
            - "go_push_start_final": absolute time when GO_PUSH_START should be complete.
            - "wait_push_final": absolute time when WAIT_PUSH should be complete.
            - All times are absolute times (relative to start of simulation).

    - Update() logic:
        - PLAN_GO_PUSH_START:
            - When current_time > times["initial"], call PlanGoPushStart(), which:
                - Computes q_goal via get_desired_start_pos() (IK to reach pusher start pose).
                - Builds a trajectory q_traj from current q_start to q_goal (via GCS + Toppra retiming).
                - Sets times["go_push_start_initial"], times["go_push_start_final"], times["wait_push_final"].
                - Transitions mode → GO_PUSH_START.
            - Otherwise, remain in PLAN_GO_PUSH_START and output q0.

        - GO_PUSH_START:
            - If current_time > times["go_push_start_final"], transition → WAIT_PUSH.
            - Else, keep following the time-parameterized q_traj.

        - WAIT_PUSH:
            - Hold the push-start posture (q_goal).
            - If current_time > times["wait_push_final"], transition → PUSHING.

        - PUSHING:
            - Planner no longer produces joint commands (CalcIiwaPosition asserts in this mode).
            - Control is expected to switch to DiffIK/policy (see CalcControlMode output).

        - reset_planner:
            - If this flag is set (i.e. by a timeout or failed pushing attempt), then reinitialize:
                - Clear the flag, set times["initial"] = now, snapshot current q into q0.
                - Immediately re-plan via PlanGoPushStart().

    - Outputs:
        - CalcControlMode:
            - PUSHING → InputPortIndex(2) (DiffIK/policy mode).
            - Else (PLAN_GO_PUSH_START, GO_PUSH_START, WAIT_PUSH) → InputPortIndex(1) (planner/joint command mode).
        - CalcDiffIKReset:
            - PUSHING → False (do not reset DiffIK).
            - Else → True (reset DiffIK while planning/moving/holding).
        - CalcIiwaPosition (joint position command):
            - PLAN_GO_PUSH_START → current q0.
            - GO_PUSH_START → q_traj evaluated at (now - go_push_start_initial).
            - WAIT_PUSH → push_start_pos (q_goal).
            - PUSHING → not used; call is invalid by design.
    """

    def __init__(
        self,
        sim_config: PlanarPushingSimConfig,
        robot_plant: MultibodyPlant,
        initial_delay=2,
        wait_push_delay=2,
    ):
        LeafSystem.__init__(self)
        self._wait_push_delay = wait_push_delay
        self._mode_index = self.DeclareAbstractState(AbstractValue.Make(IiwaPlannerMode.PLAN_GO_PUSH_START))

        # Update this on reset
        self._times_index = self.DeclareAbstractState(AbstractValue.Make({"initial": initial_delay}))

        # For GoPushStart mode:
        num_positions = robot_plant.num_positions()
        self._iiwa_position_measured_index = self.DeclareVectorInputPort(
            "iiwa_position_measured", robot_plant.num_positions()
        ).get_index()
        self.DeclareAbstractOutputPort(
            "control_mode",
            lambda: AbstractValue.Make(InputPortIndex(0)),
            self.CalcControlMode,
        )

        # This output port is not currently being used
        self.DeclareAbstractOutputPort(
            "reset_diff_ik",
            lambda: AbstractValue.Make(False),
            self.CalcDiffIKReset,
        )
        self._q0_index = self.DeclareDiscreteState(num_positions)  # for q0
        self._traj_q_index = self.DeclareAbstractState(AbstractValue.Make(PiecewisePolynomial()))
        self.DeclareVectorOutputPort("iiwa_position_command", num_positions, self.CalcIiwaPosition)
        self.DeclareInitializationDiscreteUpdateEvent(self.Initialize)
        self.DeclarePeriodicUnrestrictedUpdateEvent(0.1, 0.0, self.Update)

        self.reset_planner = False  # Flag to stop pushing and reset the planner

        self._plant = robot_plant  # i.e. for solving IK, GCS, etc.

        self._sim_config = sim_config  # For accessing start pose

        self.vel_limits = 1 * np.ones(7)
        self.accel_limits = 1 * np.ones(7)

    def Update(self, context, state):
        # FSM Logic for planner
        mode = context.get_abstract_state(self._mode_index).get_value()

        current_time = context.get_time()
        times = context.get_abstract_state(self._times_index).get_value()

        if mode == IiwaPlannerMode.PLAN_GO_PUSH_START:
            if context.get_time() >= times["initial"]:
                self.PlanGoPushStart(context, state)
            return
        elif mode == IiwaPlannerMode.GO_PUSH_START:
            if current_time >= times["go_push_start_final"]:
                # We have reached the end of the GoPushStart trajectory.
                state.get_mutable_abstract_state(int(self._mode_index)).set_value(IiwaPlannerMode.WAIT_PUSH)
                logger.debug(f"Switching to WAIT_PUSH mode at time {current_time}.")
                current_pos = self.get_input_port(self._iiwa_position_measured_index).Eval(context)
                # logger.debug(f"Current position: {current_pos}")
            return
        elif mode == IiwaPlannerMode.WAIT_PUSH:
            if current_time >= times["wait_push_final"]:
                # We have reached the end of the GoPushStart trajectory.
                state.get_mutable_abstract_state(int(self._mode_index)).set_value(IiwaPlannerMode.PUSHING)
                logger.debug(f"Switching to PUSHING mode at time {current_time}.")
                current_pos = self.get_input_port(self._iiwa_position_measured_index).Eval(context)
                # logger.debug(f"Current position: {current_pos}")
                global time_pushing_transition
                time_pushing_transition = current_time
            return
        elif self.reset_planner:
            # Resets the planner to the initial state
            # This is somewhat of a hack: ideally, reset is handled from some event and trigger
            self.reset_planner = False
            state.get_mutable_abstract_state(int(self._times_index)).set_value({"initial": current_time})
            context.get_discrete_state(self._q0_index).set_value(
                self.get_input_port(int(self._iiwa_position_measured_index)).Eval(context)
            )
            self.PlanGoPushStart(context, state)
        elif mode == IiwaPlannerMode.PUSHING:
            # Just logging current position
            current_pos = self.get_input_port(self._iiwa_position_measured_index).Eval(context)
            # logger.debug(f"PUSHING: time {context.get_time()} Current position: {current_pos}")

    def PlanGoPushStart(self, context, state):
        logger.debug(f"PlanGoPushStart at time {context.get_time()}.")
        q_start = copy(context.get_discrete_state(self._q0_index).get_value())
        q_goal = self.solve_ik(
            pose=self._sim_config.pusher_start_pose.to_pose(self._sim_config.pusher_z_offset),
            default_joint_positions=self._sim_config.default_joint_positions,
        )

        q_traj = self.create_go_push_start_traj(q_goal, q_start)
        state.get_mutable_abstract_state(int(self._traj_q_index)).set_value(q_traj)
        times = state.get_mutable_abstract_state(int(self._times_index)).get_value()

        times["go_push_start_initial"] = context.get_time()
        times["go_push_start_final"] = q_traj.end_time() + context.get_time()
        times["wait_push_final"] = times["go_push_start_final"] + self._wait_push_delay
        state.get_mutable_abstract_state(int(self._times_index)).set_value(times)
        state.get_mutable_abstract_state(int(self._mode_index)).set_value(IiwaPlannerMode.GO_PUSH_START)
        self.push_start_pos = q_goal

    def CalcControlMode(self, context, output):
        mode = context.get_abstract_state(self._mode_index).get_value()
        if mode == IiwaPlannerMode.PUSHING:
            output.set_value(InputPortIndex(2))  # Pushing (DiffIK)
        else:
            output.set_value(InputPortIndex(1))  # Wait/GoPushStart

    def CalcDiffIKReset(self, context, output):
        mode = context.get_abstract_state(self._mode_index).get_value()
        if mode == IiwaPlannerMode.PUSHING:
            output.set_value(False)  # Pushing (DiffIK)
        else:
            output.set_value(True)  # Wait/GoPushStart

    def CalcIiwaPosition(self, context, output):
        mode = context.get_abstract_state(self._mode_index).get_value()
        if mode == IiwaPlannerMode.PLAN_GO_PUSH_START:
            q_start = copy(context.get_discrete_state(self._q0_index).get_value())
            output.SetFromVector(q_start)
        elif mode == IiwaPlannerMode.GO_PUSH_START:
            traj_q = context.get_mutable_abstract_state(int(self._traj_q_index)).get_value()

            times = context.get_mutable_abstract_state(int(self._times_index)).get_value()

            traj_curr_time = context.get_time() - times["go_push_start_initial"]

            output.SetFromVector(traj_q.value(traj_curr_time))
        elif mode == IiwaPlannerMode.WAIT_PUSH:
            output.SetFromVector(self.push_start_pos)
        elif mode == IiwaPlannerMode.PUSHING:
            assert False, "Planner CalcIiwaPosition should not be called in PUSHING mode."
        else:
            assert False, "Invalid mode."

    def Initialize(self, context, discrete_state):
        discrete_state.set_value(
            int(self._q0_index),
            self.get_input_port(int(self._iiwa_position_measured_index)).Eval(context),
        )

    def reset(self):
        self.reset_planner = True

    def create_go_push_start_traj(self, q_goal, q_start):
        """
        Creates a trajectory for the iiwa to go from its current position to the desired start position.

        q_goal: Desired start position
        q_start: Current position
        """

        def make_traj_toppra(traj, plant, vel_limits, accel_limits, num_grid_points=1000):
            """Helper to retime trajectory using Toppra."""
            toppra = Toppra(
                traj,
                plant,
                np.linspace(traj.start_time(), traj.end_time(), num_grid_points),
            )
            toppra.AddJointVelocityLimit(-vel_limits, vel_limits)
            toppra.AddJointAccelerationLimit(-accel_limits, accel_limits)
            time_traj = toppra.SolvePathParameterization()
            return PathParameterizedTrajectory(traj, time_traj)

        plant = self._plant
        gcs = GcsTrajectoryOptimization(plant.num_positions())

        workspace = gcs.AddRegions(
            [HPolyhedron.MakeBox(plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits())],
            5,
            1,
            60,
        )

        vel_limits = self.vel_limits
        accel_limits = self.accel_limits
        # Set non-zero h_min for start and goal to enforce zero velocity.
        start = gcs.AddRegions([Point(q_start)], order=1, h_min=0.1)
        goal = gcs.AddRegions([Point(q_goal)], order=1, h_min=0.1)
        goal.AddVelocityBounds([0] * plant.num_positions(), [0] * plant.num_positions())
        gcs.AddEdges(start, workspace)
        gcs.AddEdges(workspace, goal)
        gcs.AddTimeCost()
        gcs.AddPathLengthCost()
        gcs.AddVelocityBounds(-vel_limits, vel_limits)

        traj, result = gcs.SolvePath(start, goal)

        if result.is_success():
            return make_traj_toppra(traj, plant, vel_limits=vel_limits, accel_limits=accel_limits)
        else:
            return PiecewisePolynomial.FirstOrderHold([0, 1], np.column_stack((q_start, q_goal)))

    def solve_ik(
        self,
        pose: RigidTransform,
        default_joint_positions: npt.NDArray[np.float64],
        disregard_angle: bool = False,
    ) -> npt.NDArray[np.float64]:
        # Plant needs to be just the robot without other objects
        # Need to create a new context that the IK can use for solving the problem
        plant = self._plant
        ik = InverseKinematics(plant, with_joint_limits=True)  # type: ignore
        pusher_frame = plant.GetFrameByName("pusher_end")
        EPS = 1e-3

        ik.AddPositionConstraint(
            pusher_frame,
            np.zeros(3),
            plant.world_frame(),
            pose.translation() - np.ones(3) * EPS,
            pose.translation() + np.ones(3) * EPS,
        )

        if disregard_angle:
            z_unit_vec = np.array([0, 0, 1])
            ik.AddAngleBetweenVectorsConstraint(
                pusher_frame,
                z_unit_vec,
                plant.world_frame(),
                -z_unit_vec,  # The pusher object has z-axis pointing up
                0 - EPS,
                0 + EPS,
            )

        else:
            ik.AddOrientationConstraint(
                pusher_frame,
                RotationMatrix(),
                plant.world_frame(),
                pose.rotation(),
                EPS,
            )

        # Cost on deviation from default joint positions
        prog = ik.get_mutable_prog()
        q = ik.q()

        q0 = default_joint_positions
        prog.AddQuadraticErrorCost(np.identity(len(q)), q0, q)
        prog.SetInitialGuess(q, q0)

        result = Solve(ik.prog())
        assert result.is_success()

        q_sol = result.GetSolution(q)
        return q_sol
