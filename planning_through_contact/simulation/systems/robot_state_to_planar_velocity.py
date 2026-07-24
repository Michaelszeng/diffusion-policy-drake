import numpy as np
from pydrake.all import JacobianWrtVariable, LeafSystem, MultibodyPlant


class RobotStateToPlanarVelocity(LeafSystem):
    """
    Converts a robot state vector (positions + velocities) to planar velocity (x, y).
    """

    def __init__(self, plant: MultibodyPlant, robot_model_name: str, end_effector_frame_name: str = "pusher_end"):
        """
        Args:
            plant: MultibodyPlant containing the robot model.
            robot_model_name: Name of the robot model instance (e.g., "iiwa").
            end_effector_frame_name: Name of the end effector frame (default: "pusher").
        """
        super().__init__()

        self._plant = plant
        self._plant_context = self._plant.CreateDefaultContext()
        self._robot_model_name = robot_model_name
        self._robot_model_instance_index = plant.GetModelInstanceByName(robot_model_name)
        self._end_effector_frame = self._plant.GetFrameByName(end_effector_frame_name)
        self._num_positions = self._plant.num_positions(self._robot_model_instance_index)
        self._num_velocities = self._plant.num_velocities(self._robot_model_instance_index)

        # Input port: state vector [positions, velocities]
        self._robot_state_input_port = self.DeclareVectorInputPort(
            "state",
            self._num_positions + self._num_velocities,
        )

        # Output port: 2D planar velocity [vx, vy]
        self._velocity_output_port = self.DeclareVectorOutputPort(
            "planar_velocity",
            2,
            self.DoCalcOutput,
        )

    def DoCalcOutput(self, context, output):
        """
        Extract positions and velocities from robot state, compute the pusher's
        translational velocity in world frame, and output the x-y components.
        """
        robot_state = self.EvalVectorInput(context, 0).get_value()
        q = robot_state[: self._num_positions]
        v = robot_state[self._num_positions :]

        # Set the robot state in the plant context
        self._plant.SetPositions(self._plant_context, self._robot_model_instance_index, q)
        self._plant.SetVelocities(self._plant_context, self._robot_model_instance_index, v)

        # Compute the translational velocity Jacobian of the end effector in world frame
        # This returns a (3 x nv) matrix where nv is the number of velocities
        J_v_WF = self._plant.CalcJacobianTranslationalVelocity(
            self._plant_context,
            with_respect_to=JacobianWrtVariable.kV,  # with respect to generalized velocities
            frame_B=self._end_effector_frame,
            p_BoBi_B=np.zeros((3, 1)),  # Point P coincides with frame B's origin
            frame_A=self._plant.world_frame(),
            frame_E=self._plant.world_frame(),
        )

        # Get ALL velocities from the plant (needed because Jacobian is w.r.t. all generalized velocities)
        v_all = self._plant.GetVelocities(self._plant_context)

        # Compute translational velocity v_WF
        v_WF = J_v_WF @ v_all  # [vx, vy, vz]

        # Extract the x and y components of the translational velocity
        planar_velocity = np.array([v_WF[0], v_WF[1]])

        output.SetFromVector(planar_velocity)
