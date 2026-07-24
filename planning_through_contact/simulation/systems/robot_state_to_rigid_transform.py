from pydrake.all import AbstractValue, LeafSystem, MultibodyPlant, RigidTransform


class RobotStateToRigidTransform(LeafSystem):
    """
    Converts a robot state vector (positions + velocities) to a RigidTransform.
    """

    def __init__(self, plant: MultibodyPlant, robot_model_name: str, offset=None):
        """
        Args:
            plant: MultibodyPlant containing the robot model.
            robot_model_name: Name of the robot model instance (e.g., "iiwa").
            offset: Optional 3D offset vector to apply to the pusher pose in the pusher's body frame.
        """
        super().__init__()

        self._plant = plant
        self._plant_context = self._plant.CreateDefaultContext()
        self._offset = offset
        self._robot_model_name = robot_model_name
        self._robot_model_instance_index = plant.GetModelInstanceByName(robot_model_name)
        self._num_positions = self._plant.num_positions(self._robot_model_instance_index)
        self._num_velocities = self._plant.num_velocities(self._robot_model_instance_index)

        # Input port: state vector [positions, velocities]
        self._robot_state_input_port = self.DeclareVectorInputPort(
            "state",
            self._num_positions + self._num_velocities,
        )

        # Output port: RigidTransform
        self._pose_output_ports = self.DeclareAbstractOutputPort(
            "pose",
            lambda: AbstractValue.Make(RigidTransform()),
            self.DoCalcOutput,
        )

    def DoCalcOutput(self, context, output):
        robot_state = self.EvalVectorInput(context, 0).get_value()
        q = robot_state[: self._num_positions]

        # Set the robot positions in the plant context
        self._plant.SetPositions(self._plant_context, self._robot_model_instance_index, q)

        # Evaluate the pose of the pusher body in world frame
        pose = self._plant.EvalBodyPoseInWorld(self._plant_context, self._plant.GetBodyByName("pusher"))

        # Apply optional offset in the pusher's body frame
        if self._offset:
            pose.set_translation(pose.translation() + pose.rotation() * self._offset)

        output.set_value(pose)
