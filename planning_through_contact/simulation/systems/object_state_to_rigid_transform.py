from pydrake.all import AbstractValue, LeafSystem, MultibodyPlant, RigidTransform


class ObjectStateToRigidTransform(LeafSystem):
    """
    Converts a slider/object state vector (positions + velocities) to a RigidTransform.
    """

    def __init__(self, plant: MultibodyPlant, object_model_name: str):
        """
        Args:
            plant: MultibodyPlant containing the object model.
            object_model_name: Name of the object model instance (e.g., "arbitrary", "box", "t_pusher").
        """
        super().__init__()

        self._plant = plant
        self._plant_context = self._plant.CreateDefaultContext()
        self._object_model_name = object_model_name
        self._object_model_instance_index = plant.GetModelInstanceByName(object_model_name)

        # Get the base body of the object (the main body of the model instance)
        # For the slider, this should be the floating base body
        self._object_body = self._plant.GetBodyByName(object_model_name)

        self._num_positions = self._plant.num_positions(self._object_model_instance_index)
        self._num_velocities = self._plant.num_velocities(self._object_model_instance_index)

        # Input port: state vector [positions, velocities]
        self._object_state_input_port = self.DeclareVectorInputPort(
            "state",
            self._num_positions + self._num_velocities,
        )

        # Output port: RigidTransform
        self._pose_output_port = self.DeclareAbstractOutputPort(
            "pose",
            lambda: AbstractValue.Make(RigidTransform()),
            self.DoCalcOutput,
        )

    def DoCalcOutput(self, context, output):
        object_state = self.EvalVectorInput(context, 0).get_value()
        q = object_state[: self._num_positions]

        # Set the positions in the plant context
        self._plant.SetPositions(self._plant_context, self._object_model_instance_index, q)

        # Evaluate the pose of the object body in world frame
        pose = self._plant.EvalBodyPoseInWorld(self._plant_context, self._object_body)

        output.set_value(pose)
