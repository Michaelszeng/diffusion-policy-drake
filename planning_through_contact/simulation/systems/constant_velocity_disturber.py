import numpy as np
from pydrake.all import (
    AbstractValue,
    ExternallyAppliedSpatialForce_,
    LeafSystem,
    SpatialForce_,
)


class ConstantVelocityDisturber(LeafSystem):
    """
    Outputs a list[ExternallyAppliedSpatialForce_[float]] to push a body so that its
    world twist approaches a desired constant planar velocity (x-y) with zero angular target.
    Connect output to plant.get_applied_spatial_force_input_port().

    The disturbance is only applied when both:
      1. The "enable" input port is True (controlled by IiwaPlanner PUSHING mode)
      2. The "success" input port is False (i.e., goal not yet reached)

    Args:
      plant: MultibodyPlant[float] (finalized)
      scene_graph: SceneGraph (not used; keep if you want geometry queries)
      body_index: the body to disturb (e.g., plant.GetBodyByName("tee").index())
      Kp, Kd: gains on velocity error (units: N·s/m for linear part)
      force_cap: optional clamp on max planar force magnitude
    """

    def __init__(self, plant, body_index, Kp=60.0, Kd=5.0, force_cap=0.5):
        super().__init__()
        self._plant = plant
        self._context_plant = plant.CreateDefaultContext()  # scratch for evals if needed
        self._body_index = body_index
        self._Kp = float(Kp)
        self._Kd = float(Kd)
        self._force_cap = force_cap
        self._v_xy_des = np.array([0.0, 0.0])  # Initialze to 0
        self._success_printed = False  # Track whether we've printed the success message

        # Input port to enable/disable the velocity disturbance. This is controlled by whether the IiwaPlanner is in
        # PUSHING mode.
        self._enable_in = self.DeclareVectorInputPort("enable", 1)

        # Input port for success indicator. Disturbance is disabled when success is achieved (success=1.0).
        self._success_in = self.DeclareVectorInputPort("success", 1)

        self._state_in = self.DeclareVectorInputPort("x_plant", self._plant.num_multibody_states())

        self._y = self.DeclareAbstractOutputPort(
            "spatial_forces",  # to connect to plant.get_applied_spatial_force_input_port()
            lambda: AbstractValue.Make([ExternallyAppliedSpatialForce_[float]()]),
            self._CalcForces,
        )

    def set_constant_velocity_disturbance(self, v_xy_des: float):
        """
        Args:
            v_xy_des: desired constant planar linear velocity in world (m/s)
        """
        self._v_xy_des = np.asarray(v_xy_des).reshape(2)

    def _CalcForces(self, context, output):
        enabled = self._enable_in.Eval(context)[0]
        success = self._success_in.Eval(context)[0]

        # Apply disturbance only if enabled AND slider hasn't reached goal yet
        # success >= 1.0 means slider has reached goal (1.0 = slider only, 2.0 = both)
        if success >= 1.0 and not self._success_printed:
            print("[ConstantVelocityDisturber] Success achieved -- disabling velocity disturbance.")
            self._success_printed = True

        if not enabled or success >= 1.0:
            output.set_value([])  # no disturbance outside PUSHING mode or after slider reaches goal
            return

        # Mirror the plant’s state so we can query velocities.
        x = self._state_in.Eval(context)
        self._plant.SetPositionsAndVelocities(self._context_plant, x)

        V_WB = self._plant.EvalBodySpatialVelocityInWorld(
            self._context_plant, self._plant.get_body(self._body_index)
        )  # SpatialVelocity
        v_WB = V_WB.translational()  # 3d np.array

        # Desired twist: planar linear velocity, no angular velocity target.
        v_des = np.array([self._v_xy_des[0], self._v_xy_des[1], 0.0])

        # Velocity error
        ev = v_des - v_WB

        # Simple PD on velocity (linear part only by default; set torque=0)
        # If you want to also damp spin, you can give angular gains below.
        F_planar = self._Kp * ev + self._Kd * (-v_WB)  # crude damping around 0
        F_W = np.array([F_planar[0], F_planar[1], 0.0])
        tau_W = np.zeros(3)

        # Optional clamp to avoid blowing up the contact solver
        if self._force_cap is not None:
            norm = np.linalg.norm(F_W[:2])
            if norm > self._force_cap and norm > 1e-12:
                F_W[:2] *= self._force_cap / norm

        # Apply force at body origin Bo, expressed in world
        sf = ExternallyAppliedSpatialForce_[float]()
        sf.body_index = self._body_index
        sf.p_BoBq_B = np.zeros(3)  # application point = Bo
        sf.F_Bq_W = SpatialForce_[float](tau_W, F_W)

        output.set_value([sf])
