from collections import deque

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
      velocity_window_size: moving average window size for velocity smoothing
      tune_mode: if True, will print debug information like to help tune the controller

    Note: DO NOT increase force_cap beyond 1.0. This makes the pushing dynamics against the iiwa weird.
    """

    def __init__(self, plant, body_index, Kp=60.0, Kd=5.0, force_cap=1.0, velocity_window_size=5, tune_mode=False):
        super().__init__()
        self._plant = plant
        self._context_plant = plant.CreateDefaultContext()  # scratch for evals if needed
        self._body_index = body_index
        self._Kp = float(Kp)
        self._Kd = float(Kd)
        self._force_cap = force_cap
        self._v_xy_des = np.array([0.0, 0.0])  # Initialze to 0
        self._success_printed = False  # Track whether we've printed the success message
        self._window_size = velocity_window_size
        self._tune_mode = tune_mode
        # Velocity history buffer for moving average smoothing
        self._velocity_history = deque(maxlen=velocity_window_size)

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

    def _get_smoothed_velocity(self, v_raw: np.ndarray) -> np.ndarray:
        """
        Apply moving average smoothing to velocity estimate.

        Args:
            v_raw: Raw velocity vector (3D)

        Returns:
            Smoothed velocity vector (3D)
        """
        self._velocity_history.append(v_raw.copy())
        if len(self._velocity_history) > 0:
            smoothed = np.mean(self._velocity_history, axis=0)
            if self._tune_mode:
                print(f"||v_WB_smooth||: {np.linalg.norm(smoothed):.4f}")
            # print(f"||v_WB_raw||: {np.linalg.norm(v_raw):.4f}")
            return smoothed
        else:
            return v_raw

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
        v_WB_raw = V_WB.translational()  # 3d np.array
        v_WB = self._get_smoothed_velocity(v_WB_raw)  # moving average smoothed

        # Desired twist: planar linear velocity, no angular velocity target.
        v_des = np.array([self._v_xy_des[0], self._v_xy_des[1], 0.0])

        # Velocity error
        ev = v_des - v_WB

        # Simple PD on velocity (linear part only by default; set torque=0)
        F_planar = self._Kp * ev + self._Kd * (-v_WB)  # crude damping around 0
        F_W = np.array([F_planar[0], F_planar[1], 0.0])
        tau_W = np.zeros(3)

        # Project force onto desired velocity direction (zero if opposing)
        v_des_norm = np.linalg.norm(v_des)
        if v_des_norm > 1e-12:
            force_dot_vdes = np.dot(F_W, v_des)
            if force_dot_vdes > 0:
                # Keep only component in direction of desired velocity
                F_W = (force_dot_vdes / (v_des_norm**2)) * v_des
            else:
                # Force opposes or is perpendicular - zero it
                F_W = np.zeros(3)

        # Optional clamp to avoid blowing up the contact solver
        if self._force_cap is not None:
            norm = np.linalg.norm(F_W[:2])
            if norm > self._force_cap and norm > 1e-12:
                F_W[:2] *= self._force_cap / norm
                if self._tune_mode:
                    print(f"Clamped force: {norm}")
            else:
                if self._tune_mode:
                    print(f"Applied force: {norm}")

        # Apply force at body origin Bo, expressed in world
        sf = ExternallyAppliedSpatialForce_[float]()
        sf.body_index = self._body_index
        sf.p_BoBq_B = np.zeros(3)  # application point = Bo
        sf.F_Bq_W = SpatialForce_[float](tau_W, F_W)

        output.set_value([sf])
