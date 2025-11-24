import numpy as np
from pydrake.all import (
    AbstractValue,
    ExternallyAppliedSpatialForce_,
    LeafSystem,
    SpatialForce_,
)


class PeriodicImpulseDisturber(LeafSystem):
    """
    Applies a *fixed impulse* to a body every `period_s` seconds by emitting a
    constant spatial force for `pulse_duration_s` seconds.

    Disturbance is active only when:
      (1) enable == 1.0  (e.g., planner in PUSHING)
      (2) success < 1.0  (i.e., goal not reached)

    Args:
      plant: finalized MultibodyPlant[float]
      body_index: BodyIndex of target body
      period_s: seconds between pulse starts
      pulse_duration_s: seconds each pulse lasts (must be > 0)
      impulse_magnitude: scalar magnitude of impulse (N·s) applied at body origin, direction randomized in XY
      angular_impulse_magnitude: scalar magnitude of angular impulse (N·m·s) around Z axis, sign randomized
      update_period_s: how often this system ticks its internal timer (should be <= your sim step)
      tune_mode: if True, prints pulse events and applied magnitudes
    """

    def __init__(
        self,
        plant,
        body_index,
        period_s=2.0,
        pulse_duration_s=0.05,
        impulse_magnitude=0.5,
        angular_impulse_magnitude=0.0,
        update_period_s=0.005,
        tune_mode=False,
    ):
        super().__init__()
        assert pulse_duration_s > 0.0, "pulse_duration_s must be > 0"

        self._plant = plant
        self._body_index = body_index
        self._period = float(period_s)
        self._pulse = float(pulse_duration_s)
        self._impulse_magnitude = float(impulse_magnitude)  # scalar magnitude
        self._angular_impulse_magnitude = float(angular_impulse_magnitude)  # scalar magnitude
        self._impulse = np.zeros(3)  # will be set when pulse starts
        self._ang_impulse = np.zeros(3)  # will be set when pulse starts
        self._tune = bool(tune_mode)

        # Reproducible RNG for direction sampling
        self._rng = np.random.default_rng()

        # Discrete state: [last_pulse_start_time], [firing_flag]
        self._idx_last_start = self.DeclareDiscreteState(1)
        self._idx_firing = self.DeclareDiscreteState(1)

        # Inputs
        self._enable_in = self.DeclareVectorInputPort("enable", 1)
        self._success_in = self.DeclareVectorInputPort("success", 1)

        # Output: externally applied spatial forces
        self._y = self.DeclareAbstractOutputPort(
            "spatial_forces",
            lambda: AbstractValue.Make([ExternallyAppliedSpatialForce_[float]()]),
            self._CalcForces,
        )

        # Periodic update to manage pulse timing & (re)direction
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=update_period_s,
            offset_sec=0.0,
            update=self._UpdateTimer,
        )

        self._success_printed = False

    ### Public helpers

    def set_rng(self, rng: np.random.Generator):
        """Set RNG for reproducible trials."""
        self._rng = rng

    def set_impulse(self, impulse_magnitude, angular_impulse_magnitude=None):
        """Set the impulse magnitudes (direction assigned randomly at each pulse)."""
        self._impulse_magnitude = float(impulse_magnitude)
        if angular_impulse_magnitude is not None:
            self._angular_impulse_magnitude = float(angular_impulse_magnitude)

    def set_period(self, period_s):
        self._period = float(period_s)

    def set_pulse_duration(self, pulse_duration_s):
        assert pulse_duration_s > 0.0
        self._pulse = float(pulse_duration_s)

    ### Internal Functions

    def _randomize_xy_direction(self):
        """
        Set the XY impulse direction to a random angle using the stored magnitude.
        Z component is always 0 (planar pushing).
        Also randomize the angular impulse sign around Z axis if magnitude > 0.
        """
        # Linear impulse: random XY direction with stored magnitude
        if self._impulse_magnitude <= 0.0:
            self._impulse = np.zeros(3)
        else:
            theta = self._rng.uniform(0.0, 2.0 * np.pi)
            self._impulse[0] = self._impulse_magnitude * np.cos(theta)
            self._impulse[1] = self._impulse_magnitude * np.sin(theta)
            self._impulse[2] = 0.0  # planar pushing

        # Angular impulse: random sign around Z axis only (planar rotation)
        if self._angular_impulse_magnitude <= 0.0:
            self._ang_impulse = np.zeros(3)
        else:
            # Random sign for rotation around Z axis
            sign = self._rng.choice([-1.0, 1.0])
            self._ang_impulse[0] = 0.0
            self._ang_impulse[1] = 0.0
            self._ang_impulse[2] = sign * self._angular_impulse_magnitude

    def _UpdateTimer(self, context, discrete_state):
        enabled = self._enable_in.Eval(context)[0] >= 0.5
        success = self._success_in.Eval(context)[0] >= 1.0
        t = context.get_time()

        last_start = discrete_state.get_mutable_vector(self._idx_last_start).GetAtIndex(0)
        firing = int(discrete_state.get_mutable_vector(self._idx_firing).GetAtIndex(0))

        if not enabled or success:
            # Gate off: stop firing and "sync" last_start to now
            discrete_state.get_mutable_vector(self._idx_firing).SetAtIndex(0, 0)
            discrete_state.get_mutable_vector(self._idx_last_start).SetAtIndex(0, t)
            return

        if firing:
            # End pulse after duration
            if t - last_start >= self._pulse:
                discrete_state.get_mutable_vector(self._idx_firing).SetAtIndex(0, 0)
                if self._tune:
                    print(f"[PeriodicImpulseDisturber] Pulse end @ t={t:.3f}")
        else:
            # Start new pulse on schedule
            if t - last_start >= self._period:
                # Choose a fresh planar direction (reproducible via rng)
                self._randomize_xy_direction()
                discrete_state.get_mutable_vector(self._idx_firing).SetAtIndex(0, 1)
                discrete_state.get_mutable_vector(self._idx_last_start).SetAtIndex(0, t)
                if self._tune:
                    print(f"[PeriodicImpulseDisturber] Pulse start @ t={t:.3f}; duration={self._pulse:.3f} s")

    def _CalcForces(self, context, output):
        enabled = self._enable_in.Eval(context)[0] >= 0.5
        success = self._success_in.Eval(context)[0] >= 1.0

        if success >= 1.0 and not self._success_printed:
            print("[PeriodicImpulseDisturber] Success achieved -- disabling impulse disturbance.")
            self._success_printed = True

        if not enabled or success >= 1.0:
            output.set_value([])
            return

        self._success_printed = False

        firing = int(context.get_discrete_state(self._idx_firing).GetAtIndex(0))
        if not firing:
            output.set_value([])
            return

        # Impulse over pulse window -> constant force/torque
        F_W = self._impulse
        tau_W = self._ang_impulse

        if self._tune:
            print(f"[PeriodicImpulseDisturber] |F|={np.linalg.norm(F_W):.3f} N, |τ|={np.linalg.norm(tau_W):.3f} N·m")

        sf = ExternallyAppliedSpatialForce_[float]()
        sf.body_index = self._body_index
        sf.p_BoBq_B = np.zeros(3)
        sf.F_Bq_W = SpatialForce_[float](tau_W, F_W)
        output.set_value([sf])
