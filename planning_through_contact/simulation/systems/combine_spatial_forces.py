from pydrake.all import AbstractValue, ExternallyAppliedSpatialForce_, LeafSystem


class CombineSpatialForces(LeafSystem):
    """Adder to combine two ExternallyAppliedSpatialForce_[T] lists into one."""

    def __init__(self):
        super().__init__()
        exemplar = [ExternallyAppliedSpatialForce_[float]()]  # empty list exemplar
        self.DeclareAbstractInputPort("forces_A", AbstractValue.Make(exemplar))
        self.DeclareAbstractInputPort("forces_B", AbstractValue.Make(exemplar))
        self.DeclareAbstractOutputPort("combined", lambda: AbstractValue.Make(exemplar), self._combine)

    def _combine(self, context, output):
        a = self.get_input_port(0).Eval(context)
        b = self.get_input_port(1).Eval(context)
        output.set_value(list(a) + list(b))
