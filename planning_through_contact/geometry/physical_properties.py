from dataclasses import dataclass
from typing import Union

import numpy as np


@dataclass
class PhysicalProperties:
    """
    A dataclass for physical properties.

    See https://drake.mit.edu/doxygen_cxx/group__hydroelastic__user__guide.html for
    more information about these properties.
    """

    mass: float
    """Mass in kg."""
    inertia: np.ndarray
    """Moment of inertia of shape (3,3)."""
    center_of_mass: np.ndarray
    """The center of mass that the inertia is about of shape (3,)."""
    is_compliant: bool
    """Whether the object is compliant or rigid in case of a Hydroelastic contact model.
    If compliant, the compliant Hydroelastic arguments are required."""
    hydroelastic_modulus: Union[float, None] = None
    """This is the measure of how stiff the material is. It directly defines how much
    pressure is exerted given a certain amount of penetration. More pressure leads to
    greater forces. Larger values create stiffer objects."""
    hunt_crossley_dissipation: Union[float, None] = None
    """A non-negative real value. This gives the contact an energy-damping property."""
    mu_dynamic: Union[float, None] = None
    """Dynamic coefficient of friction."""
    mu_static: Union[float, None] = None
    """Static coefficient of friction. Not used in discrete systems."""
    mesh_resolution_hint: Union[float, None] = None
    """A positive real value in meters. Most shapes (capsules, cylinders, ellipsoids,
    spheres) need to be tessellated into meshes. The resolution hint controls the
    fineness of the meshes. It is a no-op for mesh geometries and is consequently not
    required for mesh geometries."""

    def __post_init__(self):
        # Convert all lists to numpy arrays
        if self.inertia is not None:
            self.inertia = np.array(self.inertia)
        if self.center_of_mass is not None:
            self.center_of_mass = np.array(self.center_of_mass)

    def __eq__(self, other):
        if not isinstance(other, PhysicalProperties):
            return False

        for key, value in self.__dict__.items():
            other_value = getattr(other, key, None)

            if isinstance(value, np.ndarray):
                if not np.array_equal(value, other_value):
                    return False
            else:
                if value != other_value:
                    return False

        return True
