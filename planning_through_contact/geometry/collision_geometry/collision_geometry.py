from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, List, NamedTuple, TypeVar

import numpy as np
import numpy.typing as npt
from pydrake.geometry import Shape as DrakeShape

from planning_through_contact.geometry.hyperplane import Hyperplane


class ContactLocation(Enum):
    FACE = 1
    VERTEX = 2


class ContactMode(Enum):
    ROLLING = 1
    SLIDING_LEFT = 2
    SLIDING_RIGHT = 3


class PolytopeContactLocation(NamedTuple):
    """Named tuple for contact locations (kept for type compatibility)."""

    pos: ContactLocation
    idx: int

    def __str__(self) -> str:
        return f"{self.pos.name}_{self.idx}"


class CollisionGeometry(ABC):
    """
    Abstract base class for collision geometries used in simulation.

    This class defines the minimal interface needed for Drake-based simulation.
    All optimization-planner-specific methods have been removed.
    """

    @property
    @abstractmethod
    def collision_geometry_names(self) -> List[str]:
        """Drake collision geometry names (must match SDF file)."""
        ...

    @property
    @abstractmethod
    def vertices(self) -> List[npt.NDArray[np.float64]]:
        """Ordered vertices of the geometry."""
        ...

    @property
    @abstractmethod
    def faces(self) -> List[Hyperplane]:
        """Hyperplanes representing each face/edge of the geometry."""
        ...

    @property
    @abstractmethod
    def vertices_for_plotting(self) -> npt.NDArray[np.float64]:
        """Vertices formatted for plotting (2 x N array)."""
        ...

    @classmethod
    @abstractmethod
    def from_drake(cls, drake_shape: DrakeShape):
        """Construct from a Drake shape (required by interface)."""
        ...

    @property
    def max_dist_from_com(self) -> float:
        """Maximum distance from center of mass to any vertex."""
        dists = [np.linalg.norm(v) for v in self.vertices]
        return np.max(dists)  # type: ignore

    T = TypeVar("T", bound=Any)

    def get_p_Wv_i(self, i: int, R_WB: npt.NDArray[T], p_WB: npt.NDArray[T]) -> npt.NDArray[T]:
        """
        Get the position of vertex i in the world frame.

        Args:
            i: Vertex index
            R_WB: Rotation matrix from body to world frame
            p_WB: Position of body origin in world frame

        Returns:
            Position of vertex i in world frame
        """
        p_Bv_i = self.vertices[i]
        return p_WB + R_WB @ p_Bv_i  # type: ignore
