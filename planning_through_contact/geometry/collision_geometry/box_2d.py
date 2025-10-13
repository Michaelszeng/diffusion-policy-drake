from dataclasses import dataclass
from typing import List, Literal

import numpy as np
import numpy.typing as npt
from pydrake.geometry import Box as DrakeBox

from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    CollisionGeometry,
)
from planning_through_contact.geometry.hyperplane import (
    Hyperplane,
    construct_2d_plane_from_points,
)


@dataclass(frozen=True)
class Box2d(CollisionGeometry):
    """
    A two-dimensional box collision geometry.

    Used for simulation where the actual geometry is defined in an SDF file.
    This class provides geometric properties (vertices, faces, dimensions) needed
    for workspace sampling and visualization.
    """

    width: float
    height: float
    # v0 -- v1
    # |     |
    # v3 -- v2

    @property
    def collision_geometry_names(self) -> List[str]:
        return ["box::box_collision"]

    @classmethod
    def from_drake(cls, drake_box: DrakeBox, axis_mode: Literal["planar"] = "planar") -> "Box2d":
        """
        Constructs a two-dimensional box from a Drake 3D box.

        By default, it is assumed that the box is intended to be used with planar pushing, and
        hence the two-dimensional box is constructed with the 'depth' and 'width' from the Drake box.
        """
        if axis_mode == "planar":
            width = drake_box.depth()
            height = drake_box.width()
            return cls(width, height)
        else:
            raise NotImplementedError("Only planar conversion from 3D drake box is currently supported.")

    @property
    def _v0(self) -> npt.NDArray[np.float64]:
        return np.array([[-self.width / 2], [self.height / 2]])

    @property
    def _v1(self) -> npt.NDArray[np.float64]:
        return np.array([[self.width / 2], [self.height / 2]])

    @property
    def _v2(self) -> npt.NDArray[np.float64]:
        return np.array([[self.width / 2], [-self.height / 2]])

    @property
    def _v3(self) -> npt.NDArray[np.float64]:
        return np.array([[-self.width / 2], [-self.height / 2]])

    @property
    def vertices(self) -> List[npt.NDArray[np.float64]]:
        return [self._v0, self._v1, self._v2, self._v3]

    @property
    def vertices_for_plotting(self) -> npt.NDArray[np.float64]:
        return np.hstack([self._v0, self._v1, self._v2, self._v3])

    # v0 - f0 - v1
    # |          |
    # f3         f1
    # |          |
    # v3 --f2--- v2

    @property
    def _face_0(self) -> Hyperplane:
        return construct_2d_plane_from_points(self._v0, self._v1)

    @property
    def _face_1(self) -> Hyperplane:
        return construct_2d_plane_from_points(self._v1, self._v2)

    @property
    def _face_2(self) -> Hyperplane:
        return construct_2d_plane_from_points(self._v2, self._v3)

    @property
    def _face_3(self) -> Hyperplane:
        return construct_2d_plane_from_points(self._v3, self._v0)

    @property
    def faces(self) -> List[Hyperplane]:
        return [self._face_0, self._face_1, self._face_2, self._face_3]
