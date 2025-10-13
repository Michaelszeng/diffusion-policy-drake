from dataclasses import dataclass, field
from functools import cached_property
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from pydrake.geometry import Shape as DrakeShape
from pydrake.math import RigidTransform, RotationMatrix

from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    CollisionGeometry,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.hyperplane import (
    Hyperplane,
    construct_2d_plane_from_points,
)


@dataclass(frozen=True)
class TPusher2d(CollisionGeometry):
    """
    T-shaped collision geometry constructed from two boxes.

     ____________
    |   box 1    |
    |____________|
        | b2 |
        |    |
        |    |
        |____|

    Origin is placed at com_offset from the origin of box_1.

    Used for simulation where the actual geometry is defined in an SDF file.
    This class provides geometric properties (vertices, faces, dimensions) needed
    for workspace sampling and visualization.
    """

    box_1: Box2d = field(default_factory=lambda: Box2d(0.2, 0.05))
    box_2: Box2d = field(default_factory=lambda: Box2d(0.05, 0.15))

    @property
    def collision_geometry_names(self) -> List[str]:
        return [
            "t_pusher::t_pusher_bottom_collision",
            "t_pusher::t_pusher_top_collision",
        ]

    @classmethod
    def from_drake(cls, drake_shape: DrakeShape):
        raise NotImplementedError()

    @property
    def com_offset(self) -> npt.NDArray[np.float64]:
        """Center of mass offset for the T-pusher."""
        y_offset = -0.04285714
        return np.array([0, y_offset]).reshape((-1, 1))

    @cached_property
    def vertices(self) -> List[npt.NDArray[np.float64]]:
        """
        v0___________v1
        |              |
        v7___v6___v3___v2
            |    |
            |    |
            |    |
            v5____v4
        """
        v0 = self.box_1.vertices[0]
        v1 = self.box_1.vertices[1]
        v2 = self.box_1.vertices[2]

        box_2_center = np.array([0, -self.box_1.height / 2 - self.box_2.height / 2]).reshape((-1, 1))
        v3 = box_2_center + self.box_2.vertices[1]
        v4 = box_2_center + self.box_2.vertices[2]
        v5 = box_2_center + self.box_2.vertices[3]
        v6 = box_2_center + self.box_2.vertices[0]

        v7 = self.box_1.vertices[3]
        vs = [v0, v1, v2, v3, v4, v5, v6, v7]

        # Apply COM offset
        vs_offset = [v - self.com_offset for v in vs]
        return vs_offset

    @property
    def num_vertices(self) -> int:
        return len(self.vertices)

    @cached_property
    def faces(self) -> List[Hyperplane]:
        """
        ______f0________
        f7              f1
        |_f6________f2__|
            |    |
            f5   f3
            |    |
            |_f4_|
        """
        wrap_around = lambda num: num % self.num_vertices
        pairwise_indices = [(idx, wrap_around(idx + 1)) for idx in range(self.num_vertices)]
        hyperplane_points = [(self.vertices[i], self.vertices[j]) for i, j in pairwise_indices]
        hyperplanes = [construct_2d_plane_from_points(p1, p2) for p1, p2 in hyperplane_points]
        return hyperplanes

    @property
    def width(self) -> float:
        return self.box_1.width

    @property
    def height(self) -> float:
        return self.box_1.height + self.box_2.height

    @property
    def vertices_for_plotting(self) -> npt.NDArray[np.float64]:
        vertices = np.hstack([self.vertices[idx] for idx in range(self.num_vertices)])
        return vertices

    def get_as_boxes(self, z_value: float = 0.0) -> Tuple[List[Box2d], List[RigidTransform]]:
        """
        Get the two primitive boxes and their transforms for Drake simulation.

        Args:
            z_value: Z-coordinate for the boxes (planar objects lie in XY plane)

        Returns:
            Tuple of ([box_1, box_2], [transform_1, transform_2])
        """
        box_1 = self.box_1
        box_1_center = np.array([0, 0, z_value])
        box_1_center[:2] -= self.com_offset.flatten()
        transform_1 = RigidTransform(RotationMatrix.Identity(), box_1_center)  # type: ignore

        box_2 = self.box_2
        box_2_center = np.array([0, -self.box_1.height / 2 - self.box_2.height / 2, z_value])
        box_2_center[:2] -= self.com_offset.flatten()
        transform_2 = RigidTransform(RotationMatrix.Identity(), box_2_center)

        return [box_1, box_2], [transform_1, transform_2]

    # ============================================================================
    # Abstract method stubs (not used in current simulation pipeline)
    # ============================================================================

    @property
    def contact_locations(self) -> List["PolytopeContactLocation"]:
        """Not used in simulation-only workflow."""
        from planning_through_contact.geometry.collision_geometry.collision_geometry import (
            ContactLocation,
            PolytopeContactLocation,
        )

        return [PolytopeContactLocation(pos=ContactLocation.FACE, idx=idx) for idx in range(len(self.faces))]

    @property
    def num_collision_free_regions(self) -> int:
        """Not used in simulation-only workflow."""
        raise NotImplementedError("Collision-free regions not implemented for simulation-only workflow")

    def get_collision_free_region_for_loc_idx(self, loc_idx: int) -> int:
        """Not used in simulation-only workflow."""
        raise NotImplementedError("Collision-free regions not implemented for simulation-only workflow")

    def get_contact_planes(self, idx: int) -> List[Hyperplane]:
        """Not used in simulation-only workflow."""
        raise NotImplementedError("Contact planes not implemented for simulation-only workflow")

    def get_planes_for_collision_free_region(self, idx: int) -> List[Hyperplane]:
        """Not used in simulation-only workflow."""
        raise NotImplementedError("Collision-free regions not implemented for simulation-only workflow")

    def get_proximate_vertices_from_location(
        self, location: "PolytopeContactLocation"
    ) -> List[npt.NDArray[np.float64]]:
        """Not used in simulation-only workflow."""
        raise NotImplementedError("Vertex location queries not implemented for simulation-only workflow")

    def get_neighbouring_vertices(
        self, location: "PolytopeContactLocation"
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Not used in simulation-only workflow."""
        raise NotImplementedError("Vertex location queries not implemented for simulation-only workflow")

    def get_hyperplane_from_location(self, location: "PolytopeContactLocation") -> Hyperplane:
        """Not used in simulation-only workflow."""
        raise NotImplementedError("Hyperplane location queries not implemented for simulation-only workflow")

    def get_norm_and_tang_vecs_from_location(
        self, location: "PolytopeContactLocation"
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Not used in simulation-only workflow."""
        raise NotImplementedError("Normal/tangent vectors not implemented for simulation-only workflow")

    def get_face_length(self, location: "PolytopeContactLocation") -> float:
        """Not used in simulation-only workflow."""
        raise NotImplementedError("Face length not implemented for simulation-only workflow")
