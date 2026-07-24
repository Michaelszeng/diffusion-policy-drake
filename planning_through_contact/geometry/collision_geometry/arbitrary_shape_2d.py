"""
Geometry representation for arbitrary 2D shapes composed of axis-aligned boxes.
"""

import logging
import pickle
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from pydrake.geometry import Shape as DrakeShape
from pydrake.math import RigidTransform
from scipy.spatial.transform import Rotation as R

from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    CollisionGeometry,
)
from planning_through_contact.geometry.hyperplane import (
    Hyperplane,
    construct_2d_plane_from_points,
)
from planning_through_contact.geometry.physical_properties import PhysicalProperties
from planning_through_contact.utils import write_atomic

## SDF Construction Utils


def load_primitive_info(primitive_info_file: str) -> List[Dict[str, Any]]:
    with open(primitive_info_file, "rb") as f:
        primitive_info = pickle.load(f)
    return primitive_info


def construct_drake_proximity_properties_sdf_str(physical_properties: PhysicalProperties, is_hydroelastic: bool) -> str:
    """
    Constructs a Drake proximity properties SDF string using the proximity properties
    contained in `physical_properties`. Only adds the Hydroelastic properties if
    `is_hydroelastic` is true.
    """
    proximity_properties_str = """
            <drake:proximity_properties>
        """
    if is_hydroelastic:
        if physical_properties.is_compliant:
            assert (
                physical_properties.hydroelastic_modulus is not None
            ), "Require a Hydroelastic modulus for compliant Hydroelastic objects!"
            proximity_properties_str += f"""
                        <drake:compliant_hydroelastic/>
                        <drake:hydroelastic_modulus>
                            {physical_properties.hydroelastic_modulus}
                        </drake:hydroelastic_modulus>
                """
        else:
            proximity_properties_str += """
                    <drake:rigid_hydroelastic/>
            """
        if physical_properties.mesh_resolution_hint is not None:
            proximity_properties_str += f"""
                    <drake:mesh_resolution_hint>
                        {physical_properties.mesh_resolution_hint}
                    </drake:mesh_resolution_hint>
            """
    if physical_properties.hunt_crossley_dissipation is not None:
        proximity_properties_str += f"""
                    <drake:hunt_crossley_dissipation>
                        {physical_properties.hunt_crossley_dissipation}
                    </drake:hunt_crossley_dissipation>
            """
    if physical_properties.mu_dynamic is not None:
        proximity_properties_str += f"""
                    <drake:mu_dynamic>
                        {physical_properties.mu_dynamic}
                    </drake:mu_dynamic>
            """
    if physical_properties.mu_static is not None:
        proximity_properties_str += f"""
                    <drake:mu_static>
                        {physical_properties.mu_static}
                    </drake:mu_static>
            """
    proximity_properties_str += """
            </drake:proximity_properties>
        """
    return proximity_properties_str


def get_primitive_geometry_str(primitive_geometry: Dict[str, Any]) -> str:
    if primitive_geometry["name"] == "ellipsoid":
        radii = primitive_geometry["radii"]
        geometry = f"""
            <ellipsoid>
                <radii>{radii[0]} {radii[1]} {radii[2]}</radii>
            </ellipsoid>
        """
    elif primitive_geometry["name"] == "sphere":
        radius = primitive_geometry["radius"]
        geometry = f"""
            <sphere>
                <radius>{radius}</radius>
            </sphere>
        """
    elif primitive_geometry["name"] == "box":
        size = primitive_geometry["size"]
        geometry = f"""
            <box>
                <size>{size[0]} {size[1]} {size[2]}</size>
            </box>
        """
    elif primitive_geometry["name"] == "cylinder":
        height = primitive_geometry["height"]
        radius = primitive_geometry["radius"]
        geometry = f"""
            <cylinder>
                <radius>{radius}</radius>
                <length>{height}</length>
            </cylinder>
        """
    else:
        raise RuntimeError(f"Unsupported primitive type: {primitive_geometry['name']}")

    return geometry


def create_processed_mesh_primitive_sdf_file(
    primitive_info: List[Dict[str, Any]],
    physical_properties: PhysicalProperties,
    global_translation: np.ndarray,
    output_file_path: str,
    model_name: str,
    base_link_name: str,
    is_hydroelastic: bool,
    visual_mesh_file_path: Optional[str] = None,
    com_override: Optional[np.ndarray] = None,
    rgba: Optional[List[float]] = None,
) -> None:
    """
    Creates and saves a processed mesh consisting of primitive geometries.

    :param primitive_info: A list of dicts containing primitive params. Each dict must
        contain "name" which can for example be sphere, ellipsoid, box, etc. and
        "transform" which is a homogenous transformation matrix. The other params are
        primitive dependent but must be sufficient to construct that primitive.
    :param physical_properties: The physical properties.
    :param global_translation: The translation of the processed mesh.
    :param output_file_path: The path to save the processed mesh SDF file.
    :param is_hydroelastic: Whether to make the body rigid hydroelastic.
    :param visual_mesh_file_path: The path to the mesh to use for the visual geometry.
    :param rgba: The color of the visual geometry. Only used if visual_mesh_file_path is
        None.
    :param com_override: The center of mass to use. If None, the center of mass from
        physical_properties is used.
    """
    com = com_override if com_override is not None else physical_properties.center_of_mass
    if physical_properties.center_of_mass is None:
        logging.warning("Center of mass not provided. Using [0, 0, 0] as default.")
    procesed_mesh_sdf_str = f"""
        <?xml version="1.0"?>
        <sdf version="1.7">
            <model name="{model_name}">
                <link name="{base_link_name}">
                    <inertial>
                        <inertia>
                            <ixx>{physical_properties.inertia[0, 0]}</ixx>
                            <ixy>{physical_properties.inertia[0, 1]}</ixy>
                            <ixz>{physical_properties.inertia[0, 2]}</ixz>
                            <iyy>{physical_properties.inertia[1, 1]}</iyy>
                            <iyz>{physical_properties.inertia[1, 2]}</iyz>
                            <izz>{physical_properties.inertia[2, 2]}</izz>
                        </inertia>
                        <mass>{physical_properties.mass}</mass>
                        <pose>{com[0]} {com[1]} {com[2]} 0 0 0</pose>
                    </inertial>
        """

    if visual_mesh_file_path is not None:
        procesed_mesh_sdf_str += f"""
                    <visual name="visual">
                        <pose>0 0 0 0 0 0</pose>
                        <geometry>
                            <mesh>
                                <uri>{visual_mesh_file_path}</uri>
                            </mesh>
                        </geometry>
                    </visual>
            """
    else:
        # Use primitives for the visual geometry.
        for i, info in enumerate(primitive_info):
            transform = info["transform"]
            translation = transform[:3, 3] + global_translation
            rotation = R.from_matrix(transform[:3, :3]).as_euler("XYZ")
            geometry = get_primitive_geometry_str(info)

            procesed_mesh_sdf_str += f"""
                <visual name="visual_{i}">
                    <pose>
                        {translation[0]} {translation[1]} {translation[2]} {rotation[0]} {rotation[1]} {rotation[2]}
                    </pose>
                    <geometry>
                        {geometry}
                    </geometry>
            """
            if rgba is not None:
                procesed_mesh_sdf_str += f"""
                    <material>
                        <diffuse> {rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]} </diffuse>
                    </material>
                """
            procesed_mesh_sdf_str += """
                </visual>
            """

    # Add the primitives
    for i, info in enumerate(primitive_info):
        transform = info["transform"]
        translation = transform[:3, 3] + global_translation
        rotation = R.from_matrix(transform[:3, :3]).as_euler("XYZ")
        geometry = get_primitive_geometry_str(info)

        procesed_mesh_sdf_str += f"""
            <collision name="collision_{i}">
                <pose>
                    {translation[0]} {translation[1]} {translation[2]} {rotation[0]} {rotation[1]} {rotation[2]}
                </pose>
                <geometry>
                    {geometry}
                </geometry>
            """

        assert (
            not is_hydroelastic or physical_properties.mesh_resolution_hint is not None
        ), "Require a mesh resolution hint for Hydroelastic primitive collision geometries!"
        procesed_mesh_sdf_str += construct_drake_proximity_properties_sdf_str(physical_properties, is_hydroelastic)

        procesed_mesh_sdf_str += """
                </collision>
            """

    procesed_mesh_sdf_str += """
                    </link>
                </model>
            </sdf>
        """
    write_atomic(output_file_path, procesed_mesh_sdf_str, "w")


## Geometry Utils


def compute_box_vertices(box):
    """Compute the 2D vertices of a box given its size and transform."""
    size = box["size"]
    transform = box["transform"]

    half_size_x = size[0] / 2
    half_size_y = size[1] / 2

    relative_vertices = np.array(
        [
            [-half_size_x, -half_size_y, 0, 1],  # Bottom-left
            [half_size_x, -half_size_y, 0, 1],  # Bottom-right
            [half_size_x, half_size_y, 0, 1],  # Top-right
            [-half_size_x, half_size_y, 0, 1],  # Top-left
        ]
    )

    vertices = [transform @ vertex for vertex in relative_vertices]
    vertices_2d = [(vertex[0], vertex[1]) for vertex in vertices]

    return vertices_2d


def compute_box_bounds(box):
    """Compute the bounding box (left, right, bottom, top) for a box."""
    size = box["size"]
    transform = box["transform"]

    center_x = transform[0, 3]
    center_y = transform[1, 3]

    left = center_x - size[0] / 2
    right = center_x + size[0] / 2
    bottom = center_y - size[1] / 2
    top = center_y + size[1] / 2

    return left, right, bottom, top


def is_inside_box(box, point):
    """Check if a 2D point is strictly inside the box (not on boundary)."""
    size = box["size"]
    transform = box["transform"]

    center_x = transform[0, 3]
    center_y = transform[1, 3]

    half_size_x = size[0] / 2
    half_size_y = size[1] / 2

    left_edge = center_x - half_size_x
    right_edge = center_x + half_size_x
    bottom_edge = center_y - half_size_y
    top_edge = center_y + half_size_y

    x, y = point
    is_inside = left_edge < x < right_edge and bottom_edge < y < top_edge

    return is_inside


def is_inside_any_box(boxes, point):
    """Check if a point is inside any of the boxes."""
    for box in boxes:
        if is_inside_box(box, point):
            return True
    return False


def is_point_on_box_edge(point, box):
    """Check if the point is on the edge of the box."""
    x, y = point
    left, right, bottom, top = compute_box_bounds(box)
    return (x == left or x == right) and bottom <= y <= top or (y == bottom or y == top) and left <= x <= right


def compute_intersection_points(boxes):
    """Compute intersection points where box edges meet."""
    intersections = []

    for i, box1 in enumerate(boxes):
        left1, right1, bottom1, top1 = compute_box_bounds(box1)

        for j, box2 in enumerate(boxes):
            if i == j:
                continue

            left2, right2, bottom2, top2 = compute_box_bounds(box2)

            # Check for horizontal and vertical overlap
            if left1 < right2 and right1 > left2:
                if bottom1 < top2 and top1 > bottom2:
                    horizontal_overlap = [max(left1, left2), min(right1, right2)]
                    vertical_overlap = [max(bottom1, bottom2), min(top1, top2)]

                    candidates = [
                        (horizontal_overlap[0], vertical_overlap[0]),
                        (horizontal_overlap[0], vertical_overlap[1]),
                        (horizontal_overlap[1], vertical_overlap[0]),
                        (horizontal_overlap[1], vertical_overlap[1]),
                    ]
                    for candidate in candidates:
                        if not is_inside_box(box1, candidate) and not is_inside_box(box2, candidate):
                            intersections.append(candidate)

    actual_intersections = []
    for point in intersections:
        if any(is_point_on_box_edge(point, box) for box in boxes):
            actual_intersections.append(point)

    return list(set(actual_intersections))


def compute_outer_vertices(boxes):
    """Compute all vertices on the outer boundary of the union of boxes."""
    box_vertices = [compute_box_vertices(box) for box in boxes]
    flattened_vertices = [vertex for vertices in box_vertices for vertex in vertices]
    intersection_points = compute_intersection_points(boxes)
    unique_points = list(set(flattened_vertices + intersection_points))
    outer_vertices = []
    for point in unique_points:
        if not is_inside_any_box(boxes, point):
            outer_vertices.append(point)
    return np.array(outer_vertices)


def connect_all_points(points):
    """Generate all possible edges between the given points."""
    edges = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            edges.append((points[i], points[j]))
    return edges


def filter_axis_aligned_edges(edges):
    """Filter out edges that are not perfectly horizontal or vertical."""
    axis_aligned_edges = []

    for edge in edges:
        (start_x, start_y), (end_x, end_y) = edge
        if start_x == end_x or start_y == end_y:
            axis_aligned_edges.append(edge)

    return axis_aligned_edges


def line_intersects_box(line, box, num_samples=10):
    """Check if a line segment intersects the interior of a box."""
    p1, p2 = line

    x_values = np.linspace(p1[0], p2[0], num_samples + 2)[1:-1]
    y_values = np.linspace(p1[1], p2[1], num_samples + 2)[1:-1]
    points = np.column_stack((x_values, y_values))

    for point in points:
        if is_inside_box(box, point):
            return True
    return False


def filter_edges_with_collision(edges, boxes):
    """Filter out all edges whose non-endpoint parts are inside a box."""
    filtered_edges = []
    for edge in edges:
        intersects = False
        for box in boxes:
            if line_intersects_box(edge, box):
                intersects = True
                break
        if not intersects:
            filtered_edges.append(edge)
    return filtered_edges


def compute_outer_edges(vertices, boxes):
    """Compute edges on the outer boundary."""
    edges = connect_all_points(vertices)
    axis_aligned_edges = filter_axis_aligned_edges(edges)
    outer_edges = filter_edges_with_collision(axis_aligned_edges, boxes)
    return outer_edges


def find_next_edge(edges, current_edge):
    """Find the next edge that connects to the current edge."""
    last_vertex = current_edge[1]
    for edge in edges:
        if np.array_equal(edge[0], last_vertex):
            return edge
        if np.array_equal(edge[1], last_vertex):
            return (edge[1], edge[0])
    return None


def direct_edges_so_right_points_inside(edges, boxes):
    """Flip edges so that the right side next to the edge is inside a box."""
    width, height = compute_union_dimensions(boxes)
    scale = min(width, height) / 1000

    directed_edges = []
    for edge in edges:
        midpoint = (edge[0] + edge[1]) / 2

        start, end = edge
        diff0 = end[0] - start[0]
        diff1 = end[1] - start[1]
        normal = np.array([diff1, -diff0])
        normal /= np.linalg.norm(normal)
        scaled_normal = normal * scale
        right_point = (midpoint[0] + scaled_normal[0], midpoint[1] + scaled_normal[1])

        if is_inside_any_box(boxes, right_point):
            directed_edges.append(edge)
        else:
            directed_edges.append((edge[1], edge[0]))

    return directed_edges


def order_edges_by_connectivity(edges, boxes):
    """Order edges by greedy connectivity starting from an arbitrary edge."""
    if not edges:
        return []

    edges = [(tuple(edge[0]), tuple(edge[1])) if isinstance(edge[0], np.ndarray) else edge for edge in edges]

    edges.sort(key=lambda edge: edge[0][1], reverse=True)

    intersection_vertices = compute_intersection_points(boxes)
    if edges[0][0] in intersection_vertices:
        edges = edges[1:] + [edges[0]]

    ordered_edges = [edges[0]]
    remaining_edges = list(edges[1:])

    while remaining_edges:
        next_edge = find_next_edge(remaining_edges, ordered_edges[-1])
        if next_edge:
            ordered_edges.append(next_edge)
            remaining_edges.remove(
                next_edge if next_edge in remaining_edges else (tuple(next_edge[1]), tuple(next_edge[0]))
            )
        else:
            break

    return ordered_edges


def extract_ordered_vertices(ordered_edges):
    """Extract the ordered vertices from the list of ordered edges."""
    ordered_vertices = [ordered_edges[0][0]]
    for edge in ordered_edges:
        ordered_vertices.append(edge[1])
    ordered_vertices.pop()
    return ordered_vertices


def compute_union_dimensions(boxes):
    """Compute width and height of the union of all boxes."""
    min_x, min_y = float("inf"), float("inf")
    max_x, max_y = float("-inf"), float("-inf")

    for box in boxes:
        x_center, y_center, _ = box["transform"][:3, 3]
        half_size_x, half_size_y = box["size"][0] / 2, box["size"][1] / 2

        box_min_x = x_center - half_size_x
        box_max_x = x_center + half_size_x
        box_min_y = y_center - half_size_y
        box_max_y = y_center + half_size_y

        min_x = min(min_x, box_min_x)
        max_x = max(max_x, box_max_x)
        min_y = min(min_y, box_min_y)
        max_y = max(max_y, box_max_y)

    width = max_x - min_x
    height = max_y - min_y

    return width, height


def compute_com_from_uniform_density(boxes):
    """Compute center of mass assuming uniform density."""
    sum_weighted_x = 0
    sum_weighted_y = 0
    total_area = 0

    for box in boxes:
        size = box["size"]
        transform = box["transform"]

        width, height = size[0], size[1]
        area = width * height

        x_centroid = transform[0, 3]
        y_centroid = transform[1, 3]

        sum_weighted_x += x_centroid * area
        sum_weighted_y += y_centroid * area
        total_area += area

    center_of_mass_x = sum_weighted_x / total_area
    center_of_mass_y = sum_weighted_y / total_area

    return center_of_mass_x, center_of_mass_y


def offset_boxes(boxes, offset):
    """Offset all boxes by the given [x, y] offset."""
    for box in boxes:
        box["transform"][0, 3] += offset[0]
        box["transform"][1, 3] += offset[1]
    return boxes


@dataclass
class ArbitraryShape2D(CollisionGeometry):
    """
    A 2D collision geometry represented as the union of axis-aligned boxes.

    Used for simulation where the actual geometry is defined in an SDF file.
    This class provides geometric properties (vertices, faces, dimensions) needed
    for workspace sampling and visualization.
    """

    def __init__(self, arbitrary_shape_pickle_path: str, com: Optional[np.ndarray] = None):
        """
        Args:
            arbitrary_shape_pickle_path: Path to pickle file containing box primitives
            com: Center of mass [x, y]. If None, computed from uniform density.
        """
        assert arbitrary_shape_pickle_path is not None and arbitrary_shape_pickle_path != ""
        self.arbitrary_shape_pickle_path = arbitrary_shape_pickle_path
        self.com = com

    @property
    def collision_geometry_names(self) -> List[str]:
        """Drake collision geometry names that must match the SDF file."""
        return [
            "arbitrary_shape::arbitrary_shape_bottom_collision",
            "arbitrary_shape::arbitrary_shape_top_collision",
        ]

    @classmethod
    def from_drake(cls, drake_shape: DrakeShape):
        raise NotImplementedError()

    @cached_property
    def com_offset(self) -> npt.NDArray[np.float64]:
        """Center of mass offset used to recenter the shape."""
        boxes = load_primitive_info(self.arbitrary_shape_pickle_path)
        primitive_types = [box["name"] for box in boxes]
        assert np.all([t == "box" for t in primitive_types]), f"Only boxes are supported. Got: {primitive_types}"
        if self.com is not None:
            return np.array([self.com[0], self.com[1]]).reshape((2, 1))
        # logging.warning("COM not provided. Computing from uniform density.")
        x_com, y_com = compute_com_from_uniform_density(boxes)
        return np.array([x_com, y_com]).reshape((2, 1))

    @cached_property
    def primitive_boxes(self) -> dict:
        """
        The primitive boxes whose union represents the shape.
        Boxes are recentered around the center of mass.
        """
        boxes = load_primitive_info(self.arbitrary_shape_pickle_path)
        primitive_types = [box["name"] for box in boxes]
        assert np.all([t == "box" for t in primitive_types]), f"Only boxes are supported. Got: {primitive_types}"

        print(f"COM offset: {self.com_offset.flatten()}")
        x_com, y_com = self.com_offset.flatten()
        boxes = offset_boxes(boxes, [-x_com, -y_com])

        return boxes

    @cached_property
    def ordered_edges(self) -> List[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
        """Ordered list of edges forming the outer boundary."""
        vertices = compute_outer_vertices(self.primitive_boxes)
        edges = compute_outer_edges(vertices, self.primitive_boxes)
        directed_edges = direct_edges_so_right_points_inside(edges, self.primitive_boxes)
        ordered_edges = order_edges_by_connectivity(directed_edges, self.primitive_boxes)
        return ordered_edges

    @cached_property
    def vertices(self) -> List[npt.NDArray[np.float64]]:
        """Ordered vertices of the outer boundary polygon."""
        ordered_vertices = extract_ordered_vertices(self.ordered_edges)
        vertices_np = [np.array(v).reshape((2, 1)) for v in ordered_vertices]
        return vertices_np

    @cached_property
    def faces(self) -> List[Hyperplane]:
        """Hyperplanes representing each edge (face) of the boundary."""
        edges_np = [np.array(edge) for edge in self.ordered_edges]
        hyperplanes = [construct_2d_plane_from_points(*edge) for edge in edges_np]
        return hyperplanes

    @cached_property
    def width(self) -> float:
        """Width of the bounding box."""
        width, _ = compute_union_dimensions(self.primitive_boxes)
        return width

    @cached_property
    def height(self) -> float:
        """Height of the bounding box."""
        _, height = compute_union_dimensions(self.primitive_boxes)
        return height

    @property
    def vertices_for_plotting(self) -> npt.NDArray[np.float64]:
        """Vertices formatted for plotting (2 x N array)."""
        vertices = np.hstack([self.vertices[idx] for idx in range(len(self.vertices))])
        return vertices

    def get_as_boxes(self, z_value: float = 0.0) -> Tuple[List[Box2d], List[RigidTransform]]:
        """
        Get the primitive boxes and their transforms for Drake simulation.

        Args:
            z_value: Z-coordinate for the boxes (planar objects lie in XY plane)

        Returns:
            Tuple of (list of Box2d objects, list of RigidTransforms)
        """
        boxes_2d = []
        transforms = []
        for box in self.primitive_boxes:
            size = box["size"]
            transform = box["transform"]
            transform[:2, 3] -= self.com_offset.flatten()
            transform[2, 3] = z_value
            boxes_2d.append(Box2d(size[0], size[1]))
            transforms.append(RigidTransform(transform))
        return boxes_2d, transforms
