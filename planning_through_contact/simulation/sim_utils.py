import copy
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from lxml import etree
from pydrake.all import Box as DrakeBox
from pydrake.all import (
    ContactModel,
    DiscreteContactApproximation,
    GeometryInstance,
    LoadModelDirectives,
    MakePhongIllustrationProperties,
    ModelInstanceIndex,
    MultibodyPlant,
    Parser,
    ProcessModelDirectives,
    Rgba,
    RigidTransform,
    RollPitchYaw,
    Transform,
)
from pydrake.all import RigidBody as DrakeRigidBody

from planning_through_contact.geometry.collision_checker import CollisionChecker
from planning_through_contact.geometry.collision_geometry.arbitrary_shape_2d import (
    ArbitraryShape2D,
    create_processed_mesh_primitive_sdf_file,
    load_primitive_info,
)
from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.collision_geometry import CollisionGeometry
from planning_through_contact.geometry.collision_geometry.t_pusher_2d import TPusher2d
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.simulation.controllers.robot_system_base import (
    RobotSystemBase,
)
from planning_through_contact.utils import locked_open
from planning_through_contact.visualize.colors import COLORS

package_xml_file = os.path.join(os.path.dirname(__file__), "models/package.xml")
models_folder = os.path.join(os.path.dirname(__file__), "models")


@dataclass
class BoxWorkspace:
    """Just used for defining valid workspace for slider initialization."""

    width: float = 0.5
    height: float = 0.5
    center: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([0.0, 0.0]))
    buffer: float = 0.0

    @property
    def x_min(self) -> float:
        return self.center[0] - self.width / 2 - self.buffer

    @property
    def x_max(self) -> float:
        return self.center[0] + self.width / 2 + self.buffer

    @property
    def y_min(self) -> float:
        return self.center[1] - self.height / 2 - self.buffer

    @property
    def y_max(self) -> float:
        return self.center[1] + self.height / 2 + self.buffer

    @property
    def bounds(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        lb = np.array([self.x_min, self.y_min], dtype=np.float64)
        ub = np.array([self.x_max, self.y_max], dtype=np.float64)
        return lb, ub

    def new_workspace_with_buffer(self, new_buffer: float) -> "BoxWorkspace":
        return BoxWorkspace(self.width, self.height, self.center, new_buffer)


@dataclass
class PlanarPushingWorkspace:
    """Just used for defining valid workspace for slider initialization."""

    slider: BoxWorkspace = field(
        default_factory=lambda: BoxWorkspace(width=1.0, height=1.0, center=np.array([0.0, 0.0]), buffer=0.0)
    )


def clamp(val, min_val, max_val):
    return max(min(val, max_val), min_val)


def GetParser(plant: MultibodyPlant) -> Parser:
    """Creates a parser for a plant and adds package paths to it."""
    parser = Parser(plant)
    ConfigureParser(parser)
    return parser


def ConfigureParser(parser):
    """Add the manipulation/package.xml index to the given Parser."""
    parser.package_map().AddPackageXml(filename=package_xml_file)
    AddPackagePaths(parser)


def AddPackagePaths(parser):
    parser.package_map().PopulateFromFolder(str(models_folder))


def LoadRobotOnly(sim_config, robot_plant_file) -> MultibodyPlant:
    robot = MultibodyPlant(sim_config.time_step)
    parser = GetParser(robot)
    # Load the controller plant, i.e. the plant without the box
    directives = LoadModelDirectives(f"{models_folder}/{robot_plant_file}")
    ProcessModelDirectives(directives, robot, parser)  # type: ignore
    robot.Finalize()
    return robot


def _slider_base_filename(collision_geometry) -> str:
    """Return 'box_hydroelastic', 't_pusher', or 'arbitrary_shape' based on slider geometry."""
    if isinstance(collision_geometry, Box2d):
        return "box_hydroelastic"
    elif isinstance(collision_geometry, TPusher2d):
        return "t_pusher"
    elif isinstance(collision_geometry, ArbitraryShape2D):
        return "arbitrary_shape"
    else:
        raise NotImplementedError()


def GetSliderUrl(sim_config):
    """Return package path to slider SDF/YAML file."""
    base = _slider_base_filename(sim_config.slider.geometry)
    return f"package://planning_through_contact/{base}.sdf"


def get_slider_sdf_path(sim_config=None, collision_geometry=None) -> str:
    """Return relative file path to slider SDF file. Accepts either sim_config or collision_geometry."""
    if sim_config is not None:
        collision_geometry = sim_config.slider.geometry
    base = _slider_base_filename(collision_geometry)
    return f"{models_folder}/{base}.sdf"


def create_arbitrary_shape_sdf_file(cfg, physical_properties, collision_geometry):
    """
    Creates an SDF file for an arbitrary shape based on the pickle file.
    """
    sdf_path = get_slider_sdf_path(collision_geometry=collision_geometry)

    translation = -np.concatenate(
        [collision_geometry.com_offset.flatten(), [0]]
    )  # Plan assumes that object frame = CoM frame

    primitive_info = load_primitive_info(cfg.arbitrary_shape_pickle_path)
    create_processed_mesh_primitive_sdf_file(
        primitive_info=primitive_info,
        visual_mesh_file_path=cfg.arbitrary_shape_visual_mesh_path,
        physical_properties=physical_properties,
        global_translation=translation,
        output_file_path=sdf_path,
        model_name="arbitrary",
        base_link_name="arbitrary",
        is_hydroelastic="hydroelastic" in cfg.contact_model.lower(),
        rgba=cfg.arbitrary_shape_rgba,
        com_override=[0.0, 0.0, 0.0],  # Plan assumes that object frame = CoM frame
    )


def _set_drake_friction(root, mu_dynamic: float, mu_static: float):
    """
    Common helper to set friction values in any Drake XML (URDF/SDF).
    Mutates all drake:mu_* tags inside `root`.
    """
    for elem in root.xpath("//*[local-name()='mu_dynamic']"):
        elem.set("value", str(mu_dynamic))
    for elem in root.xpath("//*[local-name()='mu_static']"):
        elem.set("value", str(mu_static))


def configure_table_and_slider_friction(
    collision_geometry: CollisionGeometry,
    mu_dynamic: float = 0.5,
    mu_static: float = 0.5,
    table_urdf: str = "small_table_hydroelastic.urdf",
) -> None:
    """
    Modify table URDF to use specified friction coefficients.

    Args:
        mu_dynamic: Dynamic coefficient of friction to apply to table
        mu_static: Static coefficient of friction to apply to table
        table_urdf: Table's URDF filename
    """
    # Update Table friction
    base_urdf = f"{models_folder}/{table_urdf}"
    parser = etree.XMLParser(recover=True)
    tree = etree.parse(base_urdf, parser)
    root = tree.getroot()
    _set_drake_friction(root, mu_dynamic, mu_static)
    with locked_open(base_urdf, "wb") as fh:
        tree.write(fh, pretty_print=True, xml_declaration=True, encoding="UTF-8")

    # Update Slider friction
    slider_sdf_path = get_slider_sdf_path(collision_geometry=collision_geometry)
    tree = etree.parse(slider_sdf_path, parser)
    root = tree.getroot()
    _set_drake_friction(root, mu_dynamic, mu_static)
    with locked_open(slider_sdf_path, "wb") as fh:
        tree.write(fh, pretty_print=True, xml_declaration=True, encoding="UTF-8")


def AddSliderAndConfigureContact(sim_config, plant, scene_graph) -> ModelInstanceIndex:
    parser = Parser(plant, scene_graph)
    ConfigureParser(parser)

    if sim_config.contact_model != ContactModel.kHydroelastic:
        raise NotImplementedError()

    directives = LoadModelDirectives(f"{models_folder}/{sim_config.scene_directive_name}")
    ProcessModelDirectives(directives, plant, parser)  # type: ignore

    slider_sdf_url = GetSliderUrl(sim_config)
    (slider,) = parser.AddModels(url=slider_sdf_url)

    plant.set_contact_model(ContactModel.kHydroelastic)
    plant.set_discrete_contact_approximation(DiscreteContactApproximation.kLagged)
    plant.Finalize()

    return slider


def get_randomized_slider_sdf_string(sim_config, default_color=[0.1, 0.1, 0.1], color_range=0.02) -> str:
    """
    Get string for SDF file containing slider with randomized color.
    """
    sdf_file = get_slider_sdf_path(sim_config)
    tree = etree.parse(sdf_file, etree.XMLParser(recover=True))
    root = tree.getroot()

    slider_color = random_rgba_from_color_range(default_color, color_range)
    new_diffuse_value = f"{slider_color.r()} {slider_color.g()} {slider_color.b()} {slider_color.a()}"

    for diffuse in root.xpath("//model/link/visual/material/diffuse"):
        diffuse.text = new_diffuse_value

    return etree.tostring(tree, encoding="utf8").decode()


def AddRandomizedSliderAndConfigureContact(
    sim_config,
    plant,
    scene_graph,
    default_color=[0.1, 0.1, 0.1],
    color_range=0.02,
) -> ModelInstanceIndex:
    parser = Parser(plant, scene_graph)
    ConfigureParser(parser)

    if sim_config.contact_model != ContactModel.kHydroelastic:
        raise NotImplementedError()

    directives = LoadModelDirectives(f"{models_folder}/{sim_config.scene_directive_name}")
    ProcessModelDirectives(directives, plant, parser)  # type: ignore

    sdf_as_string = get_randomized_slider_sdf_string(sim_config, default_color, color_range)
    (slider,) = parser.AddModelsFromString(sdf_as_string, "sdf")

    plant.set_contact_model(ContactModel.kHydroelastic)
    plant.set_discrete_contact_approximation(DiscreteContactApproximation.kLagged)
    plant.Finalize()

    return slider


## Domain Randomization Functions


def randomize_table(
    default_color=[0.7, 0.7, 0.7],
    color_range=0.02,
    table_urdf: str = "small_table_hydroelastic.urdf",
    texture_randomization_ratio: float = 0.0,
    randomize_friction=False,
) -> None:
    """
    Randomize table appearance and friction (if randomize_friction = True).

    NOTE: randomize_friction is not implemented.
    """
    base_urdf = f"{models_folder}/{table_urdf}"
    parser = etree.XMLParser(recover=True)
    tree = etree.parse(base_urdf, parser)
    root = tree.getroot()

    rv = np.random.uniform(0, 1)
    import random

    if rv < texture_randomization_ratio:
        image_dir = f"{models_folder}/images"
        image_files = os.listdir(image_dir)
        image_file = random.choice(image_files)
        material = root.xpath('//link[@name="TableTop"]/visual/material')
        material[0].set("name", "")
        texture = etree.SubElement(material[0], "texture")
        texture.set("filename", f"{models_folder}/images/{image_file}")
    else:
        table_color = random_rgba_from_color_range(default_color, color_range)
        new_color_value = f"{table_color.r()} {table_color.g()} {table_color.b()} {table_color.a()}"
        models = root.xpath('//material[@name="LightGrey"]')
        for model in models:
            for color in model:
                color.set("rgba", new_color_value)
    # else:
    #     image_dir = f'{models_folder}/images'
    #     image_files = os.listdir(image_dir)
    #     material = root.xpath('//link[@name="TableTop"]/visual/material')
    #     material[0].set("name", "")
    #     texture = etree.SubElement(material[0], "texture")
    #     texture.set("filename", f'{models_folder}/{image_file}')
    #     # new_urdf_location = f'{models_folder}/small_table_hydroelastic_randomized.urdf'

    # Overwrite original URDF in-place so downstream YAML paths remain valid
    with locked_open(base_urdf, "wb") as fh:
        tree.write(fh, pretty_print=True, xml_declaration=True, encoding="UTF-8")


def randomize_pusher(
    default_color=[1.0, 0.345, 0.1],
    color_range=0.02,
    pusher_sdf: str = "pusher_floating_hydroelastic.sdf",
) -> None:
    """
    Randomize pusher appearance.
    """
    base_sdf = f"{models_folder}/{pusher_sdf}"

    safe_parse = etree.XMLParser(recover=True)
    tree = etree.parse(base_sdf, safe_parse)
    root = tree.getroot()

    diffuse_elements = root.xpath("//model/link/visual/material/diffuse")

    pusher_color = random_rgba_from_color_range(default_color, color_range)

    new_diffuse_value = f"{pusher_color.r()} {pusher_color.g()} {pusher_color.b()} {pusher_color.a()}"
    for diffuse in diffuse_elements:
        diffuse.text = new_diffuse_value

    # Overwrite original SDF so downstream paths remain valid
    with locked_open(base_sdf, "wb") as fh:
        tree.write(fh, pretty_print=True, xml_declaration=True, encoding="UTF-8")


def randomize_camera_config(camera_config, translation_limit=0.01, rot_limit_deg=1.0, arbitrary_background=False):
    # Randomize camera location
    new_camera_config = copy.deepcopy(camera_config)
    camera_pose = camera_config.X_PB.GetDeterministicValue()

    new_xyz = camera_pose.translation() + np.random.uniform(-translation_limit, translation_limit, 3)
    rpy = camera_pose.rotation().ToRollPitchYaw()
    rot_limit_rad = rot_limit_deg * np.pi / 180
    new_rpy = RollPitchYaw(
        rpy.roll_angle() + np.random.uniform(-rot_limit_rad, rot_limit_rad),
        rpy.pitch_angle() + np.random.uniform(-rot_limit_rad, rot_limit_rad),
        rpy.yaw_angle() + np.random.uniform(-rot_limit_rad, rot_limit_rad),
    )
    new_camera_config.X_PB = Transform(RigidTransform(new_rpy, new_xyz))

    # randomize the background color
    if arbitrary_background:
        new_rgb = np.random.uniform(0, 1, 3)
        new_camera_config.background = Rgba(new_rgb[0], new_rgb[1], new_rgb[2], 1)
    else:
        new_camera_config.background = random_rgba_from_color_range(camera_config.background, 0.05)

    return new_camera_config


def random_rgba_from_color_range(base_color, color_range):
    if color_range >= np.sqrt(3):
        r = np.random.uniform(0, 1)
        g = np.random.uniform(0, 1)
        b = np.random.uniform(0, 1)
        return Rgba(r, g, b, 1)

    if isinstance(base_color, Rgba):
        r = base_color.r()
        g = base_color.g()
        b = base_color.b()
    else:
        r = base_color[0]
        g = base_color[1]
        b = base_color[2]

    # Sample colors until valid RGB
    while True:
        # Sample random direction and offset
        direction = np.random.randn(3)
        offset = np.random.uniform(0, color_range) * direction / np.linalg.norm(direction)
        R = r + offset[0]
        G = g + offset[1]
        B = b + offset[2]
        if _valid_rgb(R, G, B):
            return Rgba(R, G, B, 1)


def random_rgba_from_color_range_legacy(base_color, color_range):
    if isinstance(base_color, Rgba):
        r = base_color.r()
        g = base_color.g()
        b = base_color.b()
    else:
        r = base_color[0]
        g = base_color[1]
        b = base_color[2]
    R = clamp(r + np.random.uniform(-color_range, color_range), 0.0, 1.0)
    G = clamp(g + np.random.uniform(-color_range, color_range), 0.0, 1.0)
    B = clamp(b + np.random.uniform(-color_range, color_range), 0.0, 1.0)
    A = 1  # assuming fully opaque
    return Rgba(R, G, B, A)


def random_rgba_euclidean_distance(base_color, min_dist, max_dist):
    if isinstance(base_color, Rgba):
        r = base_color.r()
        g = base_color.g()
        b = base_color.b()
    else:
        r = base_color[0]
        g = base_color[1]
        b = base_color[2]

    # Sample colors until valid RGB
    while True:
        # Sample random direction and offset
        direction = np.random.randn(3)
        offset = np.random.uniform(min_dist, max_dist) * direction / np.linalg.norm(direction)
        R = r + offset[0]
        G = g + offset[1]
        B = b + offset[2]
        if _valid_rgb(R, G, B):
            return Rgba(R, G, B, 1)


def _valid_rgb(r, g, b):
    return 0 <= r <= 1 and 0 <= g <= 1 and 0 <= b <= 1


## Computing collision-free, in-workspaceinitial slider poses


def get_slider_initial_pose_within_workspace(
    workspace: "PlanarPushingWorkspace",
    slider: "RigidBody",
    pusher_pose: "PlanarPose",
    collision_checker: "CollisionChecker",
    limit_rotations: bool = False,
    rotation_limit: float = None,
    timeout_s: float = 10.0,
    rng: Optional[np.random.Generator] = None,
) -> Optional["PlanarPose"]:
    """
    Generates a random valid pose for a slider object that avoids collisions with the pusher
    and satisfies workspace constraints. Uses Monte Carlo sampling with timeout protection.

    Determinism: pass a dedicated np.random.Generator in `rng`. If you derive `rng` from a
    master seed and a trial index (see example usage below), you get a fixed pose per trial.
    """
    if rng is None:
        # Falls back to a non-deterministic Generator
        rng = np.random.default_rng()

    start_time = time.time()
    EPS = 1e-2

    while True:
        if time.time() - start_time > timeout_s:
            raise ValueError("Could not find a valid slider pose within the timeout.")

        x_initial = rng.uniform(workspace.slider.x_min, workspace.slider.x_max)
        y_initial = rng.uniform(workspace.slider.y_min, workspace.slider.y_max)

        if limit_rotations:
            if rotation_limit is not None:
                th_initial = rng.uniform(-rotation_limit, rotation_limit)
            else:
                th_initial = rng.uniform(-np.pi / 2 + EPS, np.pi / 2 - EPS)
        else:
            th_initial = rng.uniform(-np.pi + EPS, np.pi - EPS)

        slider_pose = PlanarPose(x_initial, y_initial, th_initial)

        collides_with_pusher = collision_checker.check_collision(slider_pose, pusher_pose)  # Z-height doesn't matter
        # collision_checker.visualize_once(slider_pose, pusher_pose)  # debug visualization
        within_workspace = slider_within_workspace(workspace, slider_pose, slider)

        if within_workspace and not collides_with_pusher:
            return slider_pose
        # else:
        #     print(f"Collision: {collides_with_pusher}, Within workspace: {within_workspace}")


def slider_within_workspace(workspace: PlanarPushingWorkspace, pose: PlanarPose, slider: RigidBody) -> bool:
    """
    Checks whether the entire slider is within the workspace
    """
    R_WB = pose.two_d_rot_matrix()
    p_WB = pose.pos()

    slider_collision_geometry = slider.geometry
    p_Wv_s = [
        slider_collision_geometry.get_p_Wv_i(vertex_idx, R_WB, p_WB).flatten()
        for vertex_idx in range(len(slider_collision_geometry.vertices))
    ]

    lb, ub = workspace.slider.bounds
    vertices_within_workspace: bool = np.all([v <= ub for v in p_Wv_s]) and np.all([v >= lb for v in p_Wv_s])
    return vertices_within_workspace


## Meshcat visualizations


def get_slider_body(robot_system: RobotSystemBase) -> DrakeRigidBody:
    slider_body = robot_system.station_plant.GetUniqueFreeBaseBodyOrThrow(robot_system.slider)
    return slider_body


def get_slider_shapes(robot_system: RobotSystemBase) -> List[DrakeBox]:
    slider_body = get_slider_body(robot_system)
    collision_geometries_ids = robot_system.station_plant.GetCollisionGeometriesForBody(slider_body)

    inspector = robot_system._scene_graph.model_inspector()
    shapes = [inspector.GetShape(id) for id in collision_geometries_ids]

    # for now we only support Box shapes
    assert all([isinstance(shape, DrakeBox) for shape in shapes])

    return shapes


def get_slider_shape_poses(robot_system: RobotSystemBase) -> List[DrakeBox]:
    slider_body = get_slider_body(robot_system)
    collision_geometries_ids = robot_system.station_plant.GetCollisionGeometriesForBody(slider_body)

    inspector = robot_system._scene_graph.model_inspector()
    poses = [inspector.GetPoseInFrame(id) for id in collision_geometries_ids]

    return poses


def create_goal_geometries(
    robot_system: RobotSystemBase,
    desired_planar_pose: PlanarPose,
    box_color=COLORS["emeraldgreen"],
    desired_pose_alpha=0.3,
) -> List[str]:
    shapes = get_slider_shapes(robot_system)
    poses = get_slider_shape_poses(robot_system)
    heights = [shape.height() for shape in shapes]
    min_height = min(heights)
    desired_pose = desired_planar_pose.to_pose(min_height / 2, z_axis_is_positive=True)

    source_id = robot_system._scene_graph.RegisterSource()

    goal_geometries = []
    for idx, (shape, pose) in enumerate(zip(shapes, poses)):
        geom_instance = GeometryInstance(
            desired_pose.multiply(pose),
            shape,
            f"shape_{idx}",
        )
        curr_shape_geometry_id = robot_system._scene_graph.RegisterAnchoredGeometry(
            source_id,
            geom_instance,
        )
        robot_system._scene_graph.AssignRole(
            source_id,
            curr_shape_geometry_id,
            MakePhongIllustrationProperties(box_color.diffuse(desired_pose_alpha)),
        )
        geom_name = f"goal_shape_{idx}"
        goal_geometries.append(geom_name)
        robot_system._meshcat.SetObject(geom_name, shape, rgba=Rgba(*box_color.diffuse(desired_pose_alpha)))
    return goal_geometries


def visualize_desired_slider_pose(
    robot_system: RobotSystemBase,
    desired_planar_pose: PlanarPose,
    goal_geometries: List[str],
    time_in_recording: float = 0.0,
) -> None:
    shapes = get_slider_shapes(robot_system)
    poses = get_slider_shape_poses(robot_system)

    heights = [shape.height() for shape in shapes]
    min_height = min(heights)
    desired_pose = desired_planar_pose.to_pose(min_height / 2, z_axis_is_positive=True)

    for pose, geom_name in zip(poses, goal_geometries):
        robot_system._meshcat.SetTransform(geom_name, desired_pose.multiply(pose), time_in_recording)
