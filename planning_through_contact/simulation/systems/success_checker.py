from typing import Optional, Tuple

import numpy as np
import zarr
from pydrake.all import HPolyhedron, LeafSystem, VPolytope

from planning_through_contact.geometry.planar.planar_pose import PlanarPose


def load_goal_polyhedra_from_dataset(
    dataset_path: str,
    pusher_start_pose: PlanarPose,
    slider_goal_pose: PlanarPose,
    convex_hull_scale: float = 1.0,
) -> Tuple[HPolyhedron, HPolyhedron]:
    """
    Load and construct pusher and slider goal polyhedra from a dataset.

    Args:
        dataset_path: Path to the zarr dataset containing final states
        pusher_start_pose: Starting pose of the pusher (used as center for scaling)
        slider_goal_pose: Goal pose of the slider (used as center for scaling)
        convex_hull_scale: Scale factor for the convex hulls (default: 1.0)

    Returns:
        Tuple of (pusher_goal_polyhedron, slider_goal_polyhedron)
    """
    root = zarr.open(dataset_path, mode="r")
    indices = np.array(root["meta/episode_ends"]) - 1

    # Load pusher goal polyhedron
    state = np.array(root["data/state"])
    final_positions = state[indices][:, :2]
    pusher_convex_hull = HPolyhedron(VPolytope(final_positions.transpose()))
    pusher_goal_polyhedron = pusher_convex_hull.Scale(
        scale=convex_hull_scale,
        center=pusher_start_pose.vector().flatten()[:2],
    )

    # Load slider goal polyhedron
    slider_state = np.array(root["data/slider_state"])
    final_slider_states = slider_state[indices]
    slider_convex_hull = HPolyhedron(VPolytope(final_slider_states.transpose()))
    slider_goal_polyhedron = slider_convex_hull.Scale(
        scale=convex_hull_scale,
        center=slider_goal_pose.vector().flatten(),
    )

    return pusher_goal_polyhedron, slider_goal_polyhedron


def is_contained_in_polyhedron(point: np.ndarray, polyhedron: HPolyhedron) -> bool:
    A, b = polyhedron.A(), polyhedron.b()
    return np.all(A @ point <= b)


def check_success_convex_hull(
    slider_pose: PlanarPose,
    pusher_pose: PlanarPose,
    slider_goal_polyhedron: Optional[HPolyhedron] = None,
    pusher_goal_polyhedron: Optional[HPolyhedron] = None,
    dataset_path: Optional[str] = None,
    pusher_start_pose: Optional[PlanarPose] = None,
    slider_goal_pose: Optional[PlanarPose] = None,
    convex_hull_scale: float = 1.0,
    return_separate: bool = False,
) -> bool:
    """
    Global helper function to check if slider and pusher are within convex hull regions.

    Args:
        slider_pose: Current slider pose (x, y, theta)
        pusher_pose: Current pusher pose (only x, y used)
        slider_goal_polyhedron: HPolyhedron representing acceptable slider goal region (optional)
        pusher_goal_polyhedron: HPolyhedron representing acceptable pusher goal region (optional)
        dataset_path: Path to zarr dataset (alternative to providing polyhedra directly)
        pusher_start_pose: Starting pose of pusher (needed if loading from dataset)
        slider_goal_pose: Goal pose of slider (needed if loading from dataset)
        convex_hull_scale: Scale factor for convex hulls (default: 1.0)
        return_separate: If True, return (slider_success, pusher_success) tuple instead of combined bool

    Returns:
        If return_separate=False: True if both slider and pusher are within their goal regions, False otherwise
        If return_separate=True: (slider_success, pusher_success) tuple
    """
    # If polyhedra not provided, load them from dataset (with caching)
    if slider_goal_polyhedron is None or pusher_goal_polyhedron is None:
        if dataset_path is None or pusher_start_pose is None or slider_goal_pose is None:
            raise ValueError(
                "Must provide either (slider_goal_polyhedron, pusher_goal_polyhedron) "
                "or (dataset_path, pusher_start_pose, slider_goal_pose)"
            )

        # Create cache key from parameters
        cache_key = (
            dataset_path,
            tuple(pusher_start_pose.vector().flatten()),
            tuple(slider_goal_pose.vector().flatten()),
            convex_hull_scale,
        )

        # Check cache first
        if cache_key in check_success_convex_hull._cache:
            pusher_goal_polyhedron, slider_goal_polyhedron = check_success_convex_hull._cache[cache_key]
        else:
            # Load from dataset and cache the result
            pusher_goal_polyhedron, slider_goal_polyhedron = load_goal_polyhedra_from_dataset(
                dataset_path=dataset_path,
                pusher_start_pose=pusher_start_pose,
                slider_goal_pose=slider_goal_pose,
                convex_hull_scale=convex_hull_scale,
            )
            check_success_convex_hull._cache[cache_key] = (pusher_goal_polyhedron, slider_goal_polyhedron)

    slider_success = is_contained_in_polyhedron(slider_pose.vector(), slider_goal_polyhedron)
    pusher_success = is_contained_in_polyhedron(pusher_pose.vector()[:2], pusher_goal_polyhedron)

    if return_separate:
        return slider_success, pusher_success
    return slider_success and pusher_success


# Initialize the cache as a function attribute
check_success_convex_hull._cache = {}


def check_success_tolerance(
    slider_pose: PlanarPose,
    slider_goal_pose: PlanarPose,
    pusher_pose: PlanarPose,
    pusher_goal_pose: PlanarPose,
    trans_tol: float,
    rot_tol: float,
    evaluate_final_slider_rotation: bool = True,
    evaluate_final_pusher_position: bool = True,
    return_separate: bool = False,
) -> bool:
    """
    Global helper function to check if slider and pusher meet success criteria.

    Args:
        slider_pose: Current slider pose
        slider_goal_pose: Target slider pose
        pusher_pose: Current pusher pose (only x,y used)
        pusher_goal_pose: Target pusher pose (only x,y used)
        trans_tol: Translational tolerance in meters
        rot_tol: Rotational tolerance in degrees
        evaluate_final_slider_rotation: Whether to check slider orientation
        evaluate_final_pusher_position: Whether to check pusher position
        return_separate: If True, return (slider_success, pusher_success) tuple instead of combined bool

    Returns:
        If return_separate=False: True if success criteria are met, False otherwise
        If return_separate=True: (slider_success, pusher_success) tuple
    """
    # slider
    slider_error = slider_goal_pose.vector() - slider_pose.vector()
    reached_goal_slider_position = np.linalg.norm(slider_error[:2]) <= trans_tol
    reached_goal_slider_orientation = np.abs(slider_error[2]) <= np.deg2rad(rot_tol)

    # Check slider success
    slider_success = reached_goal_slider_position
    if evaluate_final_slider_rotation:
        slider_success = slider_success and reached_goal_slider_orientation

    # pusher
    pusher_error = pusher_goal_pose.vector() - pusher_pose.vector()
    # Note: pusher goal criterion is intentionally very lenient
    # since the teleoperator (me) did a poor job as well, oops :) oops
    reached_goal_pusher_position = np.linalg.norm(pusher_error[:2]) <= 0.04

    # Check pusher success
    pusher_success = True
    if evaluate_final_pusher_position:
        pusher_success = reached_goal_pusher_position

    if return_separate:
        return slider_success, pusher_success
    return slider_success and pusher_success


class SuccessChecker(LeafSystem):
    """
    Outputs a scalar (0.0, 1.0, or 2.0) indicating the success state of slider and pusher.

    Supports two modes:
    1. "tolerance" mode: Checks if poses are within specified tolerances
    2. "convex_hull" mode: Checks if poses are within convex hull regions

    The mode is automatically determined from sim_config.multi_run_config.success_criteria.
    If convex_hull mode is specified, the polyhedra are automatically loaded from the dataset.

    Output values:
    - 0.0: Neither slider nor pusher have reached their goals
    - 1.0: Slider has reached its goal, but pusher has not
    - 2.0: Both slider and pusher have reached their goals

    Args:
        plant: MultibodyPlant[float] (finalized)
        slider_model_instance: ModelInstanceIndex of the slider
        pusher_body_index: BodyIndex of the pusher end effector body
        slider_goal_pose: Target pose for the slider
        pusher_goal_pose: Target pose for the pusher (usually start pose)
        sim_config: PlanarPushingSimConfig (optional, used to determine mode and load convex hulls)
        trans_tol: Translational tolerance in meters (tolerance mode, default: 0.01)
        rot_tol: Rotational tolerance in degrees (tolerance mode, default: 5.0)
        evaluate_final_slider_rotation: Whether to check slider orientation (tolerance mode, default: True)
        evaluate_final_pusher_position: Whether to check pusher position (tolerance mode, default: True)
    """

    def __init__(
        self,
        plant,
        slider_model_instance,
        pusher_body_index,
        slider_goal_pose: PlanarPose,
        pusher_goal_pose: PlanarPose,
        sim_config=None,
        trans_tol: float = 0.01,
        rot_tol: float = 5.0,
        evaluate_final_slider_rotation: bool = True,
        evaluate_final_pusher_position: bool = True,
    ):
        super().__init__()
        self._plant = plant
        self._context_plant = plant.CreateDefaultContext()  # scratch context for queries
        self._slider_model_instance = slider_model_instance
        self._pusher_body_index = pusher_body_index

        # Tolerance mode parameters
        self._slider_goal_pose = slider_goal_pose
        self._pusher_goal_pose = pusher_goal_pose
        self._trans_tol = float(trans_tol)
        self._rot_tol = float(rot_tol)
        self._evaluate_final_slider_rotation = evaluate_final_slider_rotation
        self._evaluate_final_pusher_position = evaluate_final_pusher_position

        # Determine mode and load convex hulls if needed
        self._mode = "tolerance"  # default
        self._slider_goal_polyhedron = None
        self._pusher_goal_polyhedron = None

        if (
            sim_config is not None
            and hasattr(sim_config, 'multi_run_config')
            and sim_config.multi_run_config is not None
            and sim_config.multi_run_config.success_criteria == "convex_hull"
        ):
            # Automatically load convex hulls from dataset
            self._mode = "convex_hull"
            self._pusher_goal_polyhedron, self._slider_goal_polyhedron = load_goal_polyhedra_from_dataset(
                dataset_path=sim_config.multi_run_config.dataset_path,
                pusher_start_pose=pusher_goal_pose,
                slider_goal_pose=slider_goal_pose,
                convex_hull_scale=sim_config.multi_run_config.convex_hull_scale,
            )

        # Input port: full plant state
        self._state_in = self.DeclareVectorInputPort("x_plant", self._plant.num_multibody_states())

        # Output port: scalar success indicator (0.0, 1.0, or 2.0)
        self.DeclareVectorOutputPort("success", 1, self._CalcSuccess)

    def _CalcSuccess(self, context, output):
        # Mirror the plant's state
        x = self._state_in.Eval(context)
        self._plant.SetPositionsAndVelocities(self._context_plant, x)

        # Get slider pose
        slider_pose_vec = self._plant.GetPositions(self._context_plant, self._slider_model_instance)
        slider_pose = PlanarPose.from_generalized_coords(slider_pose_vec)

        # Get pusher pose (only translation matters)
        pusher_body = self._plant.get_body(self._pusher_body_index)
        pusher_position = self._plant.EvalBodyPoseInWorld(self._context_plant, pusher_body).translation()
        pusher_pose = PlanarPose(pusher_position[0], pusher_position[1], 0.0)

        # Check success using the appropriate mode, get separate results
        if self._mode == "tolerance":
            slider_success, pusher_success = check_success_tolerance(
                slider_pose,
                self._slider_goal_pose,
                pusher_pose,
                self._pusher_goal_pose,
                self._trans_tol,
                self._rot_tol,
                self._evaluate_final_slider_rotation,
                self._evaluate_final_pusher_position,
                return_separate=True,
            )
        else:  # convex_hull mode
            slider_success, pusher_success = check_success_convex_hull(
                slider_pose,
                pusher_pose,
                slider_goal_polyhedron=self._slider_goal_polyhedron,
                pusher_goal_polyhedron=self._pusher_goal_polyhedron,
                return_separate=True,
            )

        # Encode success state: 0 = neither, 1 = slider only, 2 = both
        if slider_success and pusher_success:
            output_value = 2.0
        elif slider_success:
            output_value = 1.0
        else:
            output_value = 0.0

        output.SetFromVector([output_value])
