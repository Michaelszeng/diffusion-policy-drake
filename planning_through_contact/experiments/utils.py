import logging
import os
from datetime import datetime
from typing import Literal, Optional

import numpy as np

# Removed ablation import - not needed
# Removed planning utils import - not needed for collision checking
from planning_through_contact.geometry.collision_geometry.arbitrary_shape_2d import (
    ArbitraryShape2D,
)
from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.t_pusher_2d import TPusher2d
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.planar.planar_plan_config import (
    ContactConfig,
    ContactCost,
    NonCollisionCost,
    PlanarPlanConfig,
    PlanarPushingWorkspace,
    SliderPusherSystemConfig,
)
from planning_through_contact.tools.utils import PhysicalProperties


def create_output_folder(output_dir: str, slider_type: str, traj_number: Optional[int]) -> str:
    os.makedirs(output_dir, exist_ok=True)
    folder_name = f"{output_dir}/run_{get_time_as_str()}_{slider_type}"
    if traj_number is not None:
        folder_name += f"_traj_{traj_number}"
    os.makedirs(folder_name, exist_ok=True)

    return folder_name


def get_time_as_str() -> str:
    current_time = datetime.now()
    # For example, YYYYMMDDHHMMSS format
    formatted_time = current_time.strftime("%Y%m%d%H%M%S")
    return formatted_time


def get_box(mass) -> RigidBody:
    box_geometry = Box2d(width=0.1, height=0.1)
    slider = RigidBody("box", box_geometry, mass)
    return slider


def get_tee(mass) -> RigidBody:
    body = RigidBody("t_pusher", TPusher2d(), mass)
    return body


def get_arbitrary(arbitrary_shape_pickle_path: str, mass: float, com: np.ndarray = None) -> RigidBody:
    "com assumes uniform density if None."
    body = RigidBody("arbitrary", ArbitraryShape2D(arbitrary_shape_pickle_path, com), mass)
    return body


def get_sugar_box() -> RigidBody:
    mass = 0.1
    box_geometry = Box2d(width=0.106, height=0.185)
    slider = RigidBody("sugar_box", box_geometry, mass)
    return slider


def get_default_contact_cost() -> ContactCost:
    contact_cost = ContactCost(
        keypoint_arc_length=10.0,
        # NOTE: This is multiplied by 1e-4 because we have forces in other units in the optimization problem
        force_regularization=100000.0,
        keypoint_velocity_regularization=100.0,
        ang_velocity_regularization=None,
        trace=None,
        mode_transition_cost=None,
        time=1.0,
    )
    return contact_cost


def get_default_non_collision_cost() -> NonCollisionCost:
    non_collision_cost = NonCollisionCost(
        distance_to_object_socp=0.1,
        pusher_velocity_regularization=10.0,
        pusher_arc_length=10.0,
        time=None,
    )
    return non_collision_cost


def get_hardware_contact_cost() -> ContactCost:
    """
    A custom cost for hardware,
    which empically generates plans that respect robot velocity
    limits etc.
    """
    contact_cost = ContactCost(
        keypoint_arc_length=10.0,
        force_regularization=100000.0,
        keypoint_velocity_regularization=100.0,
        trace=None,
        mode_transition_cost=None,
        time=1.0,
    )
    return contact_cost


def get_hardware_non_collision_cost() -> NonCollisionCost:
    non_collision_cost = NonCollisionCost(
        distance_to_object_socp=0.25,
        pusher_velocity_regularization=10.0,
        pusher_arc_length=5.0,
        time=None,
    )
    return non_collision_cost


def get_default_plan_config(
    slider_type: Literal["box", "sugar_box", "tee", "arbitrary"] = "box",
    arbitrary_shape_pickle_path: str = "",
    slider_physical_properties: PhysicalProperties = None,
    pusher_radius: float = 0.015,
    time_contact: float = 2.0,
    time_non_collision: float = 4.0,
    workspace: Optional[PlanarPushingWorkspace] = None,
    hardware: bool = False,
) -> PlanarPlanConfig:
    mass = 0.1 if slider_physical_properties is None else slider_physical_properties.mass
    com = None if slider_physical_properties is None else slider_physical_properties.center_of_mass
    if slider_physical_properties is None:
        logging.warning("Using default mass of 0.1 kg for the slider.")
    if slider_type == "box":
        slider = get_box(mass)
    elif slider_type == "sugar_box":
        slider = get_sugar_box(mass)
    elif slider_type == "tee":
        slider = get_tee(mass)
    elif slider_type == "arbitrary":
        slider = get_arbitrary(arbitrary_shape_pickle_path, mass, com)
    else:
        raise NotImplementedError(f"Slider type {slider_type} not supported")

    if hardware:
        slider_pusher_config = SliderPusherSystemConfig(
            slider=slider,
            pusher_radius=pusher_radius,
            friction_coeff_slider_pusher=0.05,
            friction_coeff_table_slider=0.5,
            integration_constant=0.3,
        )

        contact_cost = get_hardware_contact_cost()
        non_collision_cost = get_hardware_non_collision_cost()
        lam_buffer = 0.25
        contact_config = ContactConfig(cost=contact_cost, lam_min=lam_buffer, lam_max=1 - lam_buffer)
    else:
        slider_pusher_config = SliderPusherSystemConfig(
            slider=slider,
            pusher_radius=pusher_radius,
            friction_coeff_slider_pusher=0.1,
            friction_coeff_table_slider=0.5,
            integration_constant=0.3,
        )
        # Simplified config - removed optimization-based cost functions
        contact_config = ContactConfig()
        non_collision_cost = NonCollisionCost()

    # Simplified config - removed optimization-based parameters
    plan_cfg = PlanarPlanConfig(
        dynamics_config=slider_pusher_config,
        contact_config=contact_config,
        non_collision_cost=non_collision_cost,
        workspace=workspace,
    )

    return plan_cfg


# Removed get_default_solver_params - not needed for collision checking


# Removed planning functions - not needed for collision checking


# Removed run_ablation_with_default_config - not needed for collision checking


# Removed baseline comparison functions - not needed for collision checking
