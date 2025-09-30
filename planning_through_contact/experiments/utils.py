import os
from datetime import datetime
from typing import Optional

import numpy as np

# Removed ablation import - not needed
# Removed planning utils import - not needed for collision checking
from planning_through_contact.geometry.collision_geometry.arbitrary_shape_2d import (
    ArbitraryShape2D,
)
from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.t_pusher_2d import TPusher2d
from planning_through_contact.geometry.rigid_body import RigidBody


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


# Removed get_default_contact_cost

# Removed get_default_non_collision_cost

# Removed get_hardware_contact_cost

# Removed get_hardware_non_collision_cost


# Removed get_default_plan_config - it's only used for collision checking, so simplifying the logic


# Removed get_default_solver_params - not needed for collision checking


# Removed planning functions - not needed for collision checking


# Removed run_ablation_with_default_config - not needed for collision checking


# Removed baseline comparison functions - not needed for collision checking
