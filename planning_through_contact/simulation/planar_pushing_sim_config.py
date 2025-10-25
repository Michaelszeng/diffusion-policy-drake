from dataclasses import dataclass, field
from typing import List, Optional

import hydra
import numpy as np
from omegaconf import OmegaConf
from pydrake.all import Rgba, RollPitchYaw
from pydrake.common.schema import Transform
from pydrake.math import RigidTransform
from pydrake.multibody.plant import ContactModel
from pydrake.systems.sensors import CameraConfig

from planning_through_contact.geometry.collision_geometry.arbitrary_shape_2d import (
    ArbitraryShape2D,
)
from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.t_pusher_2d import TPusher2d
from planning_through_contact.geometry.physical_properties import PhysicalProperties
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.simulation.controllers.diffusion_policy_source import (
    DiffusionPolicyConfig,
)
from planning_through_contact.simulation.sim_utils import (
    BoxWorkspace,
    PlanarPushingWorkspace,
    configure_table_and_slider_friction,
    create_arbitrary_shape_sdf_file,
    randomize_camera_config,
)


class MultiRunConfig:
    def __init__(
        self,
        num_runs: int,
        max_attempt_duration: float,
        seed: int,
        slider_type: str,
        arbitrary_shape_pickle_path: str,
        pusher_start_pose: PlanarPose,
        slider_goal_pose: PlanarPose,
        workspace_width: float,
        workspace_height: float,
        trans_tol: float = 0.01,
        rot_tol: float = 0.01,  # degrees
        evaluate_final_pusher_position: bool = True,
        evaluate_final_slider_rotation: bool = True,
        success_criteria: str = "tolerance",
        dataset_path: str = None,
        convex_hull_scale: float = 1.0,
        slider_physical_properties: PhysicalProperties = None,
        num_trials_to_record: int = 0,
    ):
        # Define workspace for initial slider pose
        workspace = PlanarPushingWorkspace(
            slider=BoxWorkspace(
                width=workspace_width,
                height=workspace_height,
                center=np.array([slider_goal_pose.x, slider_goal_pose.y]),
                buffer=0,
            ),
        )
        self.workspace = workspace
        self.num_runs = num_runs
        self.seed = seed
        self.max_attempt_duration = max_attempt_duration
        self.trans_tol = trans_tol
        self.rot_tol = rot_tol
        self.evaluate_final_pusher_position = evaluate_final_pusher_position
        self.evaluate_final_slider_rotation = evaluate_final_slider_rotation
        self.success_criteria = success_criteria
        self.dataset_path = dataset_path
        self.convex_hull_scale = convex_hull_scale
        self.num_trials_to_record = num_trials_to_record

    def __str__(self):
        slider_pose_str = f"initial_slider_poses: {self.initial_slider_poses}"
        return f"{slider_pose_str}\nmax_attempt_duration: {self.max_attempt_duration}"

    def __eq__(self, other: "MultiRunConfig"):
        if len(self.initial_slider_poses) != len(other.initial_slider_poses):
            return False
        for i in range(len(self.initial_slider_poses)):
            if not self.initial_slider_poses[i] == other.initial_slider_poses[i]:
                return False
        return (
            self.num_runs == other.num_runs
            and self.seed == other.seed
            and self.max_attempt_duration == other.max_attempt_duration
            and self.trans_tol == other.trans_tol
            and self.rot_tol == other.rot_tol
            and self.evaluate_final_pusher_position == other.evaluate_final_pusher_position
            and self.evaluate_final_slider_rotation == other.evaluate_final_slider_rotation
            and self.success_criteria == other.success_criteria
            and self.dataset_path == other.dataset_path
            and self.num_trials_to_record == other.num_trials_to_record
            and self.convex_hull_scale == other.convex_hull_scale
        )


@dataclass
class PlanarPushingSimConfig:
    slider: RigidBody
    contact_model: ContactModel = ContactModel.kHydroelastic
    visualize_desired: bool = False
    slider_goal_pose: Optional[PlanarPose] = None
    pusher_start_pose: PlanarPose = field(default_factory=lambda: PlanarPose(x=0.0, y=0.5, theta=0.0))
    slider_start_pose: PlanarPose = field(default_factory=lambda: PlanarPose(x=0.0, y=0.5, theta=0.0))
    time_step: float = 1e-3
    draw_frames: bool = False
    use_realtime: bool = False
    delay_before_execution: float = 5.0
    save_plots: bool = False
    diffusion_policy_config: DiffusionPolicyConfig = None
    scene_directive_name: str = "planar_pushing_iiwa_plant_hydroelastic.yaml"
    use_hardware: bool = False
    joint_velocity_limit_factor: float = 1.0
    pusher_radius: float = 0.015
    pusher_z_offset: float = 0.05  # distance between table and pusher end
    camera_configs: List[CameraConfig] = None
    domain_randomization_color_range: float = 0.0
    log_dir: str = None  # directory for logging rollouts from output_feedback_table_environments
    multi_run_config: MultiRunConfig = None
    slider_physical_properties: PhysicalProperties = None
    arbitrary_shape_pickle_path: str = ""
    arbitrary_shape_rgba: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0])
    default_joint_positions: np.ndarray = None
    constant_velocity_disturbance: float = 0.0
    # PD gains for constant velocity disturbance controller
    constant_velocity_disturbance_Kp: float = 60.0
    constant_velocity_disturbance_Kd: float = 5.0
    constant_velocity_disturbance_force_cap: float = 0.85
    constant_velocity_disturbance_velocity_window_size: int = 5
    constant_velocity_disturbance_tune_mode: bool = False

    @classmethod
    def from_yaml(cls, cfg: OmegaConf):
        """Create sim_config from yaml file (i.e. gamepad_teleop_carbon.yaml)"""
        slider_physical_properties: PhysicalProperties = hydra.utils.instantiate(cfg.physical_properties)

        # Define geometry of slider
        if cfg.slider_type == "box":
            slider: RigidBody = RigidBody("box", Box2d(0.1, 0.1), slider_physical_properties.mass)
        elif cfg.slider_type == "tee":
            slider: RigidBody = RigidBody("tee", TPusher2d(), slider_physical_properties.mass)
        elif cfg.slider_type == "arbitrary":
            slider: RigidBody = RigidBody(
                "arbitrary",
                ArbitraryShape2D(cfg.arbitrary_shape_pickle_path, slider_physical_properties.center_of_mass),
                slider_physical_properties.mass,
            )
            # Generate new SDF file for arbitrary shape based on the pickle file
            create_arbitrary_shape_sdf_file(cfg, slider_physical_properties, slider.geometry)
        else:
            raise ValueError(f"Slider type not yet implemented: {cfg.slider_type}")

        configure_table_and_slider_friction(
            slider.geometry,
            mu_dynamic=slider_physical_properties.mu_dynamic,
            mu_static=slider_physical_properties.mu_static,
        )

        slider_goal_pose: PlanarPose = hydra.utils.instantiate(cfg.slider_goal_pose)
        pusher_start_pose: PlanarPose = hydra.utils.instantiate(cfg.pusher_start_pose)

        # Retrieve correct scene directive name based on the robot station type
        if (
            cfg.robot_station._target_
            == "planning_through_contact.simulation.controllers.iiwa_hardware_station.IiwaHardwareStation"
        ):
            scene_directive_name = "planar_pushing_iiwa_plant_hydroelastic.yaml"
        elif (
            cfg.robot_station._target_
            == "planning_through_contact.simulation.controllers.cylinder_actuated_station.CylinderActuatedStation"
        ):
            scene_directive_name = "planar_pushing_cylinder_plant_hydroelastic.yaml"
        else:
            raise ValueError(f"Invalid robot station type: {cfg.robot_station._target_}")

        sim_config = cls(
            slider=slider,
            contact_model=eval(cfg.contact_model),
            visualize_desired=cfg.visualize_desired,
            slider_goal_pose=slider_goal_pose,
            pusher_start_pose=pusher_start_pose,
            time_step=cfg.time_step,
            draw_frames=cfg.draw_frames,
            use_realtime=cfg.use_realtime,
            delay_before_execution=cfg.delay_before_execution,
            save_plots=cfg.save_plots,
            scene_directive_name=scene_directive_name,
            use_hardware=cfg.use_hardware,
            pusher_radius=cfg.pusher_radius,
            pusher_z_offset=cfg.pusher_z_offset,
            log_dir=cfg.log_dir,
            domain_randomization_color_range=cfg.domain_randomization_color_range,
            slider_physical_properties=slider_physical_properties,
            arbitrary_shape_pickle_path=cfg.arbitrary_shape_pickle_path,
            arbitrary_shape_rgba=np.array(cfg.arbitrary_shape_rgba),
            constant_velocity_disturbance=cfg.constant_velocity_disturbance,
            constant_velocity_disturbance_Kp=cfg.get("constant_velocity_disturbance_Kp", 60.0),
            constant_velocity_disturbance_Kd=cfg.get("constant_velocity_disturbance_Kd", 5.0),
            constant_velocity_disturbance_force_cap=cfg.get("constant_velocity_disturbance_force_cap", 0.85),
            constant_velocity_disturbance_velocity_window_size=cfg.get(
                "constant_velocity_disturbance_velocity_window_size", 5
            ),
            constant_velocity_disturbance_tune_mode=cfg.get("constant_velocity_disturbance_tune_mode", False),
        )

        # Optional fields
        if "slider_start_pose" in cfg:
            sim_config.slider_start_pose = hydra.utils.instantiate(cfg.slider_start_pose)
        if "default_joint_positions" in cfg:
            sim_config.default_joint_positions = np.array(cfg.default_joint_positions)
        if "joint_velocity_limit_factor" in cfg:
            sim_config.joint_velocity_limit_factor = cfg.joint_velocity_limit_factor
        # Override disturbance PD gains if provided
        if "constant_velocity_disturbance_Kp" in cfg:
            sim_config.constant_velocity_disturbance_Kp = cfg.constant_velocity_disturbance_Kp
        if "constant_velocity_disturbance_Kd" in cfg:
            sim_config.constant_velocity_disturbance_Kd = cfg.constant_velocity_disturbance_Kd
        if "diffusion_policy_config" in cfg:
            sim_config.diffusion_policy_config = hydra.utils.instantiate(cfg.diffusion_policy_config)
        if "camera_configs" in cfg and cfg.camera_configs:
            camera_configs = []
            camera_config_attrs = dir(CameraConfig)
            exclude_keys = ["focal_x", "focal_y"]
            for camera_config in cfg.camera_configs:
                kwargs = {}
                orientation = RollPitchYaw(
                    roll=camera_config.orientation.roll,
                    pitch=camera_config.orientation.pitch,
                    yaw=camera_config.orientation.yaw,
                )

                kwargs["X_PB"] = Transform(RigidTransform(orientation, np.array(camera_config.position)))
                if "parent_frame" in camera_config:
                    kwargs["X_PB"].base_frame = camera_config.parent_frame

                if "background" in camera_config:
                    kwargs["background"] = Rgba(
                        camera_config.background.r,
                        camera_config.background.g,
                        camera_config.background.b,
                        camera_config.background.a,
                    )

                if "focal_x" in camera_config and "focal_y" in camera_config:
                    kwargs["focal"] = CameraConfig.FocalLength(x=camera_config.focal_x, y=camera_config.focal_y)
                for key in camera_config:
                    if key not in kwargs and key not in exclude_keys and key in camera_config_attrs:
                        kwargs[key] = camera_config[key]

                # Recommended lighting config
                create_renderer = False
                if "lights" in camera_config:
                    assert "light_direction" not in camera_config
                    create_renderer = True
                # Legacy lighting config
                if "light_direction" in camera_config:
                    assert "lights" not in camera_config
                    create_renderer = True
                if "cast_shadows" in camera_config:
                    create_renderer = True

                # Create custom renderer
                if create_renderer:
                    from pydrake.geometry import LightParameter, RenderEngineVtkParams

                    renderer_params = RenderEngineVtkParams()

                    # Shadows
                    if "cast_shadows" in camera_config and camera_config.cast_shadows:
                        renderer_params.cast_shadows = True
                        renderer_params.shadow_map_size = 512

                    # Background
                    if "background" in camera_config:
                        renderer_params.default_clear_color = np.array(
                            [
                                camera_config.background.r,
                                camera_config.background.g,
                                camera_config.background.b,
                            ]
                        )

                    # Recommended lighting config
                    if "lights" in camera_config:
                        # Lights
                        lights = []
                        for light in camera_config.lights:
                            direction = np.array(light.direction)
                            direction = direction / np.linalg.norm(direction)
                            R, G, B = light.color[0], light.color[1], light.color[2]
                            color = Rgba(R, G, B, 1.0)
                            intensity = light.intensity
                            lights.append(
                                LightParameter(
                                    direction=direction,
                                    color=color,
                                    intensity=intensity,
                                )
                            )
                        renderer_params.lights = lights
                    # Legacy lighting config
                    elif "light_direction" in camera_config:
                        direction = np.array(camera_config["light_direction"])
                        direction = direction / np.linalg.norm(direction)
                        renderer_params.lights = [LightParameter(direction=direction)]

                    drake_camera_config = CameraConfig(
                        renderer_name=camera_config.name,
                        renderer_class=renderer_params,
                        **kwargs,
                    )
                # Use default renderer
                else:
                    drake_camera_config = CameraConfig(
                        **kwargs,
                    )

                if camera_config.randomize:
                    drake_camera_config = randomize_camera_config(drake_camera_config)
                camera_configs.append(drake_camera_config)
            sim_config.camera_configs = camera_configs
        if "multi_run_config" in cfg and cfg.multi_run_config:
            sim_config.multi_run_config = hydra.utils.instantiate(cfg.multi_run_config)

        return sim_config

    def __eq__(self, other: "PlanarPushingSimConfig"):
        # Check camera configs
        if self.camera_configs is None and other.camera_configs is not None:
            return False
        if self.camera_configs is not None and other.camera_configs is None:
            return False
        if self.camera_configs is not None:
            for camera_config in self.camera_configs:
                if camera_config not in other.camera_configs:
                    return False

        return (
            self.slider == other.slider
            and self.contact_model == other.contact_model
            and self.visualize_desired == other.visualize_desired
            and self.slider_goal_pose == other.slider_goal_pose
            and self.pusher_start_pose == other.pusher_start_pose
            and self.time_step == other.time_step
            and self.draw_frames == other.draw_frames
            and self.use_realtime == other.use_realtime
            and self.delay_before_execution == other.delay_before_execution
            and self.save_plots == other.save_plots
            and self.scene_directive_name == other.scene_directive_name
            and self.use_hardware == other.use_hardware
            and self.pusher_z_offset == other.pusher_z_offset
            and self.log_dir == other.log_dir
            and np.allclose(self.default_joint_positions, other.default_joint_positions)
            and self.diffusion_policy_config == other.diffusion_policy_config
            and self.multi_run_config == other.multi_run_config
            and self.domain_randomization_color_range == other.domain_randomization_color_range
            and self.arbitrary_shape_pickle_path == other.arbitrary_shape_pickle_path
            and np.allclose(self.arbitrary_shape_rgba, other.arbitrary_shape_rgba)
            and self.slider_physical_properties == other.slider_physical_properties
            and self.joint_velocity_limit_factor == other.joint_velocity_limit_factor
            and self.constant_velocity_disturbance == other.constant_velocity_disturbance
            and self.constant_velocity_disturbance_Kp == other.constant_velocity_disturbance_Kp
            and self.constant_velocity_disturbance_Kd == other.constant_velocity_disturbance_Kd
            and self.constant_velocity_disturbance_force_cap == other.constant_velocity_disturbance_force_cap
            and self.constant_velocity_disturbance_velocity_window_size
            == other.constant_velocity_disturbance_velocity_window_size
            and self.constant_velocity_disturbance_tune_mode == other.constant_velocity_disturbance_tune_mode
        )
