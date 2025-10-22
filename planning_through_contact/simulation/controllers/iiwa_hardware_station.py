import numpy as np
from pydrake.all import (
    Context,
    Demultiplexer,
    DiagramBuilder,
    DifferentialInverseKinematicsIntegrator,
    DifferentialInverseKinematicsParameters,
    Meshcat,
    Multiplexer,
    PortSwitch,
    StateInterpolatorWithDiscreteDerivative,
)

from manipulation_copy.station import (
    IiwaDriver,
    InverseDynamicsDriver,
    JointStiffnessDriver,
    LoadScenario,
    MakeHardwareStation,
)
from planning_through_contact.simulation.planar_pushing_sim_config import PlanarPushingSimConfig
from planning_through_contact.simulation.sim_utils import (
    GetSliderUrl,
    LoadRobotOnly,
    get_randomized_slider_sdf_string,
    models_folder,
    package_xml_file,
    randomize_pusher,
    randomize_table,
)
from planning_through_contact.simulation.systems.constant_velocity_disturber import ConstantVelocityDisturber
from planning_through_contact.simulation.systems.iiwa_planner import IiwaPlanner
from planning_through_contact.simulation.systems.joint_velocity_clamp import JointVelocityClamp
from planning_through_contact.simulation.systems.planar_translation_to_rigid_transform_system import (
    PlanarTranslationToRigidTransformSystem,
)
from planning_through_contact.simulation.systems.run_flag_system import RunFlagSystem
from planning_through_contact.simulation.systems.success_checker import SuccessChecker

from .robot_system_base import RobotSystemBase


def set_meshcat_camera_pose(meshcat: Meshcat, sim_config: PlanarPushingSimConfig):
    """Simpler helper function to set the initial camera pose for meshcat visualization"""
    zoom = 1.8
    camera_in_world = [
        sim_config.slider_goal_pose.x,
        (sim_config.slider_goal_pose.y - 1) / zoom,
        1.5 / zoom,
    ]
    target_in_world = [
        sim_config.slider_goal_pose.x,
        sim_config.slider_goal_pose.y,
        0,
    ]
    meshcat.SetCameraPose(camera_in_world, target_in_world)


class IiwaHardwareStation(RobotSystemBase):
    def __init__(
        self,
        sim_config: PlanarPushingSimConfig,
        meshcat: Meshcat,
    ):
        super().__init__()
        self._sim_config = sim_config
        self._meshcat = meshcat
        self._num_positions = 7

        if sim_config.use_hardware:
            scenario_name = "speed-optimized"
        else:
            scenario_name = "accuracy-optimized"

        scenario_file_name = "planar_pushing_iiwa_scenario.yaml"

        if sim_config.domain_randomization_color_range <= 0.0:
            # Define callback for MakeHardwareStation
            def add_slider_to_parser(parser):
                """Just add slider model by SDF url"""
                slider_sdf_url = GetSliderUrl(sim_config)
                (slider,) = parser.AddModels(url=slider_sdf_url)
        else:
            slider_grey = np.random.uniform(0.1, 0.25)
            table_grey = np.random.uniform(0.3, 0.95)
            mu_dynamic = sim_config.slider_physical_properties.mu_dynamic
            mu_static = sim_config.slider_physical_properties.mu_static
            randomize_pusher(color_range=sim_config.domain_randomization_color_range)
            randomize_table(
                default_color=[table_grey, table_grey, table_grey],
                color_range=sim_config.domain_randomization_color_range,
                mu_dynamic=mu_dynamic,
                mu_static=mu_static,
            )

            # Define callback for MakeHardwareStation
            def add_slider_to_parser(parser):
                """Just add slider model (by SDF string) with randomized color"""
                sdf_string = get_randomized_slider_sdf_string(
                    sim_config,
                    default_color=[slider_grey, slider_grey, slider_grey],
                    color_range=sim_config.domain_randomization_color_range,
                )
                (slider,) = parser.AddModelsFromString(sdf_string, "sdf")

        scenario = LoadScenario(filename=f"{models_folder}/{scenario_file_name}", scenario_name=scenario_name)

        # Add cameras to scenario (they aren't included in the scenario yaml file)
        if sim_config.camera_configs:
            for camera_config in sim_config.camera_configs:
                scenario.cameras[camera_config.name] = camera_config

        builder = DiagramBuilder()

        ################################################################################################################
        ### Add systems
        ################################################################################################################

        # Kuka station
        self.station = builder.AddSystem(
            MakeHardwareStation(
                scenario,
                meshcat=meshcat,
                package_xmls=[package_xml_file],
                hardware=sim_config.use_hardware,
                parser_prefinalize_callback=add_slider_to_parser,
            ),
        )
        if not sim_config.use_hardware:
            external_mbp = self.station.GetSubsystemByName("plant")
            self.station_plant = external_mbp
            self._scene_graph = self.station.scene_graph()
            self.slider = external_mbp.GetModelInstanceByName(sim_config.slider.name)

            # Connect velocity disturbance controller if enabled
            if sim_config.constant_velocity_disturbance > 0.0:
                self.constant_velocity_disturber = builder.AddSystem(
                    ConstantVelocityDisturber(
                        plant=external_mbp,
                        body_index=external_mbp.GetBodyByName(sim_config.slider.name).index(),
                        Kp=sim_config.constant_velocity_disturbance_Kp,
                        Kd=sim_config.constant_velocity_disturbance_Kd,
                        force_cap=sim_config.constant_velocity_disturbance_force_cap,
                        velocity_window_size=sim_config.constant_velocity_disturbance_velocity_window_size,
                        tune_mode=sim_config.constant_velocity_disturbance_tune_mode,
                    ),
                )

                # Add success checker to disable disturbance when goal is reached
                # SuccessChecker will automatically determine mode and load convex hulls if needed
                self.success_checker = builder.AddSystem(
                    SuccessChecker(
                        plant=external_mbp,
                        slider_model_instance=self.slider,
                        pusher_body_index=external_mbp.GetBodyByName("pusher").index(),
                        slider_goal_pose=sim_config.slider_goal_pose,
                        pusher_goal_pose=sim_config.pusher_start_pose,
                        sim_config=sim_config,
                        trans_tol=sim_config.multi_run_config.trans_tol,
                        rot_tol=sim_config.multi_run_config.rot_tol,
                        evaluate_final_slider_rotation=True,
                        evaluate_final_pusher_position=True,
                    )
                )

        # Iiwa Planner state machine
        INITIAL_DELAY = 0.5  # Delay between starting simulation and iiwa starting to go to home position
        WAIT_PUSH_DELAY = 1.0  # Delay between iiwa reaching home position and pusher starting to follow pushing traj
        assert sim_config.delay_before_execution > INITIAL_DELAY + WAIT_PUSH_DELAY
        self._planner = builder.AddNamedSystem(
            "IiwaPlanner",
            IiwaPlanner(
                sim_config=sim_config,
                robot_plant=LoadRobotOnly(sim_config, robot_plant_file="iiwa_controller_plant.yaml"),
                initial_delay=INITIAL_DELAY,
                wait_push_delay=WAIT_PUSH_DELAY,
            ),
        )

        # Diff IK "position controller" to follow output of diffusion policy
        EE_FRAME = "pusher_end"
        robot = LoadRobotOnly(sim_config, robot_plant_file="iiwa_controller_plant.yaml")
        self.robot = robot
        ik_params = DifferentialInverseKinematicsParameters(robot.num_positions(), robot.num_velocities())
        ik_params.set_time_step(sim_config.time_step)
        # True velocity limits for the IIWA14 and IIWA7 (in rad, rounded down to the first decimal)
        # IIWA14_VELOCITY_LIMITS = np.array([1.4, 1.4, 1.7, 1.3, 2.2, 2.3, 2.3])
        IIWA7_VELOCITY_LIMITS = np.array([1.7, 1.7, 1.7, 2.2, 2.4, 3.1, 3.1])
        velocity_limit_factor = sim_config.joint_velocity_limit_factor
        ik_params.set_joint_velocity_limits(
            (
                -velocity_limit_factor * IIWA7_VELOCITY_LIMITS,
                velocity_limit_factor * IIWA7_VELOCITY_LIMITS,
            )
        )
        ik_params.set_nominal_joint_position(self._sim_config.default_joint_positions)
        ik_params.set_joint_centering_gain(np.eye(robot.num_positions()))
        self._diff_ik = builder.AddNamedSystem(
            "DiffIk",
            DifferentialInverseKinematicsIntegrator(
                robot,
                robot.GetFrameByName(EE_FRAME),
                sim_config.time_step,
                ik_params,
            ),
        )

        # Converter between (x,y) to RigidTransform
        planar_translation_to_rigid_tranform = builder.AddSystem(
            PlanarTranslationToRigidTransformSystem(z_dist=sim_config.pusher_z_offset)
        )

        # Converter from desired position q into desired state (q, q_dot)
        driver_config = scenario.model_drivers["iiwa"]
        if isinstance(driver_config, JointStiffnessDriver) or isinstance(driver_config, InverseDynamicsDriver):
            self._state_interpolator = builder.AddNamedSystem(
                "StateInterpolatorWithDiscreteDerivative",
                StateInterpolatorWithDiscreteDerivative(
                    robot.num_positions(),
                    time_step=sim_config.time_step,
                ),
            )

        # Switch for switching between planner output (for GoPushStart), and diff IK output (for pushing)
        switch = builder.AddNamedSystem("switch", PortSwitch(robot.num_positions()))
        run_flag_system = builder.AddSystem(RunFlagSystem(true_port_index=2))  # RunFlagSystem outputs 1 if PUSHING mode

        # Iiwa state estimated multiplexer ; separate estimated state to position and velocity
        if isinstance(driver_config, IiwaDriver):
            iiwa_state_estimated_mux = builder.AddSystem(
                Multiplexer(input_sizes=[robot.num_positions(), robot.num_velocities()])
            )

        # Velocity clamp to prevent sudden spike when switching to diff IK
        joint_velocity_clamp = builder.AddNamedSystem(
            "JointVelocityClamp",
            JointVelocityClamp(
                num_positions=robot.num_positions(),
                joint_velocity_limits=velocity_limit_factor * IIWA7_VELOCITY_LIMITS,
            ),
        )

        ################################################################################################################
        ### Connect systems
        ################################################################################################################

        # Inputs to diff IK
        builder.Connect(
            planar_translation_to_rigid_tranform.get_output_port(),
            self._diff_ik.GetInputPort("X_WE_desired"),
        )

        # builder.Connect(
        #     const.get_output_port(),
        #     self._diff_ik.GetInputPort("use_robot_state"),
        # )
        # Strangely, when we use the planner's reset_diff_ik port, which sets use_robot_state to True before the pushing
        # phase and False during the pushing phase, we get persistent diff IK drift.
        builder.Connect(
            self._planner.GetOutputPort("reset_diff_ik"),
            self._diff_ik.GetInputPort("use_robot_state"),
        )
        if isinstance(driver_config, JointStiffnessDriver) or isinstance(driver_config, InverseDynamicsDriver):
            # Inputs to the planner
            # Need an additional demultiplexer to split state_estimated into position and velocity
            demux = builder.AddSystem(Demultiplexer([robot.num_positions(), robot.num_velocities()]))
            builder.Connect(
                self.station.GetOutputPort("iiwa.state_estimated"),
                demux.get_input_port(),
            )
            builder.Connect(
                demux.get_output_port(0),
                self._planner.GetInputPort("iiwa_position_measured"),
            )

            # Input to Diff IK
            builder.Connect(
                self.station.GetOutputPort("iiwa.state_estimated"),
                self._diff_ik.GetInputPort("robot_state"),
            )

            # Input to joint velocity clamp
            builder.Connect(
                switch.get_output_port(),
                joint_velocity_clamp.get_input_port(),
            )

            # Inputs to state interpolator
            builder.Connect(
                joint_velocity_clamp.get_output_port(),
                self._state_interpolator.get_input_port(),
            )

            # Inputs to station
            builder.Connect(
                self._state_interpolator.get_output_port(),
                self.station.GetInputPort("iiwa.desired_state"),
            )

        elif isinstance(driver_config, IiwaDriver):
            # Inputs to the planner
            builder.Connect(
                self.station.GetOutputPort("iiwa.position_measured"),
                self._planner.GetInputPort("iiwa_position_measured"),
            )

            # Inputs to the state estimator multiplexer
            builder.Connect(
                self.station.GetOutputPort("iiwa.position_measured"),
                iiwa_state_estimated_mux.get_input_port(0),
            )
            builder.Connect(
                self.station.GetOutputPort("iiwa.velocity_estimated"),
                iiwa_state_estimated_mux.get_input_port(1),
            )

            # Input to Diff IK
            builder.Connect(
                iiwa_state_estimated_mux.get_output_port(),
                self._diff_ik.GetInputPort("robot_state"),
            )

            # Input to joint velocity clamp
            builder.Connect(
                switch.get_output_port(),
                joint_velocity_clamp.get_input_port(),
            )

            # Inputs to station
            builder.Connect(
                joint_velocity_clamp.get_output_port(),
                self.station.GetInputPort("iiwa.position"),
            )

        # Inputs to switch
        builder.Connect(
            self._planner.GetOutputPort("iiwa_position_command"),
            switch.DeclareInputPort("planner_iiwa_position_command"),
        )
        builder.Connect(
            self._diff_ik.get_output_port(),
            switch.DeclareInputPort("diff_ik_iiwa_position_cmd"),
        )
        builder.Connect(
            self._planner.GetOutputPort("control_mode"),
            switch.get_port_selector_input_port(),
        )
        builder.Connect(
            self._planner.GetOutputPort("control_mode"),
            run_flag_system.get_input_port(),
        )

        ## Export inputs and outputs
        builder.ExportInput(
            planar_translation_to_rigid_tranform.get_input_port(),
            "planar_position_command",
        )
        builder.ExportOutput(
            joint_velocity_clamp.get_output_port(),
            "iiwa_position_command",
        )
        builder.ExportOutput(
            run_flag_system.get_output_port(),
            "run_flag",
        )

        if isinstance(driver_config, JointStiffnessDriver) or isinstance(driver_config, InverseDynamicsDriver):
            builder.ExportOutput(
                self.station.GetOutputPort("iiwa.state_estimated"),
                "robot_state_measured",
            )
        elif isinstance(driver_config, IiwaDriver):
            builder.ExportOutput(
                iiwa_state_estimated_mux.get_output_port(),
                "robot_state_measured",
            )

        if not sim_config.use_hardware:
            # Only relevant when use_hardware=False
            # If use_hardware=True, this info will be updated by the optitrack system in the state estimator directly
            builder.ExportOutput(
                self.station.GetOutputPort(f"{sim_config.slider.name}_state"),
                "object_state_measured",
            )

            # Connections to velocity disturbance controller
            if sim_config.constant_velocity_disturbance > 0.0:
                builder.Connect(
                    run_flag_system.get_output_port(),
                    self.constant_velocity_disturber.GetInputPort("enable"),
                )
                builder.Connect(
                    self.station.GetOutputPort("state"),
                    self.constant_velocity_disturber.GetInputPort("x_plant"),
                )
                # Connect success checker
                builder.Connect(
                    self.station.GetOutputPort("state"),
                    self.success_checker.GetInputPort("x_plant"),
                )
                builder.Connect(
                    self.success_checker.GetOutputPort("success"),
                    self.constant_velocity_disturber.GetInputPort("success"),
                )
                builder.Connect(
                    self.constant_velocity_disturber.get_output_port(),
                    self.station.GetInputPort("applied_spatial_force"),
                )

        if sim_config.camera_configs:
            for camera_config in self._sim_config.camera_configs:
                builder.ExportOutput(
                    self.station.GetOutputPort(f"{camera_config.name}.rgb_image"),
                    f"rgbd_sensor_{camera_config.name}",
                )

        builder.BuildInto(self)

        set_meshcat_camera_pose(self._meshcat, self._sim_config)  # For meshcat visualization

        # # Add triad to visualize pusher end effector
        # AddMultibodyTriad(
        #     self.station.GetSubsystemByName("plant").GetFrameByName("pusher_end"),
        #     self.station.scene_graph(),
        #     length=0.04,
        #     radius=0.0025,
        #     opacity=0.25,
        # )

    def pre_sim_callback(self, root_context: Context) -> None: ...

    @property
    def robot_model_name(self) -> str:
        return "iiwa"

    @property
    def slider_model_name(self) -> str:
        """The name of the object being pushed (i.e. t_pusher, box, arbitrary)."""
        if self._sim_config.slider.name == "box":
            return "box"
        elif self._sim_config.slider.name in ["tee", "t_pusher"]:
            return "t_pusher"
        elif self._sim_config.slider.name == "arbitrary":
            return "arbitrary"
        else:
            raise ValueError(f"Invalid slider name: {self._sim_config.slider.name}")

    def num_positions(self) -> int:
        return self._num_positions

    def get_station_plant(self):
        return self.station_plant

    def get_scene_graph(self):
        return self._scene_graph

    def get_slider(self):
        return self.slider

    def get_meshcat(self):
        return self._meshcat
