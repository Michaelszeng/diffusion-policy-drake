import os
import time

import numpy as np
import pydot
from manipulation.station import LoadScenario, MakeHardwareStation
from manipulation.systems import AddIiwaDifferentialIK
from pydrake.all import (
    AbstractValue,
    Context,
    DiagramBuilder,
    LeafSystem,
    LogVectorOutput,
    MultibodyPlant,
    QueryObject,
    RigidTransform,
    RobotDiagram,
    RotationMatrix,
    Simulator,
    StartMeshcat,
    ZeroOrderHold,
)

from diffusion_experiments.algorithms.diffusion_policy_drake_controller import DiffusionPolicyDrakeController
from diffusion_experiments.utils.drake_utils import change_camera_to_point_lighting


def convert_pose_to_diffusion_action(X_WG: RigidTransform, wsg_position: float) -> np.ndarray:
    action = np.zeros(13)
    action[:3] = X_WG.translation()
    action[3:12] = X_WG.rotation().matrix().flatten()
    action[12] = wsg_position
    return action


class DiffusionOutputDiffIKConverter(LeafSystem):
    """
    Convert output of policy to format for diff IK system (RigidTransform and WSG scalar position).
    """

    def __init__(self):
        super().__init__()

        self.DeclareVectorInputPort("diffusion_action", 13)

        self.DeclareAbstractOutputPort("X_WG", lambda: AbstractValue.Make(RigidTransform()), self.CalcGripperPose)
        self.DeclareVectorOutputPort("wsg_position", 1, self.CalcWsgPosition)

    def CalcGripperPose(self, context: Context, output):
        diffusion_action = self.get_input_port(0).Eval(context)
        pose = diffusion_action[:3]
        rotation = diffusion_action[3:12].reshape(3, 3)

        # do the reprojection of rotation matrix
        U, S, Vt = np.linalg.svd(rotation)
        rotation = U @ Vt
        if np.linalg.det(rotation) < 0:
            rotation[:, 2] *= -1

        X_WG = RigidTransform(RotationMatrix(rotation), pose)
        output.set_value(X_WG)

    def CalcWsgPosition(self, context: Context, output):
        diffusion_action = self.get_input_port(0).Eval(context)
        wsg_position = diffusion_action[12]
        output.SetFromVector([wsg_position])


class IiwaTableBlockDiffusionDiffIK:
    def __init__(self, cfg, X_WO, debug=False, meshcat=None, left_box=None, right_box=None):
        builder = DiagramBuilder()

        if debug:
            if meshcat is None:
                meshcat = StartMeshcat()

        scenario = LoadScenario(filename=cfg.scenario_path, scenario_name="overall")

        scenario = change_camera_to_point_lighting(scenario, main_camera_name="camera0")

        package_path = os.path.abspath("diffusion_experiments/simulation/robot_models/package.xml")
        package_list = [package_path]

        station: RobotDiagram = builder.AddSystem(
            MakeHardwareStation(scenario, meshcat=meshcat, package_xmls=package_list)
        )

        plant: MultibodyPlant = station.GetSubsystemByName("plant")
        plant.SetDefaultFreeBodyPose(plant.GetBodyByName("base_link"), X_WO["initial"])

        temp_station_context = station.CreateDefaultContext()
        temp_plant_context = plant.GetMyContextFromRoot(temp_station_context)
        wsg_instance = plant.GetModelInstanceByName("wsg")

        current_wsg_position = plant.GetPositions(temp_plant_context, wsg_instance)
        initial_gripper_action = convert_pose_to_diffusion_action(X_WO["initial"], np.sum(np.abs(current_wsg_position)))

        controller = builder.AddSystem(
            DiffusionPolicyDrakeController(
                checkpoint_path=cfg.checkpoint_path,
                initial_gripper_command=initial_gripper_action,
                plant=plant,
                policy_device=cfg.device,
            )
        )
        zero_order_hold = (
            builder.AddSystem(  # Downsample the diffusion querying to 10 Hz (hold action in between queries )
                ZeroOrderHold(
                    period_sec=0.1,
                    vector_size=13,
                )
            )
        )
        diff_to_diff_ik = builder.AddSystem(DiffusionOutputDiffIKConverter())

        # connect the camera ports
        builder.Connect(station.GetOutputPort("camera0.rgb_image"), controller.GetInputPort("overhead_camera_in"))
        builder.Connect(station.GetOutputPort("camera1.rgb_image"), controller.GetInputPort("wrist_camera_in"))
        # connect other controller ports
        builder.Connect(station.GetOutputPort("body_poses"), controller.GetInputPort("body_poses"))
        builder.Connect(station.GetOutputPort("wsg_state"), controller.GetInputPort("wsg_state_in"))

        builder.Connect(controller.get_output_port(0), zero_order_hold.get_input_port(0))

        builder.Connect(zero_order_hold.get_output_port(0), diff_to_diff_ik.get_input_port(0))

        # do diff ik connections
        robot_controller = station.GetSubsystemByName("iiwa_controller_plant_pointer_system").get()
        diff_ik = AddIiwaDifferentialIK(builder, robot_controller, frame=plant.GetFrameByName("body"))
        builder.Connect(diff_to_diff_ik.GetOutputPort("X_WG"), diff_ik.GetInputPort("X_AE_desired"))
        builder.Connect(diff_ik.get_output_port(), station.GetInputPort("iiwa.position"))
        builder.Connect(station.GetOutputPort("iiwa.state_estimated"), diff_ik.GetInputPort("robot_state"))

        builder.Connect(diff_to_diff_ik.GetOutputPort("wsg_position"), station.GetInputPort("wsg.position"))

        self.logger = LogVectorOutput(diff_to_diff_ik.GetOutputPort("wsg_position"), builder)

        self.diagram = builder.Build()

        pydot.graph_from_dot_data(self.diagram.GetGraphvizString())[0].write_pdf(  # type: ignore
            "diagram.pdf"
        )

        self.simulator = Simulator(self.diagram)
        self.debug = debug
        self.controller = controller
        self.plant = plant
        self.dt = scenario.plant_config.time_step
        self.meshcat = meshcat
        self.X_WO = X_WO

        if left_box is not None:
            if np.all(left_box.A() @ X_WO["initial"].translation()[:2] < left_box.b()):
                self.main_box = left_box
                self.other_box = right_box
            elif np.all(right_box.A() @ X_WO["initial"].translation()[:2] < right_box.b()):
                self.main_box = right_box
                self.other_box = left_box
            else:
                raise ValueError("Initial position is not in either box.")

        # configure collision filtering
        self.scene_graph = station.scene_graph()

        # get the model instances
        self.left_box_model_instance = plant.GetModelInstanceByName("bin1")
        self.right_box_model_instance = plant.GetModelInstanceByName("bin2")
        self.wsg_model_instance = plant.GetModelInstanceByName("wsg")
        self.table_model_instance = plant.GetModelInstanceByName("table")

        # get the bodies
        self.left_box_body = plant.GetBodyByName("box_link_blue", self.left_box_model_instance)
        self.right_box_body = plant.GetBodyByName("box_link_green", self.right_box_model_instance)
        self.brick_body = plant.GetBodyByName("base_link")
        self.left_finger_body = plant.GetBodyByName("left_finger", self.wsg_model_instance)
        self.right_finger_body = plant.GetBodyByName("right_finger", self.wsg_model_instance)
        self.wsg_main_body = plant.GetBodyByName("body", self.wsg_model_instance)
        self.table_body = plant.GetBodyByName("table_body", self.table_model_instance)

        # get the geometries
        self.left_box_geom_ids = plant.GetCollisionGeometriesForBody(self.left_box_body)
        self.right_box_geom_ids = plant.GetCollisionGeometriesForBody(self.right_box_body)
        self.brick_geom_ids = plant.GetCollisionGeometriesForBody(self.brick_body)
        self.left_finger_geom_ids = plant.GetCollisionGeometriesForBody(self.left_finger_body)
        self.right_finger_geom_ids = plant.GetCollisionGeometriesForBody(self.right_finger_body)
        self.wsg_main_geom_ids = plant.GetCollisionGeometriesForBody(self.wsg_main_body)
        self.table_geom_ids = plant.GetCollisionGeometriesForBody(self.table_body)
        self.finger_combined_geom_ids = self.left_finger_geom_ids + self.right_finger_geom_ids
        self.combined_static_geom_ids = self.left_box_geom_ids + self.right_box_geom_ids + self.table_geom_ids

    def run(self, save_html=None):
        """
        Run the policy in simulation for 12,000 steps.
        """
        assert self.meshcat is not None, "Meshcat must be initialized for visualization."
        simulator = self.simulator
        diagram = self.diagram
        context = simulator.get_mutable_context()
        diagram.ForcedPublish(context)

        i = 0
        dt = 0.01
        self.meshcat.Flush()
        self.meshcat.StartRecording()
        while i <= 12000:
            simulator.AdvanceTo(i * dt)
            i += 1
        self.meshcat.PublishRecording()

        log = self.logger.FindLog(simulator.get_context())
        data = log.data()

        time.sleep(2.0)
        if save_html is not None:
            print(f"Generating recording in {save_html} ...")
            html = self.meshcat.StaticHtml()
            with open(save_html, "w") as f:
                f.write(html)
        breakpoint()

    def run_eval(self, save_html=None, max_time=50):
        """
        Run policy in simulation for a maximum of 50 seconds, tracking success metrics and failure for 2-bins task.
        """
        assert self.main_box is not None, "Main box must be set for evaluation."
        assert self.other_box is not None, "Other box must be set for evaluation."

        simulator = self.simulator
        diagram = self.diagram
        plant = self.plant
        scene_graph = self.scene_graph

        context = simulator.get_mutable_context()
        plant_context = plant.GetMyMutableContextFromRoot(context)
        scene_graph_context = scene_graph.GetMyMutableContextFromRoot(context)

        brick_body = plant.GetBodyByName("base_link")
        wsg_instance = plant.GetModelInstanceByName("wsg")

        i = 0
        dt = 0.01

        if self.meshcat is not None:
            self.meshcat.Flush()
            self.meshcat.StartRecording()

        success = False
        reached_mid = False
        correct_traj_to_a_box = False

        while i * dt < max_time:
            simulator.AdvanceTo(i * dt)
            i += 1

            brick_pose_translation = plant.EvalBodyPoseInWorld(plant_context, brick_body).translation()
            gripper_command = plant.GetPositions(plant_context, wsg_instance)

            if np.sum(np.abs(gripper_command)) > 0.08:
                if reached_mid:
                    if brick_pose_translation[2] < 0.02:
                        if np.all(self.main_box.A() @ brick_pose_translation[:2] < self.main_box.b()):
                            success = True
                            break
                        elif np.all(self.other_box.A() @ brick_pose_translation[:2] < self.other_box.b()):
                            correct_traj_to_a_box = True
                            break

                elif brick_pose_translation[2] < 0.01 and np.allclose(
                    brick_pose_translation[:2], self.X_WO["goal"].translation()[:2], atol=0.05
                ):
                    reached_mid = True
                    print("Reached the mid position, now moving to the goal position.")

            sg_query_obj: QueryObject = scene_graph.get_query_output_port().Eval(scene_graph_context)
            collision_pairs = sg_query_obj.ComputeSignedDistancePairwiseClosestPoints()

            breaksim = False
            for collision_pair in collision_pairs:
                if collision_pair.distance < -3e-3:
                    if (
                        collision_pair.id_A in self.brick_geom_ids
                        and collision_pair.id_B not in self.finger_combined_geom_ids
                    ) or (
                        collision_pair.id_B in self.brick_geom_ids
                        and collision_pair.id_A not in self.finger_combined_geom_ids
                    ):
                        print("Unreasonable penetration detected with brick")
                        # breakpoint()
                        breaksim = True

                if collision_pair.distance < 1e-4:
                    if (
                        collision_pair.id_A in self.wsg_main_geom_ids
                        and collision_pair.id_B not in self.finger_combined_geom_ids
                    ) or (
                        collision_pair.id_B in self.wsg_main_geom_ids
                        and collision_pair.id_A not in self.finger_combined_geom_ids
                    ):
                        print("Unreasonable collision detected with WSG main body")
                        breaksim = True

                    if (
                        collision_pair.id_A in self.finger_combined_geom_ids
                        and collision_pair.id_B in self.combined_static_geom_ids
                    ) or (
                        collision_pair.id_B in self.finger_combined_geom_ids
                        and collision_pair.id_A in self.combined_static_geom_ids
                    ):
                        print("Unreasonable collision detected with WSG fingers")
                        breaksim = True

            if breaksim:
                break

        if success:
            print("Successfully reached the goal position.")
            return 0
        elif correct_traj_to_a_box:
            print("Successfully reached the other box.")
            return 1
        else:
            print("Failed to reach the goal position.")
            return 2
