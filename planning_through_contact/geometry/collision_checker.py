import time
from typing import Optional

import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    Box,
    CoulombFriction,
    Cylinder,
    DiagramBuilder,
    Meshcat,
    RigidTransform,
    RotationMatrix,
    SpatialInertia,
    StartMeshcat,
    UnitInertia,
)
from pydrake.visualization import AddDefaultVisualization

from planning_through_contact.geometry.collision_geometry.arbitrary_shape_2d import (
    ArbitraryShape2D,
)
from planning_through_contact.geometry.planar.planar_pose import PlanarPose


class CollisionChecker:
    """
    Check collisions between 2 bodies
    """

    def __init__(self, pickle_path: str, pusher_radius: float, meshcat: Optional[Meshcat] = None):
        """Build a diagram with slider and pusher and a collision checker."""
        # Use ArbitraryShape2D to load and recenter the geometry
        # This matches how the simulation loads the geometry (centering COM)
        self.shape_spec = ArbitraryShape2D(pickle_path)
        parts = self.shape_spec.primitive_boxes

        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)

        self.plant = plant
        self.scene_graph = scene_graph

        model_A, body_A = self._make_object_instance_from_pickle(parts, model_name="A")
        model_B, body_B = self._make_pusher_instance(pusher_radius, model_name="B")

        plant.Finalize()
        AddDefaultVisualization(builder, meshcat)
        diagram = builder.Build()
        context = diagram.CreateDefaultContext()

        self.diagram = diagram
        self.context = context
        self.body_A = body_A
        self.body_B = body_B
        self.model_A = model_A
        self.model_B = model_B

    def _add_free_body(self, model_name: str, unit_inertia: UnitInertia, mass: float = 1.0):
        model = self.plant.AddModelInstance(model_name)
        M_B = SpatialInertia(mass=mass, p_PScm_E=[0, 0, 0], G_SP_E=unit_inertia)
        body = self.plant.AddRigidBody(f"{model_name}_body", model, M_B)
        return model, body

    def _add_geom(self, body, X_BG: PlanarPose, shape, name_base: str):
        self.plant.RegisterCollisionGeometry(body, X_BG, shape, f"{name_base}_coll", CoulombFriction(0.9, 0.5))
        self.plant.RegisterVisualGeometry(body, X_BG, shape, f"{name_base}_vis", [0.6, 0.6, 0.6, 1.0])

    def _make_object_instance_from_pickle(self, parts, model_name: str = ""):
        """One free body; attach each box from pickle as fixed collision+visual."""

        def _rt_from_homog(H):
            H = np.asarray(H)
            return RigidTransform(RotationMatrix(H[:3, :3]), H[:3, 3])

        model, body = self._add_free_body(model_name, UnitInertia.SolidBox(0.1, 0.1, 0.1))
        for i, part in enumerate(parts):
            shape = Box(*part["size"])
            X_BG = _rt_from_homog(part["transform"])
            self._add_geom(body, X_BG, shape, f"{model_name}_box_{i}")
        return model, body

    def _make_pusher_instance(self, pusher_radius, pusher_height=0.0333, model_name: str = ""):
        """Vertical cylinder “pusher” (axis = +Z), centered at body origin."""
        ui = UnitInertia.SolidCylinder(radius=pusher_radius, length=pusher_height, unit_vector=np.array([0, 0, 1]))
        model, body = self._add_free_body(model_name, ui)
        self._add_geom(body, RigidTransform(), Cylinder(pusher_radius, pusher_height), f"{model_name}")
        return model, body

    def visualize_once(self, X_WA: PlanarPose, X_WB: PlanarPose):
        """tiny helper for meshcat visualization"""
        self._set_pose(self.model_A, self.body_A, X_WA)
        self._set_pose(self.model_B, self.body_B, X_WB)
        # Force a publish so Meshcat updates immediately.
        self.diagram.ForcedPublish(self.context)

    def _set_pose(self, model, body, X_WB):
        q = X_WB.to_generalized_coords(0, z_axis_is_positive=True)
        self.plant.SetPositions(self.plant.GetMyContextFromRoot(self.context), model, q)

    def check_collision(self, X_WA: PlanarPose, X_WB: PlanarPose) -> bool:
        self._set_pose(self.model_A, self.body_A, X_WA)
        self._set_pose(self.model_B, self.body_B, X_WB)

        # Ask SceneGraph for penetrations.
        query_object = self.scene_graph.get_query_output_port().Eval(
            self.scene_graph.GetMyContextFromRoot(self.context)
        )
        # inspector = query_object.inspector()

        # Compute signed distances if desired
        dists = query_object.ComputeSignedDistancePairwiseClosestPoints(max_distance=10.0)
        min_dist = min([d.distance for d in dists]) if len(dists) > 0 else float("inf")
        # print(f"Signed distance: {min_dist}")

        penetrations = query_object.ComputePointPairPenetration()
        # print(
        #     f"collisions: {[f'{inspector.GetName(pen.id_A)}, {inspector.GetName(pen.id_B)}' for pen in penetrations]}"
        # )

        if len(penetrations) >= 1:
            return True
        return False

    def get_signed_distance(self, X_WA: PlanarPose, X_WB: PlanarPose) -> float:
        self._set_pose(self.model_A, self.body_A, X_WA)
        self._set_pose(self.model_B, self.body_B, X_WB)

        # Ask SceneGraph for penetrations.
        query_object = self.scene_graph.get_query_output_port().Eval(
            self.scene_graph.GetMyContextFromRoot(self.context)
        )

        # Compute signed distances if desired
        dists = query_object.ComputeSignedDistancePairwiseClosestPoints(max_distance=10.0)
        min_dist = min([d.distance for d in dists]) if len(dists) > 0 else float("inf")
        # print(f"Signed distance: {min_dist}")
        return min_dist


# Test code
if __name__ == "__main__":
    meshcat = StartMeshcat()
    checker = CollisionChecker("arbitrary_shape_pickles/small_t_pusher.pkl", 0.015, meshcat)

    # Pose A at origin; slide B using meshcat sliders while visualizing and reporting collisions.
    X_WA = PlanarPose(0, 0, 0)  # identity

    meshcat.AddSlider("pusher_x", -0.5, 0.5, 0.0001, -0.2)
    meshcat.AddSlider("pusher_y", -0.5, 0.5, 0.0001, 0.0)

    print("Open Meshcat to control the pusher position.")

    while True:
        x = meshcat.GetSliderValue("pusher_x")
        y = meshcat.GetSliderValue("pusher_y")

        X_WB = PlanarPose(x, y, 0)

        # Update visualization immediately
        checker.visualize_once(X_WA, X_WB)

        # Collision status
        colliding = checker.check_collision(X_WA, X_WB)
        print(f"x={x:+.3f} y={y:+.3f} colliding={colliding}")

        time.sleep(0.05)
