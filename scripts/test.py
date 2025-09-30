# import numpy as np
# import random

# random.seed(100)
# np.random.seed(56)
# for trial_idx in range(10):
#     ss = np.random.SeedSequence([100, trial_idx])
#     trial_rng = np.random.default_rng(ss)
#     np.random.uniform(0, 1)
#     print(trial_rng.uniform(0, 1))

import pickle

with open('arbitrary_shape_pickles/small_t_pusher.pkl', 'rb') as file:
    loaded_data = pickle.load(file)
    print(loaded_data)

import pickle
import time

import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    Box,
    CoulombFriction,
    DiagramBuilder,
    RigidTransform,
    RotationMatrix,
    SpatialInertia,
    StartMeshcat,
    UnitInertia,
)
from pydrake.visualization import AddDefaultVisualization


def _rt_from_homog(H):
    H = np.asarray(H)
    R = RotationMatrix(H[:3, :3])
    p = H[:3, 3]
    return RigidTransform(R, p)


def _make_object_instance(plant, parts, model_name=""):
    """One free body with many box collision+visual geometries fixed to it."""
    mass = 1.0
    # Any valid inertia is fine for collision queries; pick something simple.
    G = UnitInertia.SolidBox(0.1, 0.1, 0.1)
    M_B = SpatialInertia(mass=mass, p_PScm_E=np.array([0, 0, 0]).reshape((3, 1)), G_SP_E=G)

    model = plant.AddModelInstance(model_name)
    body = plant.AddRigidBody(f"{model_name}_body", model, M_B)

    for i, part in enumerate(parts):
        sx, sy, sz = part["size"]
        X_BGi = _rt_from_homog(part["transform"])
        shape = Box(sx, sy, sz)
        mu = CoulombFriction(0.9, 0.5)
        # Collision
        plant.RegisterCollisionGeometry(body, X_BGi, shape, f"{model_name}_box_{i}", mu)
        # Visual (so you can see it in Meshcat)
        plant.RegisterVisualGeometry(
            body,
            X_BGi,
            shape,
            f"{model_name}_box_vis_{i}",
            diffuse_color=np.array([0.6, 0.6, 0.6, 1.0]).reshape((4, 1)),
        )
    return model, body


def make_collision_checker_from_pickle(pickle_path):
    """Build a diagram with two identical instances and a collision checker."""
    with open(pickle_path, "rb") as f:
        parts = pickle.load(f)

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)

    _, body_A = _make_object_instance(plant, parts, model_name="A")
    _, body_B = _make_object_instance(plant, parts, model_name="B")

    meshcat = StartMeshcat()
    plant.Finalize()
    AddDefaultVisualization(builder, meshcat)  # publish_period defaults are fine

    diagram = builder.Build()

    # Shared, reusable context for quick pose updates.
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    sg_context = scene_graph.GetMyContextFromRoot(context)

    def set_pose(body, X_WB):
        plant.SetFreeBodyPose(plant_context, body, X_WB)

    def check_collision(X_WA: RigidTransform, X_WB: RigidTransform) -> bool:
        set_pose(body_A, X_WA)
        set_pose(body_B, X_WB)

        # Ask SceneGraph for penetrations.
        query_object = scene_graph.get_query_output_port().Eval(sg_context)
        inspector = query_object.inspector()
        penetrations = query_object.ComputePointPairPenetration()
        print(
            f"penetrations: {[f'{inspector.GetName(pen.id_A)}, {inspector.GetName(pen.id_B)}' for pen in penetrations]}"
        )
        if not penetrations:
            return False

        if len(penetrations) >= 1:
            return True

        return False

    # A tiny helper for demos / visualization
    def visualize_once(X_WA: RigidTransform, X_WB: RigidTransform):
        set_pose(body_A, X_WA)
        set_pose(body_B, X_WB)
        # Force a publish so Meshcat updates immediately.
        diagram.ForcedPublish(context)

    return check_collision, visualize_once, diagram, plant, scene_graph, context


# -------------------------
# Example: visualize + sweep B past A
# -------------------------
if __name__ == "__main__":
    from pydrake.all import RigidTransform, RollPitchYaw

    # 1) Build the checker + visualizer
    check_collision, visualize_once, diagram, plant, scene_graph, context = make_collision_checker_from_pickle(
        "arbitrary_shape_pickles/small_t_pusher.pkl"
    )

    # 2) Pose A at origin; slide B along +x while visualizing and reporting collisions.
    X_WA = RigidTransform()  # identity
    z_lift = 0.0  # adjust if you want them off the ground
    y_offset = 0.0

    # Start a simple sweep
    print("Sweeping B along +x; watch Meshcat (URL printed above). Ctrl-C to stop.")
    for x in np.linspace(-0.20, 0.20, 81):
        X_WB = RigidTransform(RollPitchYaw(0, 0, 0), [x, y_offset, z_lift])

        # Update visualization immediately
        visualize_once(X_WA, X_WB)

        # Collision status
        colliding = check_collision(X_WA, X_WB)
        print(f"x={x:+.3f}  colliding={colliding}")

        time.sleep(0.1)
