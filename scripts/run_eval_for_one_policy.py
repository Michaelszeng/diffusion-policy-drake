import numpy as np
from omegaconf import OmegaConf
from pydrake.all import HPolyhedron, RigidTransform, RotationMatrix, StartMeshcat, VPolytope

from diffusion_experiments.simulation.grasp_setup_diffusion_sim import IiwaTableBlockDiffusionDiffIK

# export PYTHONPATH="${PYTHONPATH}:$(pwd)"

cfg = {
    "scenario_path": "diffusion_experiments/simulation/robot_models/two_bin_task_2.dmd.yaml",
    "checkpoint_path": "/home/michzeng/diffusion-search-learning/data/outputs/grasp_two_bins_flat/same_middle_same_return/basic_training/2_obs/checkpoints/latest.ckpt",
    "device": "cuda:0",
}

if __name__ == "__main__":
    hydra_cfg = OmegaConf.create(cfg)

    eval_from_left_box = True
    if eval_from_left_box:
        x_range = [-0.14, -0.3]
        y_range = [-0.66, -0.54]
    else:  # right box
        x_range = [0.14, 0.3]
        y_range = [-0.66, -0.54]

    meshcat = StartMeshcat()

    total_success = 0
    total_other_box = 0
    total_fail = 0

    for i in range(300, 325):
        np.random.seed(i)
        x_point = np.random.uniform(*x_range)
        y_point = np.random.uniform(*y_range)
        print(f"Eval run {i}, x_point: {x_point}, y_point: {y_point}")

        # left boxs
        X_WO = {
            "initial": RigidTransform(RotationMatrix.MakeZRotation(np.pi / 2), [x_point, y_point, 0.020]),
            "goal": RigidTransform(RotationMatrix.MakeZRotation(np.pi / 2), [0, -0.6, 0.018]),
        }

        left_box = HPolyhedron(VPolytope(np.array([[-0.1, -0.1, -0.34, -0.34], [-0.5, -0.7, -0.7, -0.5]])))
        right_box = HPolyhedron(VPolytope(np.array([[0.1, 0.1, 0.34, 0.34], [-0.5, -0.7, -0.7, -0.5]])))

        system = IiwaTableBlockDiffusionDiffIK(
            hydra_cfg, X_WO, meshcat=meshcat, debug=True, left_box=left_box, right_box=right_box
        )
        # system.run(save_html="grasp_task_16_obs_frozen_encoder_try_5.html")
        success = system.run_eval(save_html="grasp_task_16_obs_frozen_encoder_try_5_eval.html", max_time=50)

        if success == 0:
            total_success += 1
        elif success == 1:
            total_other_box += 1
        else:
            total_fail += 1

    print(f"Total success: {total_success}, total other box: {total_other_box}, total fail: {total_fail}")

    # RotationMatrix.MakeZRotation(np.pi/2), [0.36, -0.48, 0.022]
