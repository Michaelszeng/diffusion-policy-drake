"""
RLPolicyController: A Drake LeafSystem that runs a trained PPO policy to output
planar_position_command, mirroring the DiffusionPolicyController interface.

The controller loads a checkpoint saved by rl_push_t/ppo.py and runs the
actor_mean network deterministically (no exploration noise at inference).

Usage in run_sim_sim_eval.py / SimulatedRealTableEnvironment:
    from planning_through_contact.simulation.controllers.rl_policy_controller import (
        RLPolicyController,
    )
    controller = RLPolicyController(
        checkpoint="runs/push_t/model.pt",
        initial_pusher_pose=pusher_start_pose,
        target_slider_pose=slider_goal_pose,
    )
    # Use `controller` as desired_position_source in SimulatedRealTableEnvironment
"""

import logging

import numpy as np
import torch
import torch.nn as nn
from pydrake.all import AbstractValue, Context, LeafSystem, RigidTransform

from planning_through_contact.geometry.planar.planar_pose import PlanarPose

logger = logging.getLogger(__name__)

# Workspace bounds matching push_t_gym_env.py
WORKSPACE_X_LIM = (0.30, 0.90)
WORKSPACE_Y_LIM = (-0.35, 0.45)


# ── Minimal copy of the Agent network ─────────────────────────────────────────
# This mirrors the Agent class in rl_push_t/ppo.py so we can load checkpoints
# without importing the training script (which has argparse at module level).


def _layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class _Agent(nn.Module):
    """Shared-trunk MLP actor-critic (3 hidden layers, 256 units, Tanh activations)."""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.critic = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            _layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            _layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            _layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            _layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            _layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            _layer_init(nn.Linear(256, act_dim), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))


# ── RLPolicyController ─────────────────────────────────────────────────────────


class RLPolicyController(LeafSystem):
    """
    Drake LeafSystem that runs a trained PPO policy for planar pushing.

    Input ports:
        pusher_pose_measured  : RigidTransform — end-effector pose in world frame
        slider_pose_measured  : RigidTransform — slider pose in world frame

    Output port:
        planar_position_command : [x, y] absolute position in meters

    The observation (10D) matches PushTDrakeEnv:
        [pusher_x, pusher_y, slider_x, slider_y,
         cos(slider_theta), sin(slider_theta),
         goal_x, goal_y, cos(goal_theta), sin(goal_theta)]

    Actions are delta positions in [-1, 1] scaled by action_scale, integrated
    into absolute commands and clipped to workspace bounds.
    """

    def __init__(
        self,
        checkpoint: str,
        initial_pusher_pose: PlanarPose,
        target_slider_pose: PlanarPose,
        action_scale: float = 0.05,
        freq: float = 10.0,
        delay: float = 1.0,
        device: str = "cpu",
    ):
        super().__init__()
        self._initial_pusher_pose = initial_pusher_pose
        self._target_slider_pose = target_slider_pose
        self._action_scale = action_scale
        self._freq = freq
        self._dt = 1.0 / freq
        self._delay = delay
        self._device = torch.device(device)

        self._load_policy(checkpoint)

        # Absolute position tracker (integrated from delta actions)
        self._current_position = np.array([initial_pusher_pose.x, initial_pusher_pose.y], dtype=np.float64)

        # Reset bookkeeping (mirrors DiffusionPolicyController)
        self._last_reset_time = 0.0
        self._received_reset_signal = True

        # ── Input ports ────────────────────────────────────────────────────────
        self.pusher_pose_measured = self.DeclareAbstractInputPort(
            "pusher_pose_measured", AbstractValue.Make(RigidTransform())
        )
        self.slider_pose_measured = self.DeclareAbstractInputPort(
            "slider_pose_measured", AbstractValue.Make(RigidTransform())
        )

        # ── Output port ────────────────────────────────────────────────────────
        self.DeclareVectorOutputPort("planar_position_command", 2, self.DoCalcOutput)

    def _load_policy(self, checkpoint: str):
        """Load the PPO checkpoint and reconstruct the agent network."""
        payload = torch.load(checkpoint, map_location=self._device)
        obs_dim = payload["obs_dim"]
        act_dim = payload["act_dim"]

        self._agent = _Agent(obs_dim, act_dim).to(self._device)
        self._agent.load_state_dict(payload["agent_state_dict"])
        self._agent.eval()
        logger.info(f"Loaded RL policy from {checkpoint} " f"(obs_dim={obs_dim}, act_dim={act_dim})")

    def _build_obs(self, context: Context) -> np.ndarray:
        """Build the 10D observation from the Drake context."""
        pusher_tf: RigidTransform = self.pusher_pose_measured.Eval(context)
        slider_tf: RigidTransform = self.slider_pose_measured.Eval(context)

        pusher_pose = PlanarPose.from_pose(pusher_tf)
        slider_pose = PlanarPose.from_pose(slider_tf)

        return np.array(
            [
                pusher_pose.x,
                pusher_pose.y,
                slider_pose.x,
                slider_pose.y,
                np.cos(slider_pose.theta),
                np.sin(slider_pose.theta),
                self._target_slider_pose.x,
                self._target_slider_pose.y,
                np.cos(self._target_slider_pose.theta),
                np.sin(self._target_slider_pose.theta),
            ],
            dtype=np.float32,
        )

    def DoCalcOutput(self, context: Context, output):
        time = context.get_time()

        # Honour the startup delay (robot moves to home during this period)
        if self._received_reset_signal:
            self._last_reset_time = time
            self._received_reset_signal = False

        if time < self._last_reset_time + self._delay:
            output.set_value(self._current_position)
            return

        # Build observation and run actor_mean (deterministic)
        obs = self._build_obs(context)
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self._device)  # (1, 10)

        with torch.no_grad():
            action = self._agent.actor_mean(obs_tensor).squeeze(0).cpu().numpy()

        # Integrate delta action into absolute position, clip to workspace
        self._current_position = np.clip(
            self._current_position + action * self._action_scale,
            [WORKSPACE_X_LIM[0], WORKSPACE_Y_LIM[0]],
            [WORKSPACE_X_LIM[1], WORKSPACE_Y_LIM[1]],
        )
        output.set_value(self._current_position)

    def reset(self, reset_position: np.ndarray = None) -> None:
        """
        Reset controller state. Mirrors DiffusionPolicyController.reset().
        Called by SimulatedRealTableEnvironment between trials.
        """
        if reset_position is not None:
            self._current_position = np.asarray(reset_position, dtype=np.float64).copy()
        else:
            self._current_position = np.array(
                [self._initial_pusher_pose.x, self._initial_pusher_pose.y],
                dtype=np.float64,
            )
        self._received_reset_signal = True
