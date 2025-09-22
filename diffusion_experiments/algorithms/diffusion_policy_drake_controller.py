from collections import deque

import cv2
import numpy as np
import torch
from pydrake.all import AbstractValue, Context, Image, LeafSystem, PixelType, RigidTransform, Value

from diffusion_experiments.utils.loading import load_policy


class DiffusionPolicyPlanarDrakeController(LeafSystem):
    def __init__(self, checkpoint_path, initial_planar_location, plant, policy_device="cpu"):
        super().__init__()

        self.policy = load_policy(checkpoint_path, load_normalzer_from_file=True)

        self._actions = deque([], maxlen=self.policy.n_action_steps)

        self._image_buffer = deque([], maxlen=self.policy.n_obs_steps)
        self._position_buffer = deque([], maxlen=self.policy.n_obs_steps)

        # declare ports for images and gripper positions
        self._rgb_in = self.DeclareAbstractInputPort(
            "rgb_in", AbstractValue.Make(np.ndarray(shape=(128, 128, 3), dtype=np.uint8))
        )
        self._position_in = self.DeclareVectorInputPort("position_in", 2)

        # declare output port for command that diff IK will track
        self.output = self.DeclareVectorOutputPort("planar_command_out", 2, self.DoCalcOutput)

        self.current_action = initial_planar_location

        self._device = policy_device
        self.policy.to(self._device)

        self._n_obs_steps = self.policy.n_obs_steps
        self._n_action_steps = self.policy.n_action_steps

    def DoCalcOutput(self, context: Context, output):
        time = context.get_time()

        while len(self._position_buffer) < self._n_obs_steps - 1:
            self._update_deques(context)
            output.set_value(self.current_action)

        self._update_deques(context)

        if len(self._actions) == 0:
            obs_dict = self._deque_to_dict()
            with torch.no_grad():
                predicted_actions = self.policy.predict_action(obs_dict)["action_pred"][0].cpu().numpy()
            self._actions.extend(predicted_actions[self._n_obs_steps : self._n_obs_steps + self._n_action_steps])

        self.current_action = self._actions.popleft()
        output.set_value(self.current_action)

    def _update_deques(self, context):
        rgb_image = self._rgb_in.Eval(context)
        position = self._position_in.Eval(context)

        rgb_image = cv2.resize(rgb_image[..., :3], (128, 128))

        self._image_buffer.append(rgb_image)
        self._position_buffer.append(position)

    def _deque_to_dict(self):
        data = {
            "obs": {"agent_pos": torch.from_numpy(np.stack(self._position_buffer)).unsqueeze(0).to(self._device)},
        }

        image_tensor = torch.cat(
            [torch.from_numpy(np.moveaxis(img, -1, -3) / 255.0) for img in self._image_buffer]
        ).reshape(1, self._n_obs_steps, 3, 128, 128)
        data["obs"]["overhead_camera"] = image_tensor.to(self._device)

        return data


class DiffusionPolicyDrakeController(LeafSystem):
    def __init__(self, checkpoint_path, initial_gripper_command, plant, policy_device="cpu"):
        super().__init__()

        self.policy = load_policy(checkpoint_path, load_normalzer_from_file=True)

        # Pop from this every time drake execuets an action
        # The moment this deque is empty, the policy predicts the next set of actions
        self._actions = deque([], maxlen=self.policy.n_action_steps)

        # The deque only ever holds the observation length number of elements
        self._overhead_image_buffer = deque([], maxlen=self.policy.n_obs_steps)
        self._wrist_image_buffer = deque([], maxlen=self.policy.n_obs_steps)
        self._gripper_buffer = deque([], maxlen=self.policy.n_obs_steps)

        self._gripper_policy_buffer = deque([], maxlen=8)
        for _ in range(self.policy.n_obs_steps):
            self._gripper_policy_buffer.append(0.107)

        # # declare input port for gripper pose
        # self._gripper_pose_in = self.DeclareAbstractInputPort(
        #     "gripper_pose_in", AbstractValue.Make(RigidTransform())
        # )
        # declare port for gripper pose
        self._gripper_body_index = plant.GetBodyByName("body").index()
        self._body_poses_in = self.DeclareAbstractInputPort("body_poses", AbstractValue.Make([RigidTransform()]))

        # input for wsg fingers
        self._gripper_fingers_in = self.DeclareVectorInputPort("wsg_state_in", 4)  # fix this

        # declare input port for the cameras
        self._overhead_camera_in = self.DeclareAbstractInputPort(
            name="overhead_camera_in",
            model_value=Value(Image[PixelType.kRgba8U]()),
        )
        self._wrist_camera_in = self.DeclareAbstractInputPort(
            name="wrist_camera_in",
            model_value=Value(Image[PixelType.kRgba8U]()),
        )

        self.output = self.DeclareVectorOutputPort("gripper_command_out", 13, self.DoCalcOutput)

        self._device = policy_device
        self.policy.to(self._device)

        self.current_action = initial_gripper_command

    def DoCalcOutput(self, context: Context, output):
        # wait time at the start
        while len(self._gripper_buffer) < self.policy.n_obs_steps - 1:
            self._update_deques(context)
            output.set_value(self.current_action)

        self._update_deques(context)

        if len(self._actions) == 0:
            obs_dict = self._deque_to_dict()
            with torch.no_grad():
                predicted_actions = (
                    self.policy.predict_action(obs_dict, use_DDIM=True)["action_pred"][0].cpu().numpy()
                )  # action contains the entire predicted horizon; action_pred is just index 2 to 10 (i.e. action horizon)

            self._actions.extend(predicted_actions[self.policy.n_obs_steps : self.policy.n_obs_steps + 8])

            for this_action in predicted_actions[self.policy.n_obs_steps : self.policy.n_obs_steps + 8]:
                self._gripper_policy_buffer.append(this_action[12])

        self.current_action = self._actions.popleft()
        output.set_value(self.current_action)

    def _update_deques(self, context):
        overhead_image = self._overhead_camera_in.Eval(context).data
        wrist_image = self._wrist_camera_in.Eval(context).data

        gripper_pose = self._body_poses_in.Eval(context)[int(self._gripper_body_index)]

        gripper_fingers = self._gripper_fingers_in.Eval(context)[:2]

        # gripper_pose = self._gripper_pose_in.Eval(context)
        # gripper_fingers = self._gripper_fingers_in.Eval(context)

        overhead_image = cv2.resize(overhead_image[..., :3], (128, 128))
        wrist_image = cv2.resize(wrist_image[..., :3], (128, 128))

        gripper_pose_translation = gripper_pose.translation().flatten()
        gripper_pose_rotation = gripper_pose.rotation().matrix().flatten()
        gripper_command = np.sum(np.abs(gripper_fingers))
        gripper_command = self._gripper_policy_buffer.popleft() if len(self._gripper_policy_buffer) > 0 else 0.107
        gripper_command = np.hstack([gripper_pose_translation, gripper_pose_rotation, gripper_command])

        self._overhead_image_buffer.append(overhead_image)
        self._wrist_image_buffer.append(wrist_image)
        self._gripper_buffer.append(gripper_command)

    def _deque_to_dict(self):
        """
        Reformat data into a dictionary to input to policy. The names and shapes of the keys used in this dictionary are
        defined in the training config file under `shape_meta`.

        Note: currently, the model does not condition on a target; so we just add a zero target.
        """

        data = {
            "obs": {
                "agent_pos": torch.from_numpy(np.stack(self._gripper_buffer)).unsqueeze(0).to(self._device),
            },
            "target": torch.zeros((1, 3)).to(self._device),
        }

        # add addition of target optional here
        overhead_tensor = (
            torch.cat(
                [
                    torch.from_numpy(np.moveaxis(img, -1, -3) / 255.0)  # convert to [0,1]
                    for img in self._overhead_image_buffer
                ],
                dim=0,
            )
            .reshape(1, self.policy.n_obs_steps, 3, 128, 128)
            .to(self._device)
        )
        # noise = torch.randn_like(overhead_tensor, device=self._device) * 0.0
        # overhead_tensor += noise
        # noise_length = 12
        # noise = torch.zeros_like(overhead_tensor, device=self._device)
        # noise[:, :noise_length, ...] = torch.randn((1, noise_length, 3, 128, 128), device=self._device) * 0.8
        # overhead_tensor += noise
        data["obs"]["overhead_camera"] = overhead_tensor

        wrist_tensor = (
            torch.cat(
                [
                    torch.from_numpy(np.moveaxis(img, -1, -3) / 255.0)  # convert to [0,1]
                    for img in self._wrist_image_buffer
                ],
                dim=0,
            )
            .reshape(1, self.policy.n_obs_steps, 3, 128, 128)
            .to(self._device)
        )
        # noise = torch.randn_like(wrist_tensor, device=self._device) * 0.1
        # wrist_tensor = noise
        # noise = torch.zeros_like(wrist_tensor, device=self._device)
        # noise[:, :noise_length, ...] = torch.randn((1, noise_length, 3, 128, 128), device=self._device) *0.8
        # wrist_tensor += noise

        data["obs"]["wrist_camera"] = wrist_tensor

        return data
