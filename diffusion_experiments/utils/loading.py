import os
import pickle
from typing import Dict, List

import dill
import hydra
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

INFER_FROZEN_POLICY = False


def get_subdirectories(directory):
    subdirs = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    return subdirs


def load_npz(file: str):
    data = np.load(file)
    data_for_np = [data[key] for key in data]
    return np.vstack(data_for_np)


def load_policy(policy_name: str, dataset_zarr: str = None, load_normalzer_from_file: bool = False):
    """
    Main function for loading a policy.
    """
    payload = torch.load(open(policy_name, "rb"), pickle_module=dill)

    model_cfg = payload["cfg"]

    if INFER_FROZEN_POLICY:
        OmegaConf.set_struct(model_cfg.policy, False)
        model_cfg.policy.inference_loading = True
        OmegaConf.set_struct(model_cfg.policy, True)

    model_workspace_cls = hydra.utils.get_class(model_cfg._target_)
    model_workspace = model_workspace_cls(model_cfg)
    model_workspace.load_payload(payload, exclude_keys=None, include_keys=None)  # loads model weights?

    if load_normalzer_from_file:  # load normalizer if its saved
        normalizer_path = os.path.join(os.path.dirname(os.path.dirname(policy_name)), "normalizer.pt")
        normalizer = torch.load(normalizer_path, weights_only=False)
    else:
        if dataset_zarr is not None:
            model_cfg.task.dataset.zarr_path = dataset_zarr
        dataset = hydra.utils.instantiate(model_cfg.task.dataset)
        normalizer = dataset.get_normalizer()

    policy = model_workspace.model
    policy.set_normalizer(normalizer)
    if model_cfg.training.use_ema:
        policy = model_workspace.ema_model
    policy.set_normalizer(normalizer)

    policy.eval()
    return policy


def load_policy_checkpoints(data_to_checkpoint_dict: Dict):
    data_dirs = list(data_to_checkpoint_dict.keys())

    data_dir_to_policy = {}
    data_dir_to_context_length = {}

    for data_dir in data_dirs:
        list_of_policies = []
        list_of_context_lengths = []

        for policy_name in data_to_checkpoint_dict[data_dir]:
            policy = load_policy(policy_name, dataset_zarr=f"{data_dir}.zarr")
            list_of_policies.append(policy)

            list_of_context_lengths.append(int(policy_name.split("/")[-3].split("_")[-2]))

        data_dir_to_policy[data_dir] = list_of_policies
        data_dir_to_context_length[data_dir] = list_of_context_lengths

    return data_dir_to_policy, data_dir_to_context_length


def load_only_policy_checkpoints(data_to_checkpoint_dict: Dict):
    data_dirs = list(data_to_checkpoint_dict.keys())

    data_dir_to_policy = {}

    for data_dir in data_dirs:
        list_of_policies = []

        for policy_name in data_to_checkpoint_dict[data_dir]:
            policy = load_policy(policy_name, dataset_zarr=f"{data_dir}.zarr")
            list_of_policies.append(policy)

        data_dir_to_policy[data_dir] = list_of_policies

    return data_dir_to_policy


def load_gt_system_controller(data_dirs: List):
    dict_of_system_controller = {}

    for data_dir in data_dirs:
        with open(f"{data_dir}/system.pkl", "rb") as f:
            linear_system = pickle.load(f)

        with open(f"{data_dir}/controller.pkl", "rb") as f:
            controller = pickle.load(f)

        with open(f"{data_dir}/metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        dict_of_system_controller[data_dir] = (controller.system, controller, metadata)

    return dict_of_system_controller


class DictToObject:
    def __init__(self, dictionary: Dict):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, DictToObject(value))
            else:
                setattr(self, key, value)


def load_dict_yaml(file_path: str):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return DictToObject(data)
