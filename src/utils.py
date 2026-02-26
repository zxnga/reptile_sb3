from typing import Union, Dict, List, Tuple, Any
from enum import Enum

import math
import os
import warnings
import torch
from torch.nn import Module

try:
    from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
    from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
    from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
except Exception:
    TrainFreq = None  # type: ignore[assignment]
    TrainFrequencyUnit = None  # type: ignore[assignment]
    OnPolicyAlgorithm = None  # type: ignore[assignment]
    OffPolicyAlgorithm = None  # type: ignore[assignment]

def load_weights_from_source(
    source: Union[Module, Dict[str, torch.Tensor]],
    target_model: Module,
    exclude_layers: List[str] = [],
    detach: bool = True
) -> None:
    """
        Load weights from a source model or dictionary into a target model, with optional exclusion of specific layers.
        Args:
            source (Union[torch.nn.Module, Dict[str, torch.Tensor]]): The source of the weights, PyTorch model 
                or a dictionary containing pre-extracted weights.
            target_model (torch.nn.Module): The target model to load the weights into.
            exclude_layers (List[str], optional): A list of layer names to exclude from loading. Defaults to an empty list.
            detach (bool, optional): detach the weights to avoid sharing gradients between models. Defaults to True.

        Returns:
            None
    """
    if isinstance(source, torch.nn.Module):
        source_state_dict = source.state_dict()
    elif isinstance(source, dict):
        source_state_dict = source
    else:
        raise ValueError("Source must be either an nn.Module or a dictionary of weights.")

    target_state_dict = target_model.state_dict()
    if detach:
        filtered_state_dict = {
            k: v.clone().detach()
            for k, v in source_state_dict.items()
            if not any(k.startswith(layer if layer.endswith('.') else layer + '.') 
                    for layer in exclude_layers)
        }
    else:
        filtered_state_dict = {
            k: v.clone()
            for k, v in source_state_dict.items()
            if not any(k.startswith(layer if layer.endswith('.') else layer + '.') 
                    for layer in exclude_layers)
        }
    for name, param in filtered_state_dict.items():
        if name in target_state_dict and target_state_dict[name].shape == param.shape:
            target_state_dict[name].copy_(param)
        else:
            print(f"Skipping layer: {name} (shape mismatch or missing in target)")
    target_model.load_state_dict(target_state_dict)



def _is_on_policy(model: Any) -> bool:
    return OnPolicyAlgorithm is not None and isinstance(model, OnPolicyAlgorithm)


def _is_off_policy(model: Any) -> bool:
    return OffPolicyAlgorithm is not None and isinstance(model, OffPolicyAlgorithm)


def _unit_to_str(unit: Any) -> str:
    if isinstance(unit, str):
        return unit.lower()
    if isinstance(unit, Enum):
        return str(unit.value).lower()
    return str(unit).lower()


def _normalize_train_freq(train_freq: Any):
    if TrainFreq is not None and isinstance(train_freq, TrainFreq):
        return train_freq

    if isinstance(train_freq, tuple):
        if len(train_freq) != 2:
            raise ValueError(f"`train_freq` tuple must be length 2, got {train_freq}.")
        frequency, unit = train_freq
    else:
        frequency, unit = train_freq, "step"

    if not isinstance(frequency, int):
        raise ValueError(f"`train_freq` frequency must be int, got {type(frequency).__name__}.")

    unit_str = _unit_to_str(unit)
    if unit_str not in ("step", "episode"):
        raise ValueError(f"`train_freq` unit must be 'step' or 'episode', got {unit}.")

    if TrainFreq is not None and TrainFrequencyUnit is not None:
        normalized_unit = TrainFrequencyUnit.STEP if unit_str == "step" else TrainFrequencyUnit.EPISODE
        return TrainFreq(frequency=frequency, unit=normalized_unit)

    return frequency, unit_str


def _on_policy_counts(model: Any, inner_steps: int) -> Tuple[int, int, int]:
    n_envs = model.env.num_envs
    rollout_env_steps = model.n_steps * n_envs
    n_rollouts = math.ceil(inner_steps / rollout_env_steps)

    if hasattr(model, "n_epochs"):
        n_epochs = model.n_epochs
        batch_size = model.batch_size
        n_minibatches = math.ceil(rollout_env_steps / batch_size)
        updates_per_rollout = n_epochs * n_minibatches
    else:
        updates_per_rollout = 1

    total_updates = updates_per_rollout * n_rollouts
    return updates_per_rollout, total_updates, n_rollouts


def _off_policy_counts(model: Any, inner_steps: int, strict: bool = False, warn: bool = True) -> Tuple[int, int, int]:
    n_envs = model.env.num_envs
    train_freq = _normalize_train_freq(model.train_freq)

    if TrainFreq is not None and isinstance(train_freq, TrainFreq):
        frequency = train_freq.frequency
        unit_str = _unit_to_str(train_freq.unit)
    else:
        frequency, unit_str = train_freq

    if unit_str == "episode":
        msg = (
            "Off-policy exact update count with train_freq unit='episode' is not deterministic "
            "because it depends on stochastic episode lengths. "
            "Use step-based train_freq, e.g. train_freq=(k, 'step')."
        )
        if strict:
            raise ValueError(msg)
        if warn:
            warnings.warn(msg, RuntimeWarning)
        return 0, 0, 0

    rollout_env_steps = frequency * n_envs
    n_rollouts = math.ceil(inner_steps / rollout_env_steps)

    learning_starts = getattr(model, "learning_starts", 0)
    skipped_rollouts = learning_starts // rollout_env_steps
    train_rollouts = max(0, n_rollouts - skipped_rollouts)

    gradient_steps = getattr(model, "gradient_steps", 1)
    if gradient_steps >= 0:
        updates_per_rollout = gradient_steps
    else:
        # SB3 uses rollout.episode_timesteps when gradient_steps=-1,
        # which corresponds to collected environment timesteps.
        updates_per_rollout = rollout_env_steps

    total_updates = train_rollouts * updates_per_rollout
    return updates_per_rollout, total_updates, n_rollouts


def compute_updates(model, inner_steps: int, strict: bool = True, warn: bool = True) -> Tuple[int, int, int]:
    """
    Params:
      model: instantiated SB3 model
      inner_steps: Reptile inner loop steps (SB3 total_timesteps)
      strict: raise for unsupported exact cases (off-policy episodic train_freq)
      warn: emit warnings for approximate/unsupported branches if strict=False
    Returns:
      updates_per_rollout: number of gradient updates SB3 does per rollout
      total_updates:      number of gradient updates over `inner_steps` env steps
      n_rollouts: number of rollouts over inner_steps
    """
    if inner_steps <= 0:
        raise ValueError(f"`inner_steps` must be > 0, got {inner_steps}.")

    if _is_on_policy(model):
        return _on_policy_counts(model, inner_steps)

    if _is_off_policy(model):
        return _off_policy_counts(model, inner_steps, strict=strict, warn=warn)

    raise ValueError(
        "Unsupported model type for compute_updates(). "
        "Expected an SB3 on-policy or off-policy algorithm instance."
    )
