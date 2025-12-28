from typing import Union, Dict, List, Tuple

import math
import os
import torch
from torch.nn import Module

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



def compute_updates(model, inner_steps: int) -> Tuple[int, int, int]:
    """
    Params:
      model: instanciated SB3 model
      inner_steps: Reptile inner loop steps
    Returns:
      updates_per_rollout: number of gradient updates SB3 does per rollout
      total_updates:      number of gradient updates over `inner_steps` env steps
      n_rollouts: number of rollouts over inner_steps

      #TODO: modify for off_policy updates as well
    """
    n_steps    = model.n_steps
    n_envs     = model.env.num_envs   # VecEnv always has this attribute
    batch_size = model.batch_size
    n_epochs = getattr(model, "n_epochs", 1)

    # only full mini-batches are used per epoch
    updates_per_rollout = (n_steps * n_envs) // batch_size * n_epochs

    # how many rollouts to reach `inner_steps` 
    n_rollouts = math.ceil(inner_steps / n_steps)

    total_updates = updates_per_rollout * n_rollouts
    return updates_per_rollout, total_updates, n_rollouts