from typing import Any, List, Type, Optional

import torch as th
from torch.optim import Optimizer

from .base_meta_class import BaseMetaAlgorithm


class ReptileMetaRL(BaseMetaAlgorithm):
    """
    Reptile meta-RL implementation.

    It only needs to implement `meta_update`.
    """

    def __init__(
        self,
        *args,
        meta_lr: float,
        use_meta_optimizer: bool = False,
        meta_optimizer_cls: Type[Optimizer] = th.optim.Adam,
        ignored_layers: Optional[List[str]] = None,
        **kwargs
    ):
        """
        :param meta_lr:            Reptile step size epsilon.
        :param use_meta_optimizer: If True, use an optimizer over meta-parameters
                                   instead of directly adding meta_lr * delta.
        :param meta_optimizer_cls: Optimizer class to use in meta mode (default: Adam).
        :param ignored_layers:     List of parameter name prefixes to ignore in meta-update.
                                   These prefixes are matched against policy.named_parameters()
                                   (e.g. ["mlp_extractor.shared_net", "value_net"]).
        """
        super().__init__(*args, **kwargs)
        self.meta_lr = meta_lr
        self.use_meta_optimizer = use_meta_optimizer
        self.ignored_layer_prefixes = ignored_layers or []

        self.ignored_params = self._get_ignored_params(self.ignored_layer_prefixes)
        if self.ignored_params and verbose >= 1:
            print(f"[Reptile] Ignoring {len(self.ignored_params)} parameters in meta-update:")
            for name in sorted(self.ignored_params):
                print(f"  - {name}")

        if use_meta_optimizer:
            self.meta_optimizer = meta_optimizer_cls(
                self.policy.parameters(),
                lr=self.meta_lr,
            )
        else:
            self.meta_optimizer = None

    def _get_ignored_params(self, prefixes: List[str]) -> set[str]:
        """
        Map layer-name prefixes to full parameter names (ex. mlp_extractor.shared_net 
        or value_net)
        Raises ValueError if any prefix matches nothing.
        """
        if not prefixes:
            return set()

        all_param_names = [name for name, _ in self.policy.named_parameters()]

        ignored = {
            name
            for name in all_param_names
            if any(name.startswith(prefix) for prefix in prefixes)
        }

        unmatched_prefixes = [
            prefix
            for prefix in prefixes
            if not any(name.startswith(prefix) for name in all_param_names)
        ]

        if unmatched_prefixes:
            raise ValueError(
                f"Some ignored_layers prefixes did not match any parameters: "
                f"{unmatched_prefixes}\n"
                f"Available parameters include e.g.: {all_param_names[:10]}..."
            )

        return ignored

    def meta_update(self, task_models: List[Any]) -> None:
        """
        Batched Reptile update:

            φ ← φ + meta_lr * (1/B) Σ_i (φ_i - φ)

        where:
            φ = meta-policy parameters (self.meta_policy)
            φ_i = policy parameters of each task model after inner adaptation.

        Any parameter whose name is in self.ignored_params is skipped to be able to meta learn
        only a subset of the agent (ex. only value function)
        """
        original_params = self.meta_policy.state_dict()

        accumulated_deltas = {
            name: th.zeros_like(param, device=param.device)
            for name, param in original_params.items()
        }

        batch_size = len(task_models)
        for task_model in task_models:
            task_params = task_model.policy.state_dict()
            for name in original_params:
                if name in self.ignored_params:
                    continue
                accumulated_deltas[name] += (task_params[name] - original_params[name]) / batch_size

        if self.use_meta_optimizer and self.meta_optimizer is not None:
            self.meta_optimizer.zero_grad()

            for name, param in self.policy.named_parameters():
                if name in self.ignored_params:
                    continue
                if name in accumulated_deltas:
                    # -1 * accumulated_deltas = original_params - task_params
                    param.grad = -(accumulated_deltas[name]).detach()
            self.meta_optimizer.step()

        else:
            with th.no_grad():
                for name, param in self.meta_policy.named_parameters():
                    if name in self.ignored_params:
                        continue
                    if name in accumulated_deltas:
                        param.add_(self.meta_lr * accumulated_deltas[name])
