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
        if self.ignored_params and self.verbose >= 1:
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
        meta_params = dict(self.meta_policy.named_parameters())

        accumulated_deltas = {
            name: th.zeros_like(param, device=param.device)
            for name, param in meta_params.items()
            if name not in self.ignored_params
        }
        
        batch_size = len(task_models)
        if batch_size == 0:
            # TODO: check what to do in that case
            return

        for task_model in task_models:
            task_params = dict(task_model.policy.named_parameters())

            for name, meta_param in meta_params.items():
                if name in self.ignored_params:
                    continue
                accumulated_deltas[name].add_(
                    (task_params[name].detach() - meta_param.detach()) / batch_size
                )

        if self.use_meta_optimizer and self.meta_optimizer is not None:
            self.meta_optimizer.zero_grad(set_to_none=True)

            for name, param in self.policy.named_parameters():
                if name in self.ignored_params:
                    continue
                # -1 * accumulated_deltas = original_params - task_params
                param.grad = -accumulated_deltas[name].detach()

            self.meta_optimizer.step()

        else:
            with th.no_grad():
                for name, param in self.meta_policy.named_parameters():
                    if name in self.ignored_params:
                        continue
                    param.add_(self.meta_lr * accumulated_deltas[name])
