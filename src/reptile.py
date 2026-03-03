from typing import Any, Dict, List, Optional, Type

import torch as th

from .base_meta_class import BaseMetaAlgorithm
from .task_generator import TaskGenerator
from .utils import LRSchedule


class ReptileMetaRL(BaseMetaAlgorithm):
    """
    Reptile meta-RL implementation.

    It only needs to implement `meta_update`.
    """

    def __init__(
        self,
        tasks_generator_cls: Type[TaskGenerator],
        tasks_generator_params: Dict[str, Any],
        rl_algorithm: th.nn.Module,
        rl_algo_kwargs: Dict[str, Any],
        inner_steps: int,
        outer_steps: int,
        meta_lr: LRSchedule,
        use_meta_optimizer: bool = False,
        meta_optimizer_cls: Type[th.optim.Optimizer] = th.optim.Adam,
        meta_optimizer_kwargs: Optional[Dict[str, Any]] = None,
        task_batch_size: int = 1,
        inner_loop_params: Optional[Dict[str, Any]] = None,
        ignored_layers: Optional[List[str]] = None,
        verbose: int = 0,
        device: th.device | str = "auto",
        tensorboard_logs: Optional[str] = "./inner_loop_logs",
        **kwargs,
    ):
        super().__init__(
            tasks_generator_cls=tasks_generator_cls,
            tasks_generator_params=tasks_generator_params,
            rl_algorithm=rl_algorithm,
            rl_algo_kwargs=rl_algo_kwargs,
            inner_steps=inner_steps,
            outer_steps=outer_steps,
            meta_lr=meta_lr,
            use_meta_optimizer=use_meta_optimizer,
            meta_optimizer_cls=meta_optimizer_cls,
            meta_optimizer_kwargs=meta_optimizer_kwargs,
            task_batch_size=task_batch_size,
            inner_loop_params=inner_loop_params,
            ignored_layers=ignored_layers,
            verbose=verbose,
            device=device,
            tensorboard_logs=tensorboard_logs,
            **kwargs,
        )

    def meta_update(self, task_models: List[Any], outer_step: int) -> None:
        """
        Batched Reptile update:

            φ ← φ + meta_lr * (1/B) Σ_i (φ_i - φ)

        where:
            φ = meta-policy parameters (self.meta_policy)
            φ_i = policy parameters of each task model after inner adaptation.

        Any parameter whose name is in self.ignored_params is skipped to be able to meta learn
        only a subset of the agent (ex. only value function)

        Args:
            task_models: list of SB3 algorithms adapted on each task in the batch.
            outer_step: current outer loop step, needed to compute lr.
        """
        meta_lr = self.get_meta_lr(outer_step)
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
            self.sync_meta_optimizer_lr(self.meta_optimizer, meta_lr)
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
                    param.add_(meta_lr * accumulated_deltas[name])
