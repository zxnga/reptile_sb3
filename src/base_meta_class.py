from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type, Optional, Tuple

import numpy as np
import torch as th

from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.utils import get_device

from .task_generator import TaskGenerator
from .utils import load_weights_from_source, compute_updates


class BaseMetaAlgorithm(ABC):
    """
    Generic meta-RL base class that uses SB3 algo in the inner loop.

    It mirrors some of the SB3 BaseAlgorithm API:
      - .learn()   -> meta-training loop (outer loop)
      - .predict() -> forwards to the underlying SB3 meta model
      - .get_env() -> environment of the underlying meta model (first task env)
      - .set_random_seed() -> delegates to underlying SB3 algo

    But it does NOT inherit from BaseAlgorithm and it works over a TaskGenerator
    that produces many tasks/envs for the inner loop.
    """

    def __init__(
        self,
        tasks_generator_cls: Type[TaskGenerator],
        tasks_generator_params: Dict[str, Any],
        rl_algorithm: Union[th.nn.Module], # SB3 algo class, e.g. PPO/A2C
        rl_algo_kwargs: Dict[str, Any],
        inner_steps: int,
        outer_steps: int,
        task_batch_size: int = 1,
        inner_loop_params: Optional[Dict[str, Any]] = None,
        # save_frequency: int = 1,
        verbose: int = 0,
        device: th.device | str = "auto",
        tensorboard_logs: Optional[str] = './inner_loop_logs',
    ):
        assert hasattr(rl_algorithm, "learn"), "rl_algorithm must have a .learn() method (SB3)."
        assert task_batch_size > 0, f"task_batch_size must be > 0, got {task_batch_size}"

        if inner_loop_params is None:
            inner_loop_params = {}

        self.device = get_device(device)
        if verbose >= 1:
            print(f"Using {self.device} device")

        self.tasks_generator_cls = tasks_generator_cls
        self.tasks_generator_params = tasks_generator_params
        self.task_generator: TaskGenerator = self.instantiate_task_generator()

        self.rl_algorithm = rl_algorithm
        self.rl_algo_kwargs = rl_algo_kwargs

        self.inner_steps = inner_steps
        self.outer_steps = outer_steps
        self.inner_loop_params = inner_loop_params
        self.task_batch_size = task_batch_size
        
        self.verbose = verbose
        # self.save_frequency = save_frequency
        self.tensorboard_logs = tensorboard_logs

        first_env, _, _ = self.task_generator.get_task(0)
        self.meta_algo = self.instantiate_model(first_env)
        self.task_generator.reset_history()
        self.meta_policy = self.meta_algo.policy

        # self.updates_per_rollout, self.total_updates, self.n_rollouts = compute_updates(
        #     self.meta_algo, inner_steps
        # )
        # if verbose >=0 :
        #     # TODO chek and modify for off policy
        #     print(
        #         f"[BaseMetaRL] Gradient updates per inner loop: {self.total_updates:_} "
        #         f"({self.updates_per_rollout:_} per rollout * {self.n_rollouts:_} rollouts)"
        #     )
        #     print(
        #         f"[BaseMetaRL] Total env timesteps across outer loop: "
        #         f"{self.outer_steps * self.inner_steps:_}"
        #     )

    def instantiate_task_generator(self) -> TaskGenerator:
        return self.tasks_generator_cls(**self.tasks_generator_params)

    def instantiate_model(self, env: GymEnv | VecEnv):
        """
        Create an SB3 model given an environment.

        rl_algo_kwargs is expected to contain 'policy' and SB3 hyperparams:
          { "policy": "MlpPolicy", "n_steps": 128, "batch_size": 64, ... }
        """
        # maybe use .set_env instead that can be quicker (reset buffers for off policy, cast parameters to meta model params)
        policy = self.rl_algo_kwargs.get("policy", "MlpPolicy")
        algo_kwargs = {k: v for k, v in self.rl_algo_kwargs.items() if k != "policy"}
        return self.rl_algorithm(env=env, policy=policy, **algo_kwargs)

    def copy_meta_to_task(self, task_model):
        """
        Default: copy the meta_policy weights into the task model's policy.
        Subclasses can override for special behaviour (e.g., freezing some layers).
        """
        load_weights_from_source(self.meta_policy, task_model.policy, detach=True)

    def inner_adapt(self, task_model): #update_to_task
        """
        Default inner loop: SB3 learn() for a fixed number of env steps.
        """
        task_model.learn(self.inner_steps, **self.inner_loop_params)

    @abstractmethod
    def meta_update(self, task_models: List[Any]) -> None:
        """
        Algorithm-specific meta-update.

        Args:
            task_models: list of SB3 algorithms adapted on each task in the batch.
        """
        ...

    def learn(
        self,
        outer_steps: Optional[int] = None,
        reset_task_history_before_learning: bool = True,
    ) -> "BaseMetaRL":
        """
        Meta-training loop (outer loop).

        This mirrors SB3's `.learn()` name, but the "timesteps" dimension
        is split as:

            total_env_steps ≈ outer_steps * inner_steps

        Args:
            outer_steps: override number of meta-iterations; if None, use self.outer_steps.
            reset_task_history_before_learning: reset TaskGenerator history at the start.

        Returns:
            self (so you can write `meta_learner.learn(...).get_meta_policy()`).
        """
        if outer_steps is None:
            outer_steps = self.outer_steps

        if reset_task_history_before_learning:
            self.task_generator.reset_history()

        for outer in range(outer_steps):
            task_models = []

            task_batch = [
                self.task_generator.get_task(outer * self.task_batch_size + i)
                for i in range(self.task_batch_size)
            ]

            for env, task_info, first_occurrence in task_batch:
                task_model = self.instantiate_model(env)
                self.copy_meta_to_task(task_model)

                self.inner_adapt(task_model)
                task_models.append(task_model)

            self.meta_update(task_models)

        return self

    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Forward to the underlying SB3 meta_algo.predict().
        """
        return self.meta_algo.predict(
            observation,
            state=state,
            episode_start=episode_start,
            deterministic=deterministic,
        )

    def get_env(self) -> Optional[VecEnv]:
        """
        Underlying env of the meta_algo (the one from the first task).
        """
        return self.meta_algo.get_env()

    def set_random_seed(self, seed: Optional[int] = None) -> None:
        """
        Delegate seeding to the underlying SB3 algo.
        """
        if seed is None:
            return
        self.meta_algo.set_random_seed(seed)

    def save_meta_algo(self, path: str) -> None:
        """
        Save the SB3 meta model using its own .save().
        """
        self.meta_algo.save(path)

    @property
    def policy(self) -> BasePolicy:
        return self.meta_algo.policy
