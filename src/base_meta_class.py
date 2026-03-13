from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type, Optional, Tuple, Literal
from collections.abc import Iterable
import time
import os
import json
from datetime import datetime

import numpy as np
import torch as th
from torch.optim import Optimizer

from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.utils import get_device, configure_logger

from .task_generator import TaskGenerator
from .utils import (
    load_weights_from_source,
    compute_updates,
    to_json_safe,
    sanitize_name,
    build_checkpoint_run_dir,
)
from .utils import LRSchedule, normalize_lr_schedule

#TODO: add a testing rollout loop inside the .learn to tracl progress of meta-model + log

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
        rl_algorithm: th.nn.Module, # SB3 algo class, e.g. PPO/A2C
        rl_algo_kwargs: Dict[str, Any],
        inner_steps: int,
        outer_steps: int,
        meta_lr: LRSchedule,
        use_meta_optimizer: bool = False,
        meta_optimizer_cls: Type[Optimizer] = th.optim.Adam,
        meta_optimizer_kwargs: Optional[Dict[str, Any]] = None,
        task_batch_size: int = 1,
        inner_loop_params: Optional[Dict[str, Any]] = None,
        ignored_layers: Optional[List[str]] = None,
        ignore_optimizer_params: bool = False,
        verbose: int = 0,
        device: th.device | str = "auto",
        tensorboard_logs: Optional[str] = './meta-logs',
    ):
        assert hasattr(rl_algorithm, "learn"), "rl_algorithm must have a .learn() method (SB3)."
        assert task_batch_size > 0, f"task_batch_size must be > 0, got {task_batch_size}"

        self.device = get_device(device)
        self.verbose = verbose
        if self.verbose >= 1:
            print(f"[BaseMetaRL] Using {self.device} device across all loops.")

        self.tasks_generator_cls = tasks_generator_cls
        self.tasks_generator_params = tasks_generator_params
        self.rl_algorithm = rl_algorithm
        self.rl_algo_kwargs = rl_algo_kwargs

        self.inner_steps = inner_steps
        self.outer_steps = outer_steps
        self.inner_loop_params = inner_loop_params if inner_loop_params is not None else {}
        self.task_batch_size = task_batch_size
        self.ignored_layer_prefixes = ignored_layers or []
        self.ignored_params: set[str] = set()
        self.ignore_optimizer_params = ignore_optimizer_params

        self.use_meta_optimizer = use_meta_optimizer
        self.meta_optimizer_cls = meta_optimizer_cls
        self.meta_optimizer_kwargs = dict(meta_optimizer_kwargs or {})
        self.meta_lr_schedule = normalize_lr_schedule(meta_lr)
        self.meta_lr = self.meta_lr_schedule
        self.is_constant_meta_lr = isinstance(meta_lr, (int, float))
        self.current_meta_lr = float(meta_lr) if self.is_constant_meta_lr else None
        self._last_optimizer_meta_lr: Optional[float] = self.current_meta_lr

        self.tensorboard_logs = tensorboard_logs
        self.meta_logger: Optional[Logger] = None

        first_env = self._init_task_generator_and_bootstrap_env()
        self._init_meta_model(first_env)
        self._init_ignored_params()
        self.meta_optimizer = self._build_meta_optimizer()
        self._init_budget_counters()
        self._log_startup_summary()

    def _init_tensorboard_writer(
        self,
        tb_log_name: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> None:
        if self.tensorboard_logs is None:
            return
        run_name = tb_log_name or self.__class__.__name__
        try:
            self.meta_logger = configure_logger(
                verbose=self.verbose,
                tensorboard_log=self.tensorboard_logs,
                tb_log_name=run_name,
                reset_num_timesteps=reset_num_timesteps,
            )
        except ImportError:
            self.meta_logger = None
            if self.verbose >= 1:
                print("[BaseMetaRL] TensorBoard unavailable (tensorboard not installed).")
            return
        if self.verbose >= 1:
            print(
                f"[BaseMetaRL] TensorBoard outer-loop logs: {self.tensorboard_logs} "
                f"(run={run_name})"
            )

    def _init_task_generator_and_bootstrap_env(self) -> GymEnv | VecEnv:
        self.task_generator = self.instantiate_task_generator()
        first_env, _, _ = self.task_generator.get_task(0)
        self.task_generator.reset_history()
        return first_env

    def _init_meta_model(self, first_env: GymEnv | VecEnv) -> None:
        self.meta_algo = self.instantiate_model(first_env)
        self.meta_policy = self.meta_algo.policy

    def _init_ignored_params(self) -> None:
        self.ignored_params = self._get_ignored_params(self.ignored_layer_prefixes)
        if self.ignored_params and self.verbose >= 1:
            print(f"[BaseMetaRL] Ignoring {len(self.ignored_params)} parameters in meta-update:")
            for name in sorted(self.ignored_params):
                print(f"  - {name}")

    def _init_budget_counters(self) -> None:
        self.updates_per_rollout = None
        self.total_updates = None
        self.n_rollouts = None
        self.total_updates_per_outer_all_tasks = None
        self.total_updates_across_outer_all_tasks = None

        self.total_env_steps_per_outer = self.inner_steps * self.task_batch_size
        self.total_env_steps_across_outer = self.outer_steps * self.total_env_steps_per_outer

        try:
            self.updates_per_rollout, self.total_updates, self.n_rollouts = compute_updates(
                self.meta_algo, self.inner_steps
            )
            self.total_updates_per_outer_all_tasks = self.total_updates * self.task_batch_size
            self.total_updates_across_outer_all_tasks = (
                self.total_updates_per_outer_all_tasks * self.outer_steps
            )
        except ValueError as exc:
            if self.verbose >= 1:
                print(f"[BaseMetaRL] Could not compute update budget exactly: {exc}")

    def _log_startup_summary(self) -> None:
        if self.verbose < 1:
            return

        if self.total_updates is not None:
            print(
                f"[BaseMetaRL] Maximum theoretical number of steps and gradient updates:"
            )
            print(
                f"    - Gradient updates per inner loop (per task): {self.total_updates:_} "
                f"({self.updates_per_rollout:_} per rollout * {self.n_rollouts:_} rollouts)"
            )
            print(
                f"    - Gradient updates per outer step (all tasks): "
                f"{self.total_updates_per_outer_all_tasks:_} "
                f"({self.total_updates:_} per task * {self.task_batch_size:_} tasks per batch)"
            )
            print(
                f"    - Total inner loop gradient updates across all outer steps: "
                f"{self.total_updates_across_outer_all_tasks:_} "
                f"({self.total_updates_per_outer_all_tasks} updates per outer step * {self.outer_steps} outer steps)\n"
            )

        print(
            f"    - Env timesteps per outer step (all tasks): "
            f"{self.total_env_steps_per_outer:_} "
            f"({self.inner_steps:_} inner_steps * {self.task_batch_size:_} tasks)"
        )
        print(
            f"    - Total env timesteps across outer loop (all tasks): "
            f"{self.total_env_steps_across_outer:_} "
            f"({self.inner_steps:_} inner_steps * {self.task_batch_size:_} tasks per batch * "
            f"{self.outer_steps:_} outer_steps)\n"
        )
        print(
            f"    - Meta-model updates across outer loop: "
            f"{self.outer_steps:_} "
        )

    def _log_outer_step_metrics(
        self,
        outer_step: int,
        inner_total_seconds: float,
        meta_update_seconds: float,
    ) -> None:
        if self.meta_logger is None:
            return

        global_outer = outer_step + 1
        outer_step_seconds = inner_total_seconds + meta_update_seconds

        self.meta_logger.record("meta/outer_step", global_outer)
        if self.current_meta_lr is not None:
            self.meta_logger.record("meta/meta_lr", float(self.current_meta_lr))
        self.meta_logger.record("meta/task_batch_size", self.task_batch_size)
        self.meta_logger.record("meta/inner_steps", self.inner_steps)
        self.meta_logger.record(
            "meta/total_env_steps_seen",
            global_outer * self.inner_steps * self.task_batch_size,
        )

        if self.updates_per_rollout is not None and self.total_updates is not None:
            total_updates_all_tasks_per_outer = self.total_updates * self.task_batch_size
            self.meta_logger.record(
                "meta/updates_per_rollout_per_task",
                self.updates_per_rollout,
            )
            self.meta_logger.record(
                "meta/total_updates_per_task_per_outer",
                self.total_updates,
            )
            self.meta_logger.record(
                "meta/total_updates_all_tasks_per_outer",
                total_updates_all_tasks_per_outer,
            )
            self.meta_logger.record(
                "meta/total_updates_seen_all_tasks",
                total_updates_all_tasks_per_outer * global_outer,
            )

        self.meta_logger.record("time/outer_step_seconds", outer_step_seconds)
        self.meta_logger.record("time/inner_total_seconds", inner_total_seconds)
        self.meta_logger.record("time/meta_update_seconds", meta_update_seconds)
        self.meta_logger.dump(step=global_outer)

    def _save_checkpoint_with_metadata(
        self,
        checkpoint_path: str,
        outer_step_completed: int,
        task_seed_mode: str,
        save_metadata: bool = True,
    ) -> None:
        self.save_meta_algo(checkpoint_path)

        metadata = {
            "algorithm_class": self.__class__.__name__,
            "outer_step_completed": outer_step_completed,
            "outer_steps_target": self.outer_steps,
            "inner_steps": self.inner_steps,
            "task_batch_size": self.task_batch_size,
            "task_seed_mode": task_seed_mode,
            "meta_lr_current": self.current_meta_lr,
            "use_meta_optimizer": self.use_meta_optimizer,
            "meta_optimizer_cls": getattr(self.meta_optimizer_cls, "__name__", str(self.meta_optimizer_cls)),
            "meta_optimizer_kwargs": to_json_safe(self.meta_optimizer_kwargs),
            "ignored_layers": to_json_safe(self.ignored_layer_prefixes),
            "ignore_optimizer_params": self.ignore_optimizer_params,
            "rl_algorithm": getattr(self.rl_algorithm, "__name__", str(self.rl_algorithm)),
            "rl_algo_kwargs": to_json_safe(self.rl_algo_kwargs),
            "tasks_generator_cls": getattr(self.tasks_generator_cls, "__name__", str(self.tasks_generator_cls)),
            "tasks_generator_params": to_json_safe(self.tasks_generator_params),
        }
        json_path = f"{checkpoint_path}.json"
        with open(json_path, "w", encoding="utf-8") as fp:
            json.dump(metadata, fp, indent=2, sort_keys=True)

    def instantiate_task_generator(self) -> TaskGenerator:
        return self.tasks_generator_cls(**self.tasks_generator_params)

    def instantiate_model(
        self,
        env: GymEnv | VecEnv,
        rl_algo_kwargs: Optional[Dict[str, Any]] = None #used to modify inner-loop params when meta-testing
    ):
        """
        Create an SB3 model given an environment.

        rl_algo_kwargs is expected to contain 'policy' and SB3 hyperparams:
          { "policy": "MlpPolicy", "n_steps": 128, "batch_size": 64, ... }
        """
        # TODO: maybe use .set_env instead that can be quicker (reset buffers for off policy, cast parameters to meta model params)

        params = rl_algo_kwargs if rl_algo_kwargs is not None else self.rl_algo_kwargs
        policy = params.get("policy", "MlpPolicy")
        device = params.get("device", self.device)

        algo_kwargs = {
            k: v
            for k, v in params.items()
            if k not in {"policy", "device"}
        }
        return self.rl_algorithm(
            env=env,
            policy=policy,
            device=device,
            **algo_kwargs
        )

    def get_meta_optimizer_params(self) -> Iterable[th.nn.Parameter]:
        """
        Parameters optimized by the outer-loop optimizer.
        """
        if self.ignored_params and self.ignore_optimizer_params:
            return (
                param
                for name, param in self.meta_policy.named_parameters()
                if name not in self.ignored_params
            )
        return self.meta_policy.parameters()

    def _build_meta_optimizer(self) -> Optional[Optimizer]:
        if not self.use_meta_optimizer:
            return None
        lr0 = self.get_meta_lr(0)
        optimizer = self.meta_optimizer_cls(
            self.get_meta_optimizer_params(),
            lr=lr0,
            **self.meta_optimizer_kwargs,
        )
        self._last_optimizer_meta_lr = lr0
        return optimizer

    def _get_ignored_params(self, prefixes: List[str]) -> set[str]:
        """
        Map parameter-name prefixes to full parameter names and validate matches.
        """
        if not prefixes:
            return set()

        all_param_names = [name for name, _ in self.meta_policy.named_parameters()]
        ignored = {
            name
            for name in all_param_names
            if any(name.startswith(prefix) for prefix in prefixes)
        }

        if self.verbose >= 1:
            print("[BaseMetaRL] Filtering out ignored_params in meta-model.")
            if self.ignore_optimizer_params:
                print("[BaseMetaRL] Filtering out ignored_params in optimizer.")                

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

    def _inner_adapt_with_optional_logging(
        self,
        task_model: Any,
        *,
        inner_tb_log_name: Optional[str],
        inner_tb_log_root: Optional[str],
    ) -> None:
        """
        Run inner adaptation and, when possible, attach SB3-style tensorboard
        naming per task.
        """
        if inner_tb_log_name is None:
            self.inner_adapt(task_model)
            return

        if inner_tb_log_root is not None and hasattr(task_model, "tensorboard_log"):
            setattr(task_model, "tensorboard_log", inner_tb_log_root)

        learn_kwargs = dict(self.inner_loop_params)
        learn_kwargs.setdefault("tb_log_name", inner_tb_log_name)
        learn_kwargs.setdefault("reset_num_timesteps", False)
        
        task_model.learn(self.inner_steps, **learn_kwargs)

    @abstractmethod
    def meta_update(self, task_models: List[Any], outer_step: int) -> None:
        """
        Algorithm-specific meta-update.

        Args:
            task_models: list of SB3 algorithms adapted on each task in the batch.
            outer_step: current outer loop step, needed to compute lr.
        """
        ...

    def _setup_learn(
        self,
        *,
        outer_steps: Optional[int],
        reset_task_history_before_learning: bool,
        task_seed_mode: Literal["generator", "meta_step"],
        tb_log_name: Optional[str],
        log_inner_tasks: bool,
        inner_tb_log_root: Optional[str],
        save_freq: int,
        save_path: Optional[str],
        save_final: bool,
    ) -> Dict[str, Any]:
        if outer_steps is not None:
            if self.verbose >= 1:
                print(
                    f"[BaseMetaRL] Overriding class outer_steps ({self.outer_steps}) "
                    f"with new value for this run: {outer_steps}."
                )
            self.outer_steps = outer_steps

        if reset_task_history_before_learning:
            self.task_generator.reset_history()

        if task_seed_mode not in ("generator", "meta_step"):
            raise ValueError(
                f"Unknown task_seed_mode={task_seed_mode!r}. "
                "Expected 'generator' or 'meta_step'."
            )
        if save_freq < 0:
            raise ValueError(f"`save_freq` must be >= 0, got {save_freq}.")

        if self.verbose >= 1:
            print(f"[BaseMetaRL] Task seed mode: {task_seed_mode}.")
            if log_inner_tasks:
                print(
                    f"[BaseMetaRL] Inner-loop task logging enabled "
                    f"(inner_tb_log_root={inner_tb_log_root})."
                )

        checkpoint_dir = save_path or "./checkpoints"
        checkpoint_run_dir = None
        if save_freq > 0 or save_final:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_run_dir = build_checkpoint_run_dir(
                base_path=checkpoint_dir,
                run_name=tb_log_name or self.__class__.__name__,
            )
            if self.verbose >= 1:
                print(f"[BaseMetaRL] Checkpoints for this run: {checkpoint_run_dir}")

        base_run_name = tb_log_name or self.__class__.__name__
        safe_run_name = sanitize_name(base_run_name)
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_stem = f"{safe_run_name}_{run_timestamp}"

        inner_run_root = None
        if log_inner_tasks:
            inner_root_base = inner_tb_log_root or "./inner-logs"
            inner_run_root = os.path.join(inner_root_base, run_stem)
            os.makedirs(inner_run_root, exist_ok=True)
            if self.verbose >= 1:
                print(f"[BaseMetaRL] Inner-loop TensorBoard root: {inner_run_root}")

        if self.tensorboard_logs is not None:
            if self.meta_logger is not None:
                self.meta_logger.close()
                self.meta_logger = None
            self._init_tensorboard_writer(
                tb_log_name=run_stem,
                reset_num_timesteps=True,
            )

        return {
            "task_seed_mode": task_seed_mode,
            "log_inner_tasks": log_inner_tasks,
            "inner_run_root": inner_run_root,
            "checkpoint_run_dir": checkpoint_run_dir,
        }

    def learn(
        self,
        outer_steps: Optional[int] = None,
        reset_task_history_before_learning: bool = True,
        task_seed_mode: Literal["generator", "meta_step"] = "generator",
        tb_log_name: Optional[str] = None,
        log_inner_tasks: bool = False,
        inner_tb_log_root: Optional[str] = None,
        save_freq: int = 0,
        save_path: Optional[str] = None,
        save_prefix: str = "meta_ckpt",
        save_final: bool = True,
    ) -> "BaseMetaRL":
        """
        Meta-training loop (outer loop).

        This mirrors SB3's `.learn()` name, but the "timesteps" dimension
        is split as:

            total_env_steps ≈ outer_steps * inner_steps

        Args:
            outer_steps: override number of meta-iterations; if None, use self.outer_steps.
            reset_task_history_before_learning: reset TaskGenerator history at the start.
            task_seed_mode:
              - "generator": TaskGenerator draws seeds from its own RNG (default).
              - "meta_step": each sampled task uses seed=meta_step_index (for strict 
                            reproducibility purposes).
            tb_log_name: Optional TensorBoard run name. If None, defaults to class name.
            log_inner_tasks: If True, logs each inner-loop task with a dedicated
                SB3 tb_log_name (when supported by the inner learner).
            inner_tb_log_root: Optional tensorboard root directory for inner-loop
                task logs (SB3 models only).
            save_freq: Save checkpoint every `save_freq` outer steps. 0 disables periodic saves.
            save_path: Directory where checkpoints are saved. Defaults to "./checkpoints".
            save_prefix: Prefix used for checkpoint filenames.
            save_final: If True, always save final checkpoint at the end of training.

        Returns:
            self (so you can write `meta_learner.learn(...).get_meta_policy()`).
        """
        setup = self._setup_learn(
            outer_steps=outer_steps,
            reset_task_history_before_learning=reset_task_history_before_learning,
            task_seed_mode=task_seed_mode,
            tb_log_name=tb_log_name,
            log_inner_tasks=log_inner_tasks,
            inner_tb_log_root=inner_tb_log_root,
            save_freq=save_freq,
            save_path=save_path,
            save_final=save_final,
        )
        task_seed_mode = setup["task_seed_mode"]
        log_inner_tasks = setup["log_inner_tasks"]
        inner_run_root = setup["inner_run_root"]
        checkpoint_run_dir = setup["checkpoint_run_dir"]

        last_saved_outer = 0
        try:
            for outer in range(self.outer_steps):
                task_models = []
                task_batch = []
                for i in range(self.task_batch_size):
                    task_meta_step = outer * self.task_batch_size + i
                    if task_seed_mode == "meta_step":
                        task_batch.append(
                            self.task_generator.get_task(task_meta_step, seed=task_meta_step)
                        )
                    else:
                        task_batch.append(self.task_generator.get_task(task_meta_step))

                inner_start = time.perf_counter()
                for task_idx, (env, task_info, first_occurrence) in enumerate(task_batch):
                    task_model = self.instantiate_model(env)
                    self.copy_meta_to_task(task_model)
                    task_meta_step = outer * self.task_batch_size + task_idx
                    inner_task_run_name = None
                    if log_inner_tasks:
                        inner_task_run_name = (
                            f"inner_outer_{outer + 1}_task_{task_idx + 1}_meta_{task_meta_step}"
                        )
                    self._inner_adapt_with_optional_logging(
                        task_model,
                        inner_tb_log_name=inner_task_run_name,
                        inner_tb_log_root=inner_run_root,
                    )
                    task_models.append(task_model)
                inner_total_seconds = time.perf_counter() - inner_start

                meta_start = time.perf_counter()
                self.meta_update(task_models, outer)
                meta_update_seconds = time.perf_counter() - meta_start

                self._log_outer_step_metrics(
                    outer_step=outer,
                    inner_total_seconds=inner_total_seconds,
                    meta_update_seconds=meta_update_seconds,
                )

                global_outer = outer + 1
                if save_freq > 0 and (global_outer % save_freq == 0):
                    ckpt_base = os.path.join(
                        checkpoint_run_dir,
                        f"{save_prefix}_outer_{global_outer}",
                    )
                    self._save_checkpoint_with_metadata(
                        checkpoint_path=ckpt_base,
                        outer_step_completed=global_outer,
                        task_seed_mode=task_seed_mode,
                        save_metadata=False
                    )
                    last_saved_outer = global_outer

            if save_final and last_saved_outer != self.outer_steps:
                final_base = os.path.join(
                    checkpoint_run_dir,
                    f"{save_prefix}_outer_{self.outer_steps}",
                )
                self._save_checkpoint_with_metadata(
                    checkpoint_path=final_base,
                    outer_step_completed=self.outer_steps,
                    task_seed_mode=task_seed_mode,
                )
        finally:
            if self.meta_logger is not None:
                self.meta_logger.close()
                self.meta_logger = None

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

    def get_meta_lr(self, outer_step: int) -> float:
        total = max(self.outer_steps - 1, 1)  # so final step reaches end LR
        lr = float(self.meta_lr_schedule(outer_step, total))
        self.current_meta_lr = lr
        return lr

    def sync_meta_optimizer_lr(
        self,
        meta_optimizer: Optional[Optimizer],
        meta_lr: float,
        force: bool = False,
    ) -> None:
        """
        Update optimizer lr only when needed.

        - For constant schedules, updates are skipped after optimizer init.
        - For dynamic schedules, param-groups are updated only when lr changed.
        """
        if meta_optimizer is None:
            return

        if not force and self.is_constant_meta_lr:
            return

        if force or self._last_optimizer_meta_lr != meta_lr:
            for group in meta_optimizer.param_groups:
                group["lr"] = meta_lr
            self._last_optimizer_meta_lr = meta_lr

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
