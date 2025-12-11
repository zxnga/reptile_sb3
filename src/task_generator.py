from typing import Any, Dict, Optional, List, Callable, Tuple
import random

import gymnasium as gym
import numpy as np

ListTask = List[Tuple[gym.Env, Dict[str, Any]]]

# TODO: decaying revisits

class TaskGenerator:
    """
    Meta-RL task generator.

    Modes:
      1) Static tasks: pass a list of (env, info) in `tasks`.
      2) Dynamic tasks: pass a callable
           task_callable(random_seed: int, **task_callable_params) -> (env, info)
         and this class will sample/stash seeds for revisits.

    For dynamic tasks, tasks are fully defined by:
      - random_seed
      - task_callable_params

    Revisits are handled by storing seed + a bit of metadata,
    then re-calling `task_callable` with the same seed.
    """

    def __init__(
        self,
        tasks: Optional[ListTask] = None,
        task_callable: Optional[Callable[..., Tuple[gym.Env, Dict[str, Any]]]] = None,
        task_callable_params: Optional[Dict[str, Any]] = None,
        revisit_ratio: float = 0.15,
        revisit_start: int = 1,
        sampling_method: str = "random",      # "cyclic" | "random" | "weighted"
        sampling_weights: Optional[List[float]] = None,
        generator_seed: Optional[int] = None,
        revisit_weight_fn: Optional[Callable[[List[Dict[str, Any]]], List[float]]] = None,
    ):
        assert tasks is not None or task_callable is not None, (
            "Either 'tasks' (list of tasks) or 'task_callable' "
            "(callable to generate tasks) must be provided."
        )

        self.tasks = tasks
        self.task_callable = task_callable
        self.task_callable_params = dict(task_callable_params or {})

        self.revisit_ratio = revisit_ratio
        self.revisit_start = revisit_start
        self.sampling_method = sampling_method
        self.sampling_weights = sampling_weights

        # for meta-level decisions (new vs revisit, which past task, etc.)
        if generator_seed is not None:
            self._rng = np.random.default_rng(generator_seed)
            self._py_rng = random.Random(generator_seed)
        else:
            self._rng = np.random.default_rng()   # non-deterministic
            self._py_rng = random

        self.revisit_weight_fn = revisit_weight_fn # pass selected tasks and return a weight for each one
        self.revisit_counter = 0

        # list of past sampled tasks for revisiting in the form  {
        #   "seed": int,
        #   "first_meta_step": int,
        #   "meta_steps": [int, ...],
        #   "task_info": dict 
        # }
        self.selected_tasks: List[Dict[str, Any]] = []

        if self.tasks is not None and self.sampling_method == "weighted":
            if self.sampling_weights is None or len(self.sampling_weights) != len(self.tasks):
                raise ValueError("`sampling_weights` must be provided and match the number of tasks.")

    def reset_history(self) -> None:
        """Reset TaskGenerator."""
        self.selected_tasks = []
        self.revisit_counter = 0

    def get_task(
        self,
        meta_step: int,
        seed: Optional[int] = None,
    ) -> Tuple[gym.Env, Dict[str, Any], Optional[int]]:
        """
        Generate or retrieve a task.

        Returns:
            env:   a freshly created Gymnasium environment
            info:  task metadata
            origin_meta_step:
              - for dynamic tasks: meta-step when this task spec first appeared
              - for static task lists: None
        """

        # predefined list of tasks
        if self.tasks is not None:
            if len(self.tasks) == 0:
                raise ValueError("`tasks` must be non-empty.")

            if self.sampling_method == "cyclic":
                env, info = self.tasks[meta_step % len(self.tasks)]
            elif self.sampling_method == "random":
                env, info = self._py_rng.choice(self.tasks)
            elif self.sampling_method == "weighted":
                env, info = self._py_rng.choices(
                    self.tasks,
                    weights=self.sampling_weights,
                    k=1,
                )[0]
            else:
                raise ValueError(f"Unknown sampling_method: {self.sampling_method!r}")

            return env, info, None

        # task generated using callable
        can_revisit = (meta_step >= self.revisit_start) and bool(self.selected_tasks)
        if can_revisit and (self._rng.random() < self.revisit_ratio):
            task_idx = self._select_task_index_for_revisit()
            record = self.selected_tasks[task_idx]

            env, info = self.task_callable(
                random_seed=record["seed"],
                **self.task_callable_params,
            )
            origin_meta_step = record["first_meta_step"]
            record["meta_steps"].append(meta_step)
            return env, info, origin_meta_step

        if seed is None:
            seed = int(self._rng.integers(0, 2**63 - 1))

        env, info = self.task_callable(random_seed=seed, **self.task_callable_params)

        self.selected_tasks.append(
            {
                "seed": seed,
                "first_meta_step": meta_step,
                "meta_steps": [meta_step],
                "task_info": info,
            }
        )
        return env, info, meta_step

    def _select_task_index_for_revisit(self) -> int:
        """Select past task by index in selected_tasks to revisit."""
        if not self.selected_tasks:
            raise RuntimeError("No tasks available for revisit.")

        if self.sampling_method == "cyclic":
            idx = self.revisit_counter % len(self.selected_tasks)
            self.revisit_counter += 1
            return idx

        elif self.sampling_method == "random":
            self.revisit_counter += 1
            return self._py_rng.randint(0, len(self.selected_tasks) - 1)

        elif self.sampling_method == "weighted":
            # if special weights per task, maybe for incremental learning difficulty
            self.revisit_counter += 1
            if self.revisit_weight_fn is None:
                raise ValueError(
                    "sampling_method='weighted' for revisits but no "
                    "revisit_weight_fn was provided."
                )

            weights = self.revisit_weight_fn(self.selected_tasks)
            if len(weights) != len(self.selected_tasks):
                raise ValueError(
                    f"revisit_weight_fn must return one weight per task: "
                    f"{len(weights)} != {len(self.selected_tasks)}"
                )

            idx = self._py_rng.choices(
                population=list(range(len(self.selected_tasks))),
                weights=weights,
                k=1,
            )[0]
            return idx

        else:
            raise ValueError(f"Unknown sampling_method for revisit: {self.sampling_method!r}")
