# Reptile_sb3

Extension/wrapper around Stable-Baselines3 (SB3) to run meta-RL with first-order meta-learning updates.

## What this project is

This repository provides a meta-training loop on top of SB3-style inner learners.

It is designed for:
- Meta-RL experiments where each outer iteration samples tasks and adapts an inner learner per task.
- First-order approximation methods (no second-order Hessian terms), with a current focus on Reptile-style updates.
- Fast prototyping by reusing SB3 training code in the inner loop instead of rewriting RL algorithms.

## Core idea

SB3 handles the inner training (`.learn()` on a task environment).  
This project adds the outer meta-loop:

1. Sample a batch of tasks.
2. Clone/copy meta-parameters to task learners.
3. Run inner adaptation on each task.
4. Apply a meta-update to the shared initialization.

This gives you a reusable meta-optimization layer while keeping SB3-compatible policies/algorithms underneath.

## First-order MAML perspective

This codebase targets first-order meta-learning behavior:
- Reptile is implemented and is first-order.
- The architecture is intentionally wrapper-based so other first-order MAML-like updates can be added as new subclasses of `BaseMetaAlgorithm`.

## Main modules

- `src/base_meta_class.py`
  - `BaseMetaAlgorithm`: generic outer-loop meta-training wrapper.
  - Handles task generator, model instantiation, inner adaptation call, and shared API (`learn`, `predict`, `policy`, `save_meta_algo`).

- `src/reptile.py`
  - `ReptileMetaRL`: Reptile first-order meta-update implementation.
  - Supports direct meta-step or optimizer-based meta-step.
  - Supports ignored parameter prefixes in meta-update.

- `src/task_generator.py`
  - `TaskGenerator`: static or callable-based task sampling.
  - Supports revisit policies (`random`, `cyclic`, `weighted`).

- `src/utils.py`
  - Weight copy helper.
  - `compute_updates(...)` utility for SB3 on-policy/off-policy update counting.

- `src/reproducibility/*`
  - Sine-wave supervised benchmark used to sanity-check the meta-learning flow based on the original paper:
    - `sine_tasks.py`
    - `supervised_algo.py`
    - `run_sine_reptile.py`

## How to extend to your own SB3 setup

- Instanciate a task generator that returns task environments. Tasks are sampled/created inside the task generator following two methods:
  - `tasks: Optional[ListTask]`: Contruct all your training envs beforehand and feed them as a list to the `TaskGenerator` through the `tasks` parameter.
  - `task_callable: Optional[Callable[..., Tuple[gym.Env, Dict[str, Any]]]]`: Construct a function that generates training envs and pass the callable to the `tasks_generator_cls` param and optional associated parameters to the `tasks_generator_params` param.
- Pass your SB3 algorithm class as `rl_algorithm` and hyperparameters in `rl_algo_kwargs` to the `ReptileMetaRL` object.
- Learn the meta-initialization using the `.learn()` method.

### How to create your own task generator

The `TaskGenerator` class is provided to facilitate the creation, sampling and revisit of task especially for complex environments that need specific initialization logic. One can create its own task generator as long as it implements the 2 necessary methods called during the meta-learning process:
- get_tasks: that generates or retreive a env 

```python
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
```
- reset_history: that resets the list of already sampled tasks:

```python
def reset_history(self) -> None:
        """Reset TaskGenerator."""
```


 We provide an example of how to create a minimal task generator by extending an existing gymnasium environment.


```python
from typing import Any, Dict, Optional, Tuple
import gymnasium as gym


class SimpleCartPoleTaskGenerator:
    """
    Minimal TaskGenerator for testing ReptileMetaRL.

    Always returns a fresh CartPole-v1 env, ignoring seed/revisits.
    Compatible with BaseMetaRL's expected API.
    """

    def __init__(self, env_id: str = "CartPole-v1"):
        self.env_id = env_id

    def reset_history(self) -> None:
        # nothing to reset in this simple version
        pass

    def get_task(
        self,
        meta_step: int,
        seed: Optional[int] = None,
    ) -> Tuple[gym.Env, Dict[str, Any], Optional[int]]:

        env = gym.make(self.env_id)
        info: Dict[str, Any] = {"meta_step": meta_step}
        origin_meta_step: Optional[int] = meta_step

        return env, info, origin_meta_step

```


## How to implement your own meta-update

1. Use `BaseMetaAlgorithm` as the parent class.
2. Implement `meta_update(task_models)` in a child class.


## Results validation

Fast run on supervised task:

```powershell
python -m src.reproducibility.run_sine_reptile
```

![](src\reproducibility\figures\figure1_like_reptile.png)