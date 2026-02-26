import numpy as np
from typing import Any, Dict, Tuple, Optional, List

import torch as th
from torch import Tensor

from src.task_generator import TaskGenerator


class SineTask:
    """
    One sine-wave regression task:
        y = a * sin(x + b)
    with its own amplitude and phase.
    """

    def __init__(
        self,
        amplitude: float,
        phase: float,
        point_seed: Optional[int] = None,
        device: str = "cpu",
    ):
        self.amplitude = float(amplitude)
        self.phase = float(phase)
        self.device = device
        self.point_seed = point_seed

        self._torch_gen: Optional[th.Generator] = None
        if point_seed is not None:
            self._torch_gen = th.Generator(device="cpu")
            self._torch_gen.manual_seed(point_seed)

    def sample(self, n_points: int) -> Tuple[Tensor, Tensor]:
        """
        Sample (x, y) pairs:
            x ~ U[-5, 5]
            y = a * sin(x + b)
        """
        if self._torch_gen is None:
            x = th.empty(n_points, 1, device=self.device).uniform_(-5.0, 5.0)
        else:
            x = -5.0 + 10.0 * th.rand(
                n_points, 1, generator=self._torch_gen, device=self.device
            )

        y = self.amplitude * th.sin(x + self.phase)
        return x, y

    def dense_grid(self, n_points: int = 50) -> Tuple[Tensor, Tensor]:
        x = th.linspace(-5.0, 5.0, n_points, device=self.device).unsqueeze(1)
        y = self.amplitude * th.sin(x + self.phase)
        return x, y


class SineTaskGenerator(TaskGenerator):
    """
    Minimal TaskGenerator implementation for Reptile.

    Expected interface:
        - reset_history()
        - get_task(i) -> (task, task_info, first_occurrence_index)
    """

    def __init__(
        self,
        amplitude_range: Tuple[float, float] = (0.1, 5.0),
        phase_range: Tuple[float, float] = (0.0, 2 * np.pi),
        generator_seed: Optional[int] = None,
        device: str = "cpu",
    ):
        self.amp_min, self.amp_max = amplitude_range
        self.phase_min, self.phase_max = phase_range
        self.device = device

        self.rng = (
            np.random.default_rng(generator_seed)
            if generator_seed is not None
            else np.random.default_rng()
        )

        self.selected_tasks: List[Dict[str, Any]] = []
        self.revisit_counter = 0

    def reset_history(self) -> None:
        self.selected_tasks = []
        self.revisit_counter = 0

    def get_task(
        self,
        meta_step: int,
        seed: Optional[int] = None,
    ) -> Tuple[SineTask, Dict[str, Any], Optional[int]]:
        """
        Always create a new task (no revisits).
        """
        if seed is None:
            seed = int(self.rng.integers(0, 2**32 - 1))

        amplitude = float(self.rng.uniform(self.amp_min, self.amp_max))
        phase = float(self.rng.uniform(self.phase_min, self.phase_max))

        # point_seed=seed for task-local reproducibility
        task = SineTask(
            amplitude=amplitude,
            phase=phase,
            point_seed=seed,
            device=self.device,
        )

        info = {"amplitude": amplitude, "phase": phase, "seed": seed}

        self.selected_tasks.append(
            {"seed": seed, "first_meta_step": meta_step, "task_info": info}
        )

        return task, info, meta_step