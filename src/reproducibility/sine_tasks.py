import math
import random
import numpy as np
from typing import Any, Dict, Tuple, Optional

import torch as th
from torch import Tensor

from src.task_generator import TaskGenerator


class SineTask:
    """
    One sine-wave regression task:
        y = a * sin(x + b)
    with its own amplitude and phase.
    """

    def __init__(self, amplitude: float, phase: float):
        self.amplitude = amplitude
        self.phase = phase

    def sample(self, n_points: int) -> Tuple[Tensor, Tensor]:
        """
        Sample (x, y) pairs:
            x ~ U[-5, 5]
            y = a * sin(x + b)
        """
        x = th.empty(n_points, 1).uniform_(-5.0, 5.0)
        y = self.amplitude * th.sin(x + self.phase)
        return x, y


class SineTaskGenerator(TaskGenerator):
    """
    Minimal TaskGenerator implementation for your ReptileAgent.

    ReptileAgent expects:
        - reset_history()
        - get_task(i) -> (task, task_info, first_occurrence_index)
    """

    def __init__(
        self,
        amplitude_range: Tuple[float, float] = (0.1, 5.0),
        phase_range: Tuple[float, float] = (0.0, 2 * np.pi),
        generator_seed: Optional[int] = None,
    ):
        self.amp_min, self.amp_max = amplitude_range
        self.phase_min, self.phase_max = phase_range

        if generator_seed is not None:
            self.rng = np.random.default_rng(generator_seed)
        else:
            self.rng = np.random.default_rng()

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
        For now: always create a brand new task (no revisits),
        to keep the repro simple.

        Returns:
            task: SineTask instance
            info: dict with amplitude, phase
            origin_meta_step: meta_step when first created (== meta_step here)
        """
        if seed is None:
            seed = int(self.rng.integers(0, 2**32 - 1))

        # make sampling deterministic per task if you want
        np.random.seed(seed)

        amplitude = self.rng.uniform(self.amp_min, self.amp_max)
        phase = self.rng.uniform(self.phase_min, self.phase_max)

        task = SineTask(amplitude=amplitude, phase=phase)
        info = {"amplitude": amplitude, "phase": phase, "seed": seed}

        self.selected_tasks.append(
            {"seed": seed, "first_meta_step": meta_step, "task_info": info}
        )

        return task, info, meta_step