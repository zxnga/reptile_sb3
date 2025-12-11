import math
import random
from typing import Any, Dict, Tuple

import torch as th
from torch import Tensor

from ..task_generator import TaskGenerator


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

    def __init__(self, amp_min: float = 0.1, amp_max: float = 5.0):
        self.amp_min = amp_min
        self.amp_max = amp_max
        self.reset_history()

    def reset_history(self):
        # For this experiment we don’t need to reuse tasks,
        # so we just clear any tracking.
        self._seen_tasks: Dict[int, int] = {}

    def _sample_task(self) -> Tuple[SineTask, Dict[str, float]]:
        a = random.uniform(self.amp_min, self.amp_max)
        b = random.uniform(0.0, 2.0 * math.pi)
        task = SineTask(a, b)
        info = {"amplitude": a, "phase": b}
        return task, info

    def get_task(self, idx: int):
        """
        Returns:
            task: SineTask
            task_info: dict
            first_occurrence: int
        """
        # For simplicity every call returns a fresh task; we treat
        # each idx as a new task.
        task, info = self._sample_task()
        first_occurrence = idx  # we never reuse, so first time = idx
        return task, info, first_occurrence
