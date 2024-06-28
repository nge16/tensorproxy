from typing import List
from abc import ABC, abstractmethod

from scipy.optimize import OptimizeResult

from .optimization_task import SurrogateOptimizationTask


class OptimizationSolver(ABC):
    """Базовый класс для оптимизаторов."""

    def __init__(self, task: SurrogateOptimizationTask) -> None:
        super().__init__()
        self._task = task

    @property
    def task(self):
        return self._task

    @abstractmethod
    def minimize(self, x0: List[float]) -> OptimizeResult:
        """Выполняет решение оптимизационной задачи.

        Args:
            x0 (List[float]): начальное приближение
        """
        pass
