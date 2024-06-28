"""Солвер на основе PyOptSparse:
https://github.com/mdolab/pyoptsparse/tree/main/examples
"""

from tensorproxy.optimize.optimization_solver import OptimizationSolver
from tensorproxy.optimize.optimization_task import SurrogateOptimizationTask


class PyOptSolver(OptimizationSolver):
    def __init__(self, task: SurrogateOptimizationTask) -> None:
        super().__init__(task)
