from typing import List, Callable

import scipy.optimize
from scipy.optimize import OptimizeResult

from ..optimization_solver import OptimizationSolver
from ..optimization_task import SurrogateOptimizationTask

from tensorproxy.utils import check_unknown_options


class ScipySolver(OptimizationSolver):
    """Солвер на основе scipy.

    Args:
      task (SurrogateOptimizationTask): решаемая задача оптимизации
      method (str | callable): метод оптимизации
    """

    def __init__(
        self,
        task: SurrogateOptimizationTask,
        args=(),
        method: str | Callable | None = None,
        tol: float | None = None,
        options: dict | None = None,
        callback: Callable[[], OptimizeResult] = None,
        **unknown_options,
    ) -> None:
        super().__init__(task)

        self.method = method
        self.args = args
        self.tol = tol
        self.options = options
        self.callback = callback

        check_unknown_options(unknown_options)

    def minimize(self, x0: List[float]) -> OptimizeResult:

        bounds = None
        if "bounds" in self.task.__class__.__dict__:
            low, upper = self.task.bounds()
            bounds = ((low[i], upper[i]) for i in range(len(low)))

        # TODO: Посмотреть, можно ли сразу вычислять и функцию, и градиенты
        #  If `jac` is a Boolean and is True,
        #  `fun` is assumed to return a tuple ``(f, g)`` containing the
        #  objective function and the gradient.

        return scipy.optimize.minimize(
            fun=self.task.objective,
            x0=x0,
            args=self.args,
            method=self.method,
            jac=(
                None
                if "gradient" not in self.task.__class__.__dict__
                else self.task.gradient
            ),
            hess=(
                None
                if "hessian" not in self.task.__class__.__dict__
                else self.task.hessian
            ),
            bounds=bounds,
            constraints=[
                {
                    "type": constraint.cn_type,
                    "fun": constraint.fun,
                    "jac": constraint.jac,
                    "args": constraint.args,
                }
                for constraint in self.task.constraints_description
            ],
            tol=self.tol,
            options=self.options,
            callback=self.callback,
        )
