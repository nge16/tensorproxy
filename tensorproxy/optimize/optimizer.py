from typing import List

from scipy.optimize import OptimizeResult

from tensorproxy.surrogate import SurrogateTrainer

from .optimization_task import SurrogateOptimizationTask
from .optimization_solver import OptimizationSolver

from tensorproxy.optimize.solvers.scipy_solver import ScipySolver
from tensorproxy.optimize.solvers.pygmo_solver import PygmoSolver


class SurrogateOptimizer:
    """Суррогатный оптимизатор.

    В основной функции ``minimize`` выполняет минимизацию
    суррогатной целевой функции с помощью выбранного солвера
    и сравнение ее значения с аналогичным значением исходной
    полноразмерной модели. Итерирует до наступления сходимости
    или превышения максимального числа итераций.
    """

    SOLVERS = {"scipy": ScipySolver, "pygmo": PygmoSolver}

    def __init__(
        self,
        opt_task: SurrogateOptimizationTask,
        trainer: SurrogateTrainer,
        solver="scipy",
        *args,
        **kwargs,
    ) -> None:
        self.task = opt_task
        clazz = SurrogateOptimizer.SOLVERS.get(solver, None)
        if clazz is None:
            raise AttributeError(
                f"Указанный солвер ({solver}) недоступен. "
                "Доступные солверы: "
                f"{', '.join(list(SurrogateOptimizer.SOLVERS.keys()))}"
            )

        self._callback = kwargs.pop("callback", None)
        self._solver = clazz(task=opt_task, *args, **kwargs)
        self.surrogate_trainer = trainer

    @property
    def solver(self) -> OptimizationSolver:
        return self._solver

    def minimize(
        self, x0: List[float], force_converge: bool = True, iter_max: int = 100
    ) -> OptimizeResult:
        """Выполняет суррогатную оптимизацию.

        Args:
            x0 (List[float]): начальное приближение
            force_converge (bool): если True, добиваться сходимости суррогатной
              и полноразмерной моделей
            iter_max (int): максимальное число итераций алгоритма
        """
        i = 0
        converged = False
        while not converged:
            res: OptimizeResult = self.solver.minimize(x0)

            try:
                if not res["success"]:
                    break
            except KeyError:
                raise RuntimeError(
                    f"Солвер {self.solver.__class__.__name__} вернул "
                    "результат выполнения функции minimize, "
                    "не соответствующий соглашению о формате OptimizeResult "
                    "(в словаре отсутствует ключ 'success')"
                )

            if not force_converge:
                # не требуется обеспечивать сходимость суррогатной
                # и полноразмерной моделей
                # value = self.task.objective(res.x)
                break

            # TODO: если ошибки в errors?
            errors, converged, y, error_rate = self.task.cmp_s2s(res.x)

            if not converged:
                if self.surrogate_trainer is None:
                    break

                self.surrogate_trainer.further_train(res.x, y)

            x0 = res.x
            i += 1

            res["iter"] = i
            res["errors"] = errors
            res["surrogate_converged"] = converged
            res["error_rate"] = error_rate

            if self._callback:
                self._callback(intermediate_result=res)

            if i > iter_max:
                res["success"] = False
                res["message"] = (
                    f"За максимальное число итераций ({iter_max}) не удалось "
                    "найти сходящееся решение"
                )
                break

        return res
