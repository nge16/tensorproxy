from typing import List, Tuple, Callable, Any, Dict
from abc import ABC
from dataclasses import dataclass

import numpy as np

from tensorproxy.simulate import SimulationModel
from tensorproxy.surrogate import SurrogateModel


@dataclass
class Constraint:
    cn_type: str
    """Тип ограничения: "ineq" для неравенств типа c(x) <= 0,
    "eq" для ограничений в виде равенство  c(x) = 0"""

    fun: Callable[[List, Any], float]
    """Функция для расчета значения ограничения"""

    jac: Callable[[List, Any], float] | None = None
    """Функция для расчета градиента функции-ограничения"""

    args: List = ()
    """Экстра-параметры для передачу в функцию-ограничение и ее градиента"""


@dataclass
class SurrogateConstraint:
    cn_type: str
    """Тип ограничения: "ineq" для неравенств типа c(x) <= 0,
    "eq" для ограничений в виде равенство  c(x) = 0"""

    surrogate_model: SurrogateModel
    """Суррогатная модель для вычисления значений функции-ограничения."""

    inverse: bool = False
    """Инвертировать значения ограничений (для неравенств типа c(x) >= 0)"""


class SurrogateOptimizationTask(ABC):
    """Базовый класс для определения задачи суррогатной оптимизации."""

    def __init__(
        self,
        simulation_model: SimulationModel,
        objective_model: SurrogateModel,
        constraints_models: List[SurrogateConstraint],
    ) -> None:
        super().__init__()
        self.simulation_model = simulation_model
        self.objective_model = objective_model
        self.constraints_models = constraints_models

    @property
    def constraints_description(self) -> List[Constraint]:
        """Возвращает описание ограничений."""
        return [
            Constraint(
                cn_type=constraint.cn_type,
                # для неравенств типа c(x) >= 0 применяется умножение на -1
                fun=lambda x: (-1.0 if constraint.inverse else 1.0)
                * constraint.surrogate_model.predict(x),
                jac=lambda x: (-1.0 if constraint.inverse else 1.0)
                * constraint.surrogate_model.predict_gradient(x),
            )
            for constraint in self.constraints_models
        ]

    @property
    def name(self) -> str:
        """Возвращает наименование задачи.

        Предполагается, что свойство может быть переопределено в наследнике.
        """
        return self.__class__.__name__

    def objective(self, x: List[float], args=()) -> float:
        """Вычисляет значение целевой функции.

        Args:
            x (List[float]): значения управляемых переменных, 1-D массив
            размерности (n,)
            args: кортеж фиксированных параметров для полной спецификации
            функции

        Returns:
            float: значение целевой функции в точке x
        """
        return self.objective_model.predict(x)

    def gradient(self, x: List[float], args=()) -> np.ndarray:
        """Возвращает градиент целевой функции.

        Args:
            x (List[float]): значения управляемых переменных, 1-D массив
            размерности (n,)
            args: кортеж фиксированных параметров для полной спецификации
            функции

        Returns:
            np.ndarray: градиент, 1-D массив размерности (n,)

        """
        return self.objective_model.predict_gradient(x)

    def constraints(self, x: List[float], args=()) -> np.ndarray:
        """Возвращает значения ограничений.

        Args:
            x (List[float]): значения управляемых переменных

        Returns:
            np.ndarray: значения ограничений в точке `x`
        """
        return [
            constraint.fun() for constraint in self.constraints_description
        ]

    def jacobian(self, x: List[float], args=()):
        """Колбэк для вычисления якобиана.

        Args:
            x (List[float]): значения управляемых переменных

        Returns:
            np.ndarray: якобиан
        """
        print("calculating jacobian...")
        return [
            constraint.jac(x, args)
            for constraint in self.constraints_description
        ]

    def has_hessians(self):
        return self.hessianstructure() is not None

    def hessianstructure(
        self,
    ):
        """Возвращает номера строк и столбцов для ненулевых элементов
        матрицы Гессе."""
        return None

    def hessian(
        self, x: List[float], lagrange: List[float], obj_factor: float
    ):
        """Возвращает ненулевые значения Гессиана Лагранжиана целевой
        функции

        Args:
            x (List[float]): значения управляемых переменных

        Returns:
            Гессиан Лагранжиана целевой функции
        """
        return None

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Возвращает кортеж векторов с нижними границами для значений
        управляемых переменных и верхними границами.

        Returns:
            Tuple[np.ndarray, np.ndarray]: вектора с нижними и верхними
              границами управляемых переменных
        """
        lower = np.array(
            [param.lower for param in self.simulation_model.domain]
        )
        upper = np.array(
            [param.upper for param in self.simulation_model.domain]
        )
        return lower, upper

    def cmp_s2s(self, x: List[float], args=()):
        """Сравнивает значения суррогатной и полноразмерной моделей и
        ограничений в точке x.

        Args:
            x (List[float]): значения управляемых переменных
            args (tuple, optional): фиксированные аргументы, необходимые для
                расчета значений целевой функции и ограничений


        Returns:
            Tuple[List[str] | None, bool, Dict[str, List[float]], float]:
                - список ошибок, возникших при вызове симулятора
                - признак, выполнен ли критерий сходимости суррогатной и
                  полноразмерной моделей
                - словарь, содержащий имя суррогатной функции и истинные
                  значения в точке `x`
                - метрика отклонения значений суррогатных моделей от истинного
                  значения

        """
        self.simulation_model.simulation_service.reset_calculations()
        self.simulation_model.assign_x(x)
        errors = self.simulation_model.simulation_service.calculate()

        surrogates = {
            self.objective_model.model_name: self.objective_model,
        } | {
            constraint.surrogate_model.model_name: constraint.surrogate_model
            for i, constraint in enumerate(self.constraints_models)
        }
        # расчет суррогатных значений целевой функции и ограничений в точке x
        ysurr = {
            name: surrogates[name].predict(x, args=())
            for name in surrogates.keys()
        }

        # расчет истинных значений целевой функции и ограничений в точке x
        y = {
            name: [f() for f in surrogates[name].results]
            for name in surrogates.keys()
        }

        # сравнение суррогатных и истинных значений
        ysurr_f = np.array(
            [yi for vl in ysurr.values() for yi in np.atleast_1d(vl)]
        )
        y_f = np.array([yi for vl in y.values() for yi in np.atleast_1d(vl)])

        print("\n")
        with np.printoptions(precision=3, suppress=False):
            print(f"====Суррогатные значения: {ysurr_f}")
            print(f"====Истинные значения:    {y_f}")
        print("\n")

        error_rate = np.linalg.norm(ysurr_f - y_f)
        converged = error_rate < 1e-2

        return None if not errors else (errors, converged, y, error_rate)
