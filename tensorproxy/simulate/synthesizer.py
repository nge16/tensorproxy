from typing import Generator, Dict, List, Tuple
import numpy as np

from smt.utils.design_space import DesignSpace, FloatVariable

from .simulation_service import SimulationService
from .simulation_model import SimulationModel
from .simulation_result import CallableSimulationResult


MAX_ITERS_TO_CONVERGE = 10


class SimulationSynthesizer:
    """Синтезатор расчетных данных для обучения суррогатных моделей.

    Args:
        budget (int): бюджет вычислений (число симуляций)
    """

    def __init__(
        self, budget: int, simulation_service: SimulationService
    ) -> None:
        self.n = budget
        self.simulation_service = simulation_service

    def synth(
        self,
        simulation_model: SimulationModel,
        results: Dict[str, List[CallableSimulationResult]],
        only_valid: bool = True,
        force_convergence: bool = True,
    ) -> Generator[Tuple[np.ndarray, Dict[str, np.ndarray]], None, None]:
        """Синтезирует данные путем симуляций.

        Args:
            simulation_model (SimulationModel): расчетная модель
            results (Dict[str, List[CallableSimulationResult]]): списки функций
              для заполнения результатов, сгруппированные по категориям
            only_valid (bool): игнорировать расчеты с ошибками
            force_convergence (bool): итерировать до сходимости результатов

        Yields:
            Generator[np.ndarray, Dict[str, np.ndarray]]: входные данные и
            результаты расчета (X, Y)
        """
        design_space = DesignSpace(
            [FloatVariable(p.lower, p.upper) for p in simulation_model.domain]
        )

        x_sampled, _ = design_space.sample_valid_x(self.n)
        for x in x_sampled:
            simulation_model.assign_x(x)

            yo = {
                key: [np.inf for _ in simulation_results]
                for (key, simulation_results) in results.items()
            }

            i = 1
            converged = False

            self.simulation_service.reset_calculations()
            while not converged:
                errors = self.simulation_service.calculate()
                # time.sleep(0.1)

                if errors and only_valid:
                    print("Для заданных значений не найдено валидное решение")
                    break

                y = {
                    key: [f() for f in simulation_results]
                    for (key, simulation_results) in results.items()
                }

                yo_flatten = np.array([yi for vl in yo.values() for yi in vl])
                y_flatten = np.array([yi for vl in y.values() for yi in vl])

                converged = (
                    np.linalg.norm(yo_flatten - y_flatten) < 1e-6
                ) or not force_convergence

                yo = y
                i += 1

                if i > MAX_ITERS_TO_CONVERGE:
                    print("Нет сходимости решения")
                    break

            else:
                yield x, y
