import numpy as np

from tensorproxy.simulate.simulator import Simulator
from tensorproxy.deprecated.simulation_task import SimulationTask

from tensorproxy.utils.log import deprecated

MAX_ITERS_TO_CONVERGE = 10


class DataSynthesizer:
    """Синтезатор данных суррогатной модели.

    Args:
        budget (int): бюджет вычислений (число симуляций)
    """

    def __init__(self, budget: int, simulator: Simulator) -> None:
        self.n = budget
        self.sim = simulator

    @deprecated(
        "Для синтеза данных используйте методы пакета "
        "pyhydrosym.simulate.synthesize"
    )
    def synth(
        self,
        task: SimulationTask,
        only_valid: bool = True,
        force_convergence: bool = True,
    ):
        """Синтезирует данные путем симуляций.

        Args:
            task (SimulationTask): задача вычислительного эксперимента
            only_valid (bool): игнорировать расчеты с ошибками
            force_convergence (bool): итерировать до сходимости результатов

        Yields:
            (np.ndarray, np.ndarray): входные данные и результаты
            расчета (X, Y)
        """
        x_sampled, _ = task.design_space.sample_valid_x(self.n)
        for x in x_sampled:
            task.assign_x(x)

            yo = np.array([1e3 for _ in task.fget_y])
            i = 1
            converged = False

            task.flowsheet.reset_calculations()
            while not converged:
                errors = self.sim.calculate_flowsheet()
                # time.sleep(0.1)

                if errors and only_valid:
                    break

                y = np.array([f() for f in task.fget_y])

                converged = (
                    np.linalg.norm(yo - y) < 1e-6
                ) or not force_convergence

                yo = y
                i += 1

                if i > MAX_ITERS_TO_CONVERGE:
                    break

            else:
                yield x, *y
