"""Солвер на основе pygmo:
https://github.com/esa/pygmo2
https://esa.github.io/pygmo2/index.html
"""

from typing import List, Tuple
from scipy.optimize import OptimizeResult

import numpy as np
import pygmo as pg

from tensorproxy.optimize.optimization_solver import OptimizationSolver
from tensorproxy.optimize.optimization_task import SurrogateOptimizationTask

# from pyhydrosim.utils import check_unknown_options

# TODO: numba для ускорения вычислений
# см. https://esa.github.io/pygmo2/tutorials/coding_udp_simple.html


class PygmoProblem:
    """Обертка над pygmo.problem."""

    def __init__(self, task: SurrogateOptimizationTask) -> None:
        self.task = task

    def fitness(self, x: List[float]) -> List[float]:
        """Возвращает конкатенацию значений целевой функции,
        ограничений-равенств и ограничений-неравенств.

        Args:
            x (List[float]): значения управляющих переменных

        Returns:
            List[float]: значения целевой функции,
        ограничений-равенств и ограничений-неравенств
        """
        eq_cns = filter(
            lambda constraint: constraint.cn_type == "eq",
            self.task.constraints_description,
        )
        ineq_cns = filter(
            lambda constraint: constraint.cn_type == "ineq",
            self.task.constraints_description,
        )

        return (
            [self.task.objective(x)]
            + [-cns.fun(x) for cns in eq_cns]
            + [-cns.fun(x) for cns in ineq_cns]
        )

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Возвращает кортеж векторов с нижними границами для значений
        управляемых переменных и верхними границами.

        Returns:
            Tuple[np.ndarray, np.ndarray]: вектора с нижними и верхними
              границами управляемых переменных
        """
        lb, up = self.task.bounds()
        if not lb or not up:
            raise RuntimeError(
                "Для работы платформы pygmo описание задачи должно содержать "
                "реализацию метода bounds()"
            )

        return (
            [low if low is not None else -1e19 for low in lb],
            [up if up is not None else 1e19 for up in up],
        )

    def get_nec(self) -> int:
        """Возвращает число ограничений-равенств."""
        return len(
            list(
                filter(
                    lambda constraint: constraint.cn_type == "eq",
                    self.task.constraints_description,
                )
            )
        )

    def get_nic(self):
        """Возвращает число ограничений-неравенств."""
        return len(
            list(
                filter(
                    lambda constraint: constraint.cn_type == "ineq",
                    self.task.constraints_description,
                )
            )
        )

    def get_nobj(self) -> int:
        """Возвращает число целевых функций.

        Returns:
            int: число целевых функций
        """
        return 1

    def has_gradient(self) -> bool:
        """Сообщает, доступен ли градиент

        Returns:
            bool: доступность градиента
        """
        return "gradient" in self.task.__class__.__dict__

    def gradient(self, x: List[float]) -> List[float]:
        """Возвращает градиент целевой функции.

        Args:
            x (List[float]): значения управляемых переменных, 1-D массив
            размерности (n,)

        Returns:
            градиент, 1-D массив размерности (n,)

        """
        return np.concatenate(
            (
                self.task.gradient(x),
                np.array(
                    [
                        constraint.jac(x)
                        for constraint in self.task.constraints_description
                    ]
                ).flatten(),
            )
        )

    def get_name(self) -> str:
        return self.task.name

    def get_extra_info(self):
        return self.__class__.__name__

    def get_nix(self):
        return 0

    def batch_fitness(self, x: List):
        pass

    def has_batch_fitness(self):
        return False

    def has_gradient_sparsity(self):
        return False

    def gradient_sparsity(self):
        pass

    def has_hessians(self):
        return self.task.has_hessians()

    def hessians(self, x: List):
        # TODO: ?
        return self.task.hessian(x)

    def has_hessians_sparsity(self):
        return False

    def hessians_sparsity(self):
        pass

    def has_set_seed(self):
        return False

    def set_seed(self, s):
        pass


class IPOPT(pg.ipopt):
    def __init__(self, *args, **kwargs):
        super().__init__()
        for option, value in kwargs.items():
            if isinstance(value, float):
                self.set_numeric_option(option, value)
            elif isinstance(value, int):
                self.set_integer_option(option, value)
            elif isinstance(value, str):
                self.set_string_option(option, value)


class NLOpt(pg.nlopt):
    def __init__(self, *args, **kwargs):
        super().__init__(solver=kwargs.pop("method"))


class PygmoSolver(OptimizationSolver):

    ALGORITHMS = {
        # глобальная оптимизация
        "gaco": pg.gaco,
        "de": pg.de,
        "sade": pg.sade,
        "de1220": pg.de1220,
        "gwo": pg.gwo,
        "ihs": pg.ihs,
        "pso": pg.pso,
        "pso_gen": pg.pso_gen,
        "sea": pg.sea,
        "sga": pg.sga,
        "simulated_annealing": pg.simulated_annealing,
        "bee_colony": pg.bee_colony,
        "cmaes": pg.cmaes,
        "xnes": pg.xnes,
        "nsga2": pg.nsga2,
        "moead": pg.moead,
        "moead_gen": pg.moead_gen,
        "maco": pg.maco,
        "nspso": pg.nspso,
        # локальная оптимизация
        "compass_search": pg.compass_search,
        "scipy": pg.scipy_optimize,
        "nlopt": NLOpt,
        "ipopt": IPOPT,
    }

    def __init__(
        self,
        task: SurrogateOptimizationTask,
        algorithm: str,
        popsize: int = 10,
        n_islands=1,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(task)

        udp = PygmoProblem(task)
        self.problem = pg.problem(udp)

        algo_clazz = PygmoSolver.ALGORITHMS.get(algorithm, None)
        if algo_clazz is None:
            raise AttributeError(
                f"Указанный алгоритм ({algorithm}) недоступен. "
                "Доступные алгоритмы: "
                f"{', '.join(list(PygmoSolver.ALGORITHMS.keys()))}"
            )

        self.algorithm = pg.algorithm(algo_clazz(*args, **kwargs))
        self.algorithm.set_verbosity(5)

        self.archi = None
        if n_islands > 1:
            self.archi = pg.archipelago(
                n=n_islands,
                algo=self.algorithm,
                prob=self.problem,
                pop_size=70,
                seed=41,
            )
            print(self.archi)

        self.popsize = popsize

        # check_unknown_options(unknown_options)

    def minimize(self, x0: List[float]) -> OptimizeResult:
        pop = pg.population(self.problem, size=self.popsize, seed=0)

        if x0 is not None:
            pop.set_x(0, x0)

        if self.archi is not None:
            self.archi.evolve()
            self.archi.wait()
            print(f"\n\nChampions f: {self.archi.get_champions_f()}\n\n")

            return OptimizeResult(
                # pop=res,
                # TODO: здесь надо выбрать остров-чемпион и его x
                x=self.archi.get_champions_f()[0],
                fun=self.archi.get_champions_f()[0],
                # jac=g[:-1],
                # nit=int(majiter),
                # nfev=res.problem.get_fevals(),
                # njev=res.problem.get_gevals(),
                # status=int(mode),
                # message=exit_modes[int(mode)],
                # TODO: надо возвращать реальный статус success, но pygmo
                # возвращает только population и не возвращает результат
                # scipy.optimize.minimize()
                success=True,
            )

        res = self.algorithm.evolve(pop)

        return OptimizeResult(
            pop=res,
            x=res.champion_x,
            fun=res.champion_f[0],
            # jac=g[:-1],
            # nit=int(majiter),
            nfev=res.problem.get_fevals(),
            njev=res.problem.get_gevals(),
            # status=int(mode),
            # message=exit_modes[int(mode)],
            # TODO: надо возвращать реальный статус success, но pygmo
            # возвращает только population и не возвращает результат
            # scipy.optimize.minimize()
            success=True,
        )
