from typing import List

from tensorproxy.simulate.simulator import Simulator
from tensorproxy.simulate.flowsheet import Flowsheet

from tensorproxy.domain.midstream.transmission.gas_transmission_system import (
    GasTransmissionSystem,
)

from tensorproxy.simulate.arifma_hydraulics.core import solve_steady_state
from .hydro_utils import get_hydraulics_params


class ArifmaSimulator(Simulator):
    """Базовый класс для работы с API симуляторов."""

    def calculate_flowsheet(
        self,
        flowsheet: Flowsheet | None = None,
        model: GasTransmissionSystem = None,
        tol: float = 1e-1,
        **kwargs
    ) -> List[str] | None:

        if model is None:
            raise ValueError("В перечне атрибутов не указана модель ГТС")

        x0, params = get_hydraulics_params(model, 0, False)
        xopt, eps = solve_steady_state(x0, params, tol=tol)

        if eps > tol:
            raise Exception("Не удалось найти начальное приближение")

        return xopt, eps
