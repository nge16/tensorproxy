from typing import List, Callable, Any
from abc import ABC, abstractmethod

from tensorproxy.domain import DomainParam
from .simulation_service import SimulationService


class SimulationModel(ABC):
    """Класс для представления расчетной модели."""

    def __init__(self, simulation_service: SimulationService) -> None:
        self.simulation_service = simulation_service

        self.__fsetters_x: List[Callable[[Any], None]] | None = None

    @property
    @abstractmethod
    def domain(self) -> List[DomainParam]:
        """Абстрактный метод для переопределения в наследниках.

        Должен возвращать список параметров, описывающих область определения
        модели.

        Returns:
            List[DomainParam]: список ``DomainParam``
        """
        return NotImplemented

    @property
    def __fset_x(self) -> List[Callable[[Any], None]]:
        """Возвращает список функций для внесения исходных данных X
        в расчетную модель.

        Returns:
            List[Callable[[], None]]: функции (сеттеры)
        """
        if self.__fsetters_x is None:
            self.__fsetters_x = []
            for dp in self.domain:
                setter = self.simulation_service.fset_obj_attr(
                    obj_id=dp.obj_id,
                    attr_id=dp.attr_id,
                    unit=dp.unit,
                    **dp.extra_params,
                )
                self.__fsetters_x.append(setter)
        return self.__fsetters_x

    def assign_x(self, x: List) -> None:
        """Устанавливает значения параметров объектов расчетной модели.

        Args:
            x (List): значения параметров области определения модели

        Raises:
            ValueError: в случае, если длина массива `x` отличается от длины
            массива сеттеров (`self._fsetters`)
        """
        if len(self.__fset_x) != len(x):
            raise ValueError(
                f"Число переменных ({len(x)}) должно соответствовать числу "
                f"описанию ({len(self.__fset_x)}) области определения "
                "моделируемой функции ``domain``"
            )

        for x, fset in zip(x, self.__fset_x):
            fset(x)
