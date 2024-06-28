from typing import List, Callable, Any
from abc import ABC, abstractmethod

from smt.utils.design_space import DesignSpace, FloatVariable

from tensorproxy.simulate import Flowsheet
from tensorproxy.simulate.simulation_result import (
    CallableSimulationResult,
)

from tensorproxy.domain import DomainParam


class SimulationTask(ABC):
    """Имитационная задача.

    Args:
        flowsheet (Flowsheet): технологическая схема, на которой
        решается задача
    """

    def __init__(self, flowsheet: Flowsheet) -> None:
        super().__init__()
        self.flowsheet = flowsheet

        self.__fsetters_x: List[Callable[[Any], None]] | None = None
        self.__design_space: DesignSpace | None = None

    @property
    @abstractmethod
    def design_params(self) -> List[DomainParam]:
        """Список параметров, описывающих область определения
        моделируемой функции.

        Returns:
            List[DomainParam]: описание области определения
        """
        return NotImplemented

    @property
    def __fset_x(self) -> List[Callable[[Any], None]]:
        """Возвращает список функций для внесения исходных данных X
        в расчетную схему.

        Returns:
            List[Callable[[], None]]: функции (сеттеры)
        """
        if self.__fsetters_x is None:
            self.__fsetters_x = []
            for dp in self.design_params:
                setter = self.flowsheet.fset_obj_attr(
                    obj_id=dp.obj_id,
                    attr_id=dp.attr_id,
                    unit=dp.unit,
                    **dp.extra_params
                )
                self.__fsetters_x.append(setter)
        return self.__fsetters_x

    @property
    def design_space(self) -> DesignSpace:
        if self.__design_space is None:
            self.__design_space = DesignSpace(
                [FloatVariable(p.lower, p.upper) for p in self.design_params]
            )
        return self.__design_space

    def assign_x(self, xs: List) -> None:
        """Устанавливает значения параметров объектов технологической схемы.

        Args:
            xs (List): список значений

        Raises:
            ValueError: в случае, если длина массива `x` отличается от длины
            массива сеттеров (`self._fsetters`)
        """
        if len(self.__fset_x) != len(xs):
            raise ValueError(
                "Число переменных должно соответствовать числу описанию "
                "области определения моделируемой функции ``design_space``"
            )

        for x, fset in zip(xs, self.__fset_x):
            fset(x)

    @property
    @abstractmethod
    def fget_y(self) -> List[CallableSimulationResult]:
        """Возвращает список геттеров для Y (результатов моделирования).

        Returns:
            List[CallableSimulationResult]: список геттеров
        """
        return NotImplemented
