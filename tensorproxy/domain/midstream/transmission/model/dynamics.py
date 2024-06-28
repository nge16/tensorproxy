"""Инструменты для работы с динамикой значений атрибутов объектов
расчетной схемы."""

from typing import Generic, TypeVar, Dict, Tuple
from dataclasses import dataclass

from collections import OrderedDict, namedtuple
import copy
import numpy as np

DynamicsShape = namedtuple(
    "DynamicsShape", ["T", "max_object_id"], defaults=[None]
)
"""Размерность динамики параметров: (максимальный временной индекс,
максимальный id объекта)"""

T = TypeVar("T")


@dataclass
class Dynamic(Generic[T]):
    """Динамика значений параметра.

    Используется в аннотации свойств объектов.
    Для свойства, отмеченного Annotated[..., Dynamic[...](...)],
    создается инстанс класса `Dynamic`, который содержит
    динамику значений этого свойства для всех объектов этого класса.

    Example::

      >>> P: Annotated[
      >>>     float | None,
      >>>     Unit("МПа (абс.)", 1e6),
      >>>     Dynamic[float]("P"),
      >>> ] = Field(
      >>>     None,
      >>>     title="Давление газа [МПа (абс.)]",
      >>> )

    """

    class_name: str
    """Имя класса объектов."""

    prop_name: str
    """Наименование параметра"""

    dynamics: Dict[Tuple[int, int], T] | None = None
    """Динамика значений параметра в формате словаря
    (<id объекта>, <time>): <value>"""

    def add_dynamics(self, id: int, dynamics: Dict[int, T]):
        """Добавляет динамику значений свойства (атрибута) `prop_name`.

        Значения могут быть заданы в произвольные моменты времени.
        Незаданные значения впоследствие могут быть рассчитаны
        (интерполированы) с помощью метода `to_dense_array`.

        Args:
            id (int): идентификатор объекта
            dynamics (Dict[int, T]): динамика значений в формате
              <временной слой>: <значение>
        """
        times, values = list(dynamics.keys()), list(dynamics.values())
        ids = np.repeat(id, len(dynamics)).tolist()

        if self.dynamics is None:
            self.dynamics = OrderedDict({})

        self.dynamics |= {
            (id, time): data for id, time, data in zip(ids, times, values)
        }

        # сортировка по времени
        self.dynamics = OrderedDict(
            sorted(self.dynamics.items(), key=lambda item: item[0][1])
        )

    def clear(self, id: int | None = None) -> None:
        """Очищает значения динамики параметров для объекта
        с идентификатором `id` или для всех объектов, если
        `id` равен None.

        Args:
            id (int | None ): идентификатор объекта, для которого
                выполняется очистка значений динамики, или None,
                если требуется удалить информацию о динамике значений
                всех объектов

        """
        self.dynamics = (
            OrderedDict(
                {
                    (i, time): data
                    for (i, time), data in self.dynamics.items()
                    if i != id
                }
            )
            if id is not None and self.dynamics is not None
            else None
        )

    def get_dynamics(self, id: int) -> Dict[int, T]:
        """Возвращает динамику значений свойства (атрибута) для объекта
        с идентификатором `id`.

        Args:
            id (int): идентификатор объекта

        Returns:
            Dict[int, T]: значение параметров в разных временных слоях
        """
        return OrderedDict(
            {
                time: value
                for (i, time), value in self.dynamics.items()
                if i == id
            }
            if self.dynamics is not None
            else {}
        )

    def has_value(self, id: int, t: int | None = None) -> bool:
        """Проверяет, задавалось ли значение для указанного объекта
        на указанном временном слое.

        Если временной слой не указан, проверяет, задавалось ли значение
        для объекта с указанным `id`.

        Args:
            id (int): идентификатор объекта
            t (int | None): временной слой или None

        Returns:
            bool: признак, задавалось ли значение
        """
        if t is None:
            return (
                any([id == i for (i, _) in self.dynamics.keys()])
                if self.dynamics is not None
                else False
            )
        return (
            (id, t) in self.dynamics.keys()
            if self.dynamics is not None
            else False
        )

    def to_dense_array(
        self, id: int, num_layers: int | None = None
    ) -> np.ndarray | None:
        """Возвращает динамику значений свойства (атрибута) для объекта
        с идентификатором `id`.

        Размерность возвращаемого 1-D массива равна максимально заданному
        временному слою. Для незаданных значений выполняется линейная
        интерполяция ближайших заданных значений.

        Args:
            id (int): идентификатор объекта
            num_layers (int | None): число временных слоев, для которых
              необходимо вернуть значения. None по умолчанию (возвращаются
              все известные значения).

        Returns:
            `T`: массив динамики значений или None, если массив пустой (None)
        """
        if self.dynamics is None:
            return None

        # словарь <временной слой>: <значение>
        arr = {
            tup[1]: value
            for tup, value in zip(self.dynamics.keys(), self.dynamics.values())
            if tup[0] == id
        }
        max_time = max(arr.keys())

        if num_layers is not None:
            max_time = max(max_time, num_layers)

        arr = np.array(
            [arr.get(t) for t in range(max_time + 1)], dtype=np.float32
        )
        mask = np.isnan(arr)

        # линейная интерполяция для незаданных значений
        arr[mask] = np.interp(
            np.flatnonzero(mask), np.flatnonzero(~mask), arr[~mask]
        )

        cut = (
            min(max_time, num_layers)
            if num_layers is not None
            else max_time + 1
        )
        return arr[:cut]

    def to_dict(self) -> dict:
        return {
            self.prop_name: self.dynamics,
        }

    def from_dict(self, dynamics: Dict[Tuple[int, int], T]) -> None:
        dynamics = copy.deepcopy(dynamics)
        # сортировка по времени
        self.dynamics = OrderedDict(
            sorted(dynamics.items(), key=lambda item: item[0][1])
        )
