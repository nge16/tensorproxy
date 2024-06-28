from typing import Any, Callable
from dataclasses import dataclass


@dataclass
class CallableSimulationResult:
    """Вспомогательный класс, содержащий именованный результат
    численного моделирования с функцией для его получения.
    """

    fres: Callable[[Any], float | int]
    """Функция, возвращающая результат моделирования (Y)"""

    name: str | None = None
    """Наименование результата моделирования."""

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.fres(*args, **kwds)
