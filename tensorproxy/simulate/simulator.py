from typing import List
from abc import ABC

from tensorproxy.simulate.flowsheet import Flowsheet
from tensorproxy.utils import LOG


class Simulator(ABC):
    """Базовый класс для работы с API симуляторов."""

    def __init__(self):
        self.logger = LOG.create_logger(self.__class__.__name__)

    def load_flowsheet(self, **kwargs) -> Flowsheet | None:
        """Загружает технологическую схему.

        Returns:
            Flowsheet: загруженная схема или None
        """
        return NotImplemented

    def save_flowsheet(self, flowsheet: Flowsheet, **kwargs) -> None:
        """Сохраняет технологическую схему."""
        raise NotImplementedError

    def calculate_flowsheet(
        self, flowsheet: Flowsheet | None = None, **kwargs
    ) -> List[str] | None:
        """Выполняет расчет с ранее заданными исходными данными.

        Args:
            flowsheet (Flowsheet | None, optional): Технологическая схема,
            для которой должен быть выполнен расчета.

        Returns:
            List[str]: список ошибок или None в случае их отсутствия
        """
        raise NotImplementedError

    def shutdown(self):
        """Завершает все процессы моделирования, выполняет очистку."""
        pass
