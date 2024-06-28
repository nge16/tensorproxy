from typing import List, Any, Tuple, Dict, Callable

from abc import ABC, abstractmethod
import numpy as np

from tensorproxy.simulate import SimulationModel, CallableSimulationResult
from tensorproxy.domain import DomainParam


class SurrogateModel(ABC):
    """Базовый класс для работы с суррогатными моделями.

    Args:
      simulation_model (SimulationModel): полноразмерная имитационная модель
    """

    def __init__(
        self, simulation_model: SimulationModel, name: str | None = None
    ) -> None:
        self._simulation_model = simulation_model
        self.model_name = name

    @property
    def simulation_model(self) -> SimulationModel:
        return self._simulation_model

    @property
    def domain(self) -> List[DomainParam]:
        """Возвращает список параметров, описывающих
        область определения модели.

        Предназначен для переопределения в наследниках.

        Returns:
            List[DomainParam]: список ``DomainParam``
        """
        return self.simulation_model.domain

    @property
    @abstractmethod
    def results(self) -> List[CallableSimulationResult]:
        """Абстрактное свойство для переопределения в наследниках.

        Должно возвращать описание результатов моделирования, т.е.
        выходных данных суррогатной модели.

        Returns:
            List[CallableSimulationResult]: список результирующих параметров
              суррогатной модели
        """
        return NotImplemented

    @abstractmethod
    def fit(
        self,
        x: Any,
        y: Any,
        validation_data: Tuple | None = None,
        verbose: int = 0,
    ) -> None:
        """Выполняет обучение суррогатной модели.

        Args:
            x (Any): входные данные
            y (Any): выходные данные
            validation_data: Tuple `(x_val, y_val)`, numpy arrays
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self, x: Any, y: Any, metrics: Dict[str, Callable]
    ) -> Dict[str, float]:
        """Возвращает метрики на тестовой выборке."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: List, args=()) -> np.ndarray:
        """Выполняет прогноз значений функции.

        Args:
            x (Any): входные данные
        """
        raise NotImplementedError

    @abstractmethod
    def predict_gradient(self, x: List, args=()) -> np.ndarray:
        """Выполняет прогноз градиента dy_dx.

        Args:
            x (List): входные данные
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> bool:
        """Загружает ранее сохраненную суррогатную модель.

        Args:
            path (str): путь до каталога/файла

        Returns:
            bool: True, если загружено успешно
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> bool:
        """Выполняет сохранение суррогатной модели.

        Args:
            path (str): путь до каталога/файла для сохранения модели

        Returns:
            bool: True, если сохранено успешно
        """
        raise NotImplementedError
