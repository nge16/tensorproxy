from typing import Callable, Any
from abc import ABC, abstractmethod

from tensorproxy.domain import ObjId, AttrId, AttrUnit


class SimulationService(ABC):
    r"""Базовый класс сервисов для работы с API симуляторов."""

    def reset_calculations(self) -> None:
        """Сбрасывает результаты ранее выполненного расчета.

        Метод для переопределения в наследникахю
        """

        pass

    @abstractmethod
    def calculate(self) -> bool:
        """Запускает расчет в симуляторе."""
        pass

    @abstractmethod
    def load_flowsheet(self, filepath: str) -> bool:
        """Загружает технологическую схему.

        Args:
            filepath (str): абсолютный путь до файла с расчетной схемой

        Returns:
            bool: признак, загружена ли схема
        """
        return NotImplemented

    @abstractmethod
    def fget_obj_attr(
        self, obj_id: ObjId, attr_id: AttrId, unit: AttrUnit = None, **kwargs
    ) -> Callable[[], Any]:
        """Возвращает функцию для получения значения атрибута объекта
        в указанных единицах измерения.

        Args:
            obj_id (ObjId): идентификатор объекта
            attr_id (AttrId): идентификатор атрибута объекта
            unit (AttrUnit): единица измерения (если не указано, значение
            возвращается в СИ)

        Returns:
            Callable[[], Any] : функция для получения значения атрибута
        """
        return NotImplemented

    def fset_obj_attr(
        self, obj_id: ObjId, attr_id: AttrId, unit: AttrUnit = None, **kwargs
    ) -> Callable[[float], str] | None:
        """Возвращает функцию для присвоения значение атрибуту объекта.

        Args:
            obj_id (ObjId): идентификатор объекта
            attr_id (AttrId): идентификатор атрибута объекта
            unit (AttrUnit): единица измерения (если не указано, значение
            возвращается в СИ)

        Returns:
            Callable[[float], str] : функция для присвоения значения
            атрибуту объекта
        """
        return NotImplemented
