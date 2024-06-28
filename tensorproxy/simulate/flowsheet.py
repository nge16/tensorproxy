from typing import Callable, Any
from abc import ABC, abstractmethod

from tensorproxy.domain import ObjId, AttrId, AttrUnit


class Flowsheet(ABC):
    """Базовый класс для работы с технологическими (расчетными) схемами."""

    def __init__(self) -> None:
        super().__init__()

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

    def reset_calculations(self) -> None:
        """Сбрасывает результаты предыдущих расчетов.

        Метод переопределяется в наследниках.
        """
        pass

    def to_dict(self) -> dict:
        """Преобразует расчетную схему в словарь.

        Словарь должен иметь структуру универсального формата
        обмена данными и результатами расчетов для данного типа
        схемы.

        Returns:
            dict: словарь с исходными данными и результатами
            расчетов.
        """
        pass
