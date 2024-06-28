"""Интерфейсные классы для бесшовного взаимодействия с разными симуляторами.
"""

from dataclasses import dataclass

from .parameters import Parameter


@dataclass
class ObjId:
    """Идентификатор объекта."""

    iid: int = None
    """Целочисленный идентификатор.
    """

    sid: str = None
    """Строковый идентификатор.
    """


@dataclass
class AttrId:
    """Идентификатор атрибута (параметра, свойства) объекта."""

    param: Parameter = None
    """Идентификатор атрибута."""

    extra: str = None
    """Дополнительные сведения об атрибуте.
    """


@dataclass
class AttrUnit:
    """Единица измерения."""

    unit: str = None
    """Строковое наименование единицы измерения."""
