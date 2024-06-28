from dataclasses import dataclass, field
from collections import defaultdict

from .interface import ObjId, AttrId, AttrUnit


@dataclass
class DomainParam:
    """Исходные данные для симуляций, варьирование которых
    представляет интерес для построения суррогатной модели.

    """

    label: str
    """Имя параметра для нужд отладки."""

    obj_id: ObjId
    """Идентификатор объекта"""

    attr_id: AttrId
    """Идентификатор атрибута объекта"""

    unit: AttrUnit = None
    """Единица измерения."""

    lower: float | int = None
    """Нижняя граница значения атрибута."""

    upper: float | int = None
    """Верхняя граница значения атрибута."""

    extra_params: defaultdict[dict] = field(
        default_factory=lambda: defaultdict(dict)
    )
    """Дополнительная информация."""
