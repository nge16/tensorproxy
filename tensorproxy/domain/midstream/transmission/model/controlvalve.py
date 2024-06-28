"""Класс, описывающий объект типа 'Кран-регулятор'."""

from typing import Literal, Annotated
from pydantic import BaseModel, Field

from .object import TObject, EdgeObject
from .geometry import MultiPoint
from .unit import Unit
from .dynamics import Dynamic


class ControlValveProps(BaseModel):
    """Свойства объекта типа 'Кран-регулятор'."""

    type: TObject = "ControlValve"

    name: str | None = Field(
        None,
        title="Наименование",
    )

    state: Annotated[
        Literal["on", "off"],
        Dynamic[int]("ControlValve", "state"),
    ] = Field(
        "on",
        title="Состояние работы",
        description="'on' – задействован в транспортировке газа, "
        "'off' – отключен",
    )

    D: Annotated[float | None, Unit("мм", 1e-3)] = Field(
        None,
        description="Диаметр полностью открытого крана [мм]",
    )

    op: Annotated[
        float | None,
        Dynamic[float]("ControlValve", "op"),
    ] = Field(
        0.5,
        description="Процент открытия крана [доля единицы]",
    )


class ControlValve(EdgeObject[MultiPoint, ControlValveProps]):
    """Кран-регулятор."""
