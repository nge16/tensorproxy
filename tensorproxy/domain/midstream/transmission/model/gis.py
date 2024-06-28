"""Класс, описывающий объект типа 'Газоизмерительная станция'."""

from typing import Literal, Annotated
from pydantic import BaseModel, Field

from .object import TObject, EdgeObject
from .geometry import MultiPoint
from .dynamics import Dynamic


class GISProps(BaseModel):
    """Свойства объекта типа 'Газоизмерительная станция'."""

    type: TObject = "GIS"

    name: str | None = Field(
        None,
        title="Наименование",
    )

    state: Annotated[
        Literal["on", "off"],
        Dynamic[int]("GIS", "state"),
    ] = Field(
        "on",
        title="Состояние работы",
        description="'on' – задействован в транспортировке газа, "
        "'off' – отключен",
    )


class GIS(EdgeObject[MultiPoint, GISProps]):
    """Газоизмерительная станция."""
