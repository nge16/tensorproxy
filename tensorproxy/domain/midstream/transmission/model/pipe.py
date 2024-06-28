"""Класс, описывающий объект типа 'Газопровод'."""

from typing import Literal, Annotated
from pydantic import BaseModel, Field

from .object import TObject, EdgeObject
from .geometry import LineString
from .unit import Unit
from .dynamics import Dynamic


class PipeProps(BaseModel):
    """Свойства объекта типа 'Газопровод'."""

    type: TObject = "Pipe"

    name: str | None = Field(
        None,
        title="Наименование участка газопровода",
    )

    ptype: Literal["t", "b"] | None = Field(
        None,
        description="'t' – газопровод (по умолчанию), 'b' – перемычка",
    )

    state: Annotated[
        Literal["on", "off"],
        Dynamic[int]("Pipe", "state"),
    ] = Field(
        "on",
        title="Состояние работы",
        description="'on' – задействован в транспортировке газа, "
        "'off' – отключен",
    )

    X0: Annotated[float | None, Unit("км", 1e3)] = Field(
        None,
        description="Километр начала [км]",
    )

    X1: Annotated[float | None, Unit("км", 1e3)] = Field(
        None,
        description="Километр конца [км]",
    )

    L: Annotated[float | None, Unit("км", 1e3)] = Field(
        None,
        title="Протяженность [км]",
        description="Протяженность газопровода [км]",
    )

    D: Annotated[float | None, Unit("мм", 1e-3)] = Field(
        None,
        description="Внешний диаметр трубы [мм]",
    )

    Di: Annotated[float | None, Unit("мм", 1e-3)] = Field(
        None,
        description="Внутренний диаметр трубы [мм]",
    )

    Dw: Annotated[float | None, Unit("мм", 1e-3)] = Field(
        None,
        description="Диаметр трубы для расчета запаса газа [мм]",
    )

    Pa: Annotated[float | None, Unit("МПа (абс.)", 1e6)] = Field(
        None,
        description="Максимально допустимое расчетное давление газа "
        "в газопроводе [МПа (абс.)]",
    )

    E: float = Field(
        0.95,
        description="Коэффициент гидравлической эффективности трубопровода",
    )

    Kto: float = Field(
        1.3,
        description="Коэффициент теплообмена трубы с внешней "
        "средой [Вт/(м2○К)]",
    )

    k: Annotated[float | None, Unit("мм", 1e-3)] = Field(
        0.03,
        description="Коэффициент эквивалентной шероховатости трубы [мм]",
    )

    Te: Annotated[float | None, Unit("°C", coeff_b_SI=273.15)] = Field(
        5.0,
        description="Температура окружающей среды (грунта) [°C]",
    )


class Pipe(EdgeObject[LineString, PipeProps]):
    """Линейная часть любого трубопровода, предназначенного для
    транспортирования природного газа, в том числе линейная часть
    магистральных газопроводов, магистральных распределительных
    газопроводов, газопроводов-отводов, лупингов, а также перемычки.
    """
