"""Класс, описывающий объект типа 'Поставщик (приток) газа'."""

from typing import Literal, Annotated
from pydantic import BaseModel, Field

from .object import TObject, BOUNDARY_TYPE, NodeObject
from .geometry import Point
from .unit import Unit
from .dynamics import Dynamic


class InProps(BaseModel):
    """Свойства объекта типа 'Поставщик (приток) газа'."""

    type: TObject = "In"

    name: str = Field(
        ...,
        title="Наименование",
    )

    state: Annotated[
        Literal["on", "off"],
        Dynamic[int]("In", "state"),
    ] = Field(
        "on",
        title="Состояние работы",
        description="'on' – задействован в транспортировке газа, "
        "'off' – отключен",
    )

    ctype: Literal["e", "i"] = Field(
        "e",
        title="Тип поставщика",
        description="'e' – вход в систему МГ, 'i' – попутный приток газа",
    )

    P: Annotated[
        float | None,
        Unit("МПа (абс.)", 1e6),
        Dynamic[float]("In", "P"),
    ] = Field(
        None,
        title="Давление газа [МПа (абс.)]",
    )

    Q: Annotated[
        float | None,
        Unit("млн м3/сут", 1e6 / (60 * 60 * 24)),
        Dynamic[float]("In", "Q"),
    ] = Field(
        None,
        title="Коммерческий расход газа, поступающий "
        "от поставщика [млн м3/сут]",
    )

    T: Annotated[
        float | None,
        Unit("°C", coeff_b_SI=273.15),
        Dynamic[float]("In", "T"),
    ] = Field(
        ...,
        title="Температура газа [°C]",
    )

    Ro: float = Field(
        0.68,
        title="Плотность газа при стандартных условиях [кг/м3]",
    )

    _boundary_type: BOUNDARY_TYPE | None = None
    """Тип граничного условия: задано давление или расход."""


class In(NodeObject[Point, InProps]):
    """Поставщик (приток) газа."""

    def validate_before_calculation(self):
        """Выполняет проверку объекта перед запуском расчета
        на предмет корректности данных.

        Raises:
            ValueError: при наличии ошибок в данных

        """
        self.props._boundary_type = self.get_boundary_type()
        if self.props._boundary_type == "null":
            raise ValueError(
                f"У поставщика {self.props.name} "
                "не задано ни давление, ни расход газа"
            )

    def get_boundary_type(self, use_cache: bool = True) -> BOUNDARY_TYPE:
        """Определяет тип заданных для поставщика граничных условий.

        Args:
            use_cache (bool, optional): использовать ли кэшированное значение.
              True по умолчанию.

        Returns:
            BOUNDARY_TYPE: тип граничных условий
        """
        if use_cache and self.props._boundary_type is not None:
            return self.props._boundary_type

        btype: BOUNDARY_TYPE | None = None
        if self.has_dyn_value("P"):
            btype = "P"
        elif self.has_dyn_value("Q"):
            btype = "Q"

        if btype is None:
            if self.props.P is not None and self.props.Q is None:
                btype = "P"
            elif self.props.Q is not None and self.props.P is None:
                btype = "Q"

        self.props._boundary_type = btype or "null"
        return self.props._boundary_type

    def invalidate_cache(self):
        self.get_boundary_type(use_cache=False)
        super().invalidate_cache()
