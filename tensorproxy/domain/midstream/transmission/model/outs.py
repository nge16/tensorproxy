"""Класс, описывающий объект типа 'Потребитель (сток) газа'."""

from typing import Literal, Annotated
from pydantic import BaseModel, Field

from .object import TObject, BOUNDARY_TYPE, NodeObject
from .geometry import Point
from .unit import Unit
from .dynamics import Dynamic


class OutProps(BaseModel):
    """Свойства объекта типа 'Потребитель (сток) газа'."""

    type: TObject = "Out"

    name: str = Field(
        ...,
        title="Наименование",
    )

    state: Annotated[
        Literal["on", "off"],
        Dynamic[int]("Out", "state"),
    ] = Field(
        "on",
        title="Состояние работы",
        description="'on' – потребляет газ, 'off' – не потребляет газ",
    )

    ctype: Literal["e", "i"] = Field(
        "i",
        title="Тип потребителя ",
        description="'o' – выход из ГТС, 'i' – ГРС или другой "
        "попутный потребитель газа (по умолчанию)",
    )

    P: Annotated[
        float | None,
        Unit("МПа (абс.)", 1e6),
        Dynamic[float]("Out", "P"),
    ] = Field(
        None,
        title="Давление газа [МПа (абс.)]",
    )

    Q: Annotated[
        float | None,
        Unit("млн м3/сут", 1e6 / (60 * 60 * 24)),
        Dynamic[float]("Out", "Q"),
    ] = Field(
        None,
        title="Коммерческий расход газа, поступающий "
        "потребителю [млн м3/сут]",
    )
    
    Qmin: Annotated[
        float | None,
        Unit("млн м3/сут", 1e6 / (60 * 60 * 24)),
        # Dynamic[float]("In", "Q"),
    ] = Field(
        ...,
        title="Ограничение по минимальному объему подачи газа "
        "для потребителя [млн м3/сут]",
    )
    
    
    Qmax: Annotated[
        float | None,
        Unit("млн м3/сут", 1e6 / (60 * 60 * 24)),
        # Dynamic[float]("In", "Q"),
    ] = Field(
        ...,
        title="Ограничение по максимальному подачи газа "
        "для потребителя [млн м3/сут]",
    )
    
    Pmin: Annotated[
        float | None,
        Unit("МПа (абс.)", 1e6),
        # Dynamic[float]("In", "P"),
    ] = Field(
        None,
        title="Ограничения по минимальному значению давления газа [МПа (абс.)]",
    )
    
    Pmax: Annotated[
        float | None,
        Unit("МПа (абс.)", 1e6),
        # Dynamic[float]("In", "P"),
    ] = Field(
        None,
        title="Ограничения по максимальному значению давления газа [МПа (абс.)]",
    )

    _boundary_type: BOUNDARY_TYPE | None = None
    """Тип граничного условия: задано давление или расход."""


class Out(NodeObject[Point, OutProps]):
    """Потребитель (сток) газа."""

    def validate_before_calculation(self):
        """Выполняет проверку объекта перед запуском расчета
        на предмет корректности данных.

        Raises:
            ValueError: при наличии ошибок в данных

        """
        self.props._boundary_type = self.get_boundary_type()
        if self.props._boundary_type == "null":
            raise ValueError(
                f"У потребителя {self.props.name} "
                "не задано ни давление, ни расход газа"
            )

    def get_boundary_type(self, use_cache: bool = True) -> BOUNDARY_TYPE:
        """Определяет тип заданных для потребителя граничных условий.

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
