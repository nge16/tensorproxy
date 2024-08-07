"""Класс, описывающий объект типа 'Компрессорный цех'."""

from typing import Literal, Annotated
from pydantic import BaseModel, Field

from .object import TObject, EdgeObject
from .geometry import MultiPoint
from .unit import Unit
from .dynamics import Dynamic


class ShopProps(BaseModel):
    """Свойства объекта типа 'Компрессорный цех'."""

    type: TObject = "Shop"

    KS: str | None = Field(
        None,
        title="Наименование компрессорной станции",
        description="Идентификатор компрессорной станции",
    )

    n: int | None = Field(
        None,
        title="Номер цеха",
        description="Номер цеха",
    )

    state: Annotated[
        Literal["on", "off", "tr"],
        Dynamic[int]("Shop", "state"),
    ] = Field(
        "on",
        title="Состояние работы",
        description="'on' – задействован в транспортировке газа, "
        "'off' – отключен, tr – работает на проход",
    )

    Ta: Annotated[float | None, Unit("°C", coeff_b_SI=273.15)] = Field(
        5.0,
        title="Температура воздуха",
        description="Температура воздуха [°C]",
    )

    Pa: Annotated[float | None, Unit("кПа", 1e3)] = Field(
        101.325,
        title="Давление воздуха",
        description="Давление воздуха [кПа]",
    )

    dPin: Annotated[float | None, Unit("МПа (абс.)", 1e6)] = Field(
        0.12,
        title="Потери давления газа на входе",
        description="Потери давления газа в трубопроводах и оборудовании "
        "на входе в цех [МПа (абс.)]",
    )

    dPout: Annotated[float | None, Unit("МПа (абс.)", 1e6)] = Field(
        0.08,
        title="Потери давления газа на выходе",
        description="Потери давления газа в трубопроводах и оборудовании "
        "на выходе цеха [МПа (абс.)]",
    )

    E: Annotated[
        float | None,
        Dynamic[int]("Shop", "E"),
    ] = Field(
        None,
        title="Cтепень сжатия по цеху",
        description="Степень сжатия с учетом потерь давления газа во входной "
        "и выходной обвязке КЦ [безр.]",
    )

    Qg: Annotated[
        float | None,
        Unit("млн м3/сут", 1e6 / (60 * 60 * 24)),
    ] = Field(
        0.1,
        title="Расчетный расход топливного газа",
        description="Расчетный расход топливного газа [млн м3/сут]",
    )

    etam: float = Field(
        0.95,
        title="Механический к.п.д. ЦБН цеха",
        description="Механический к.п.д. ЦБН цеха [безр.]",
    )

    etap: float = Field(
        0.80,
        title="Политропный к.п.д. ЦБН цеха",
        description="Политропный к.п.д. ЦБН цеха [безр.]",
    )

    kcn: float = Field(
        0.99,
        title="Коэффициент технического состояния ЦБН по мощности",
        description="Коэффициент технического состояния ЦБН "
        "по мощности [безр.]",
    )

    emin: float = Field(
        1.1,
        title="Минимально-допустимая степень сжатия",
        description="Минимально-допустимая степень сжатия [безр.]",
    )

    emax: float = Field(
        1.7,
        title="Максимально-допустимая степень сжатия",
        description="Максимально-допустимая степень сжатия [безр.]",
    )
    
    
    N: Annotated[
        float | None,
        Unit("МВт", 1e6),
    ] = Field(
        25.0,
        title="Потребляемая мощность",
        description="Потребляемая мощность [МВт]",
    )
    
    Na: Annotated[
        float | None,
        Unit("МВт", 1e6),
    ] = Field(
        None,
        title="Располагаемая мощность цеха (компрессорной станции)",
        description="Располагаемая мощность цеха (компрессорной станции) [МВт]",
    )
    
    We: Annotated[
        float | None,
        Unit("кВт*час", 1e3/(60*60)),
    ] = Field(
        None,
        title="Расход электроэнергии на компримирование",
        description="Расход электроэнергии на компримирование [кВт*час]",
    )
    
    r0: Annotated[
        float | None,
        Unit("кг/м3"),
    ] = Field(
        None,
        title="Плотность газа на входе в цех",
        description="Плотность газа на входе в цех [кг/м3]",
    )
    
    Wa: Annotated[
        float | None,
        Unit("кВт*час", 1e3/(60*60) ),
    ] = Field(
        None,
        title="Расход электроэнергии на АВО",
        description="Расход электроэнергии на АВО [кВт*час]",
    )


class Shop(EdgeObject[MultiPoint, ShopProps]):
    """Компрессорный цех."""

    def get_state_int(self) -> int:
        """Возвращает целое число, характеризующее работу объекта.

        Returns:
            int: целое число, характеризующее работу объекта
        """
        return {"on": 0, "off": 1, "tr": 2}[self.props.state]
