from typing import List, Dict, Any, Annotated, Literal
import math
from warnings import warn
from collections import defaultdict

import numpy as np

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_serializer,
    model_validator,
    ValidationInfo,
)

from tensorproxy.domain.midstream.transmission.model.unit import Unit


class Convertable(BaseModel):
    @classmethod
    def convert_from_SI(
        cls, prop_name: str, value: float | np.ndarray
    ) -> float | np.ndarray | None:
        """Конвертирует значение во внутреннюю размерность из единиц СИ.

        Args:
            prop_name (str): наименование атрибута
            value (float): значение для перевода из СИ

        Returns:
            float: значение, переведенное во внутреннюю размерность из СИ
        """
        if value is None:
            return None

        if prop_name not in cls.model_fields.keys():
            raise ValueError(
                f"Модель {cls.__name__} не содержит атрибут {prop_name}"
            )

        try:
            converter = next(
                meta
                for meta in cls.model_fields[prop_name].metadata
                if isinstance(meta, Unit)
            )
            return converter.from_SI(value)
        except Exception:
            warn(
                f"Атрибут {prop_name} класса {cls.__name__} "
                "не содержит аннотацию Unit, в связи с чем "
                "перевод размерности не выполнен"
            )

        return value

    @classmethod
    def convert_to_SI(
        cls, prop_name: str, value: float
    ) -> float | np.ndarray | None:
        """Конвертирует значение из внутренней размерности в единицы СИ.

        Args:
            prop_name (str): наименование атрибута
            value (float): значение для перевода в СИ

        Returns:
            float: значение, переведенное из внутренней размерности в СИ
        """
        if value is None:
            return None

        if prop_name not in cls.model_fields.keys():
            raise ValueError(
                f"Модель {cls.__name__} не содержит атрибут {prop_name}"
            )

        try:
            converter = next(
                meta
                for meta in cls.model_fields[prop_name].metadata
                if isinstance(meta, Unit)
            )
            return converter.to_SI(value)
        except Exception:
            warn(
                f"Атрибут {prop_name} класса {cls.__name__} "
                "не содержит аннотацию Unit, в связи с чем "
                "перевод в СИ не выполнен"
            )

        return value


class ResultsPipe(Convertable):
    """Результаты гидравлического расчета газопровода."""

    P0: Annotated[
        float,
        Unit("МПа (абс.)", 1e6),
    ] = Field(
        None,
        description="Расчетное давление газа в начале "
        "трубопровода [МПа (абс.)]",
    )

    P1: Annotated[
        float,
        Unit("МПа (абс.)", 1e6),
    ] = Field(
        None,
        description="Расчетное давление газа в конце "
        "трубопровода [МПа (абс.)]",
    )

    Q0: Annotated[
        float,
        Unit("млн м3/сут", 1e6 / (60 * 60 * 24)),
    ] = Field(
        None,
        description="Расчетный коммерческий расход газа в начале "
        "трубопровода [млн м3/сут]",
    )

    Q1: Annotated[
        float,
        Unit("млн м3/сут", 1e6 / (60 * 60 * 24)),
    ] = Field(
        None,
        description="Расчетный коммерческий расход газа в конце "
        "трубопровода [млн м3/сут]",
    )

    T0: Annotated[
        float | None,
        Unit("°C", coeff_b_SI=273.15),
    ] = Field(
        None,
        description="Расчетная температура газа в начале трубопровода [°C]",
    )

    T1: Annotated[
        float | None,
        Unit("°C", coeff_b_SI=273.15),
    ] = Field(
        None,
        description="Расчетная температура газа в конце трубопровода [°C]",
    )

    @model_serializer
    def serialize_model(self) -> List[float]:
        # Преобразование в более сжатый формат сериализации (list)
        return list(
            map(
                lambda field_name: getattr(self, field_name),
                ResultsPipe.model_fields.keys(),
            )
        )

    @model_validator(mode="before")
    @classmethod
    def validate_input_params(
        cls, values: Any, inf: ValidationInfo
    ) -> Dict[str, float]:
        if inf.mode == "json" and isinstance(values, list):
            if len(values) != len(ResultsPipe.model_fields):
                raise ValueError(
                    "Некорректное количество результатов расчетов газопровода"
                )
            return {
                key: values[i]
                for i, key in enumerate(ResultsPipe.model_fields.keys())
            }
        return values


class ResultsShop(Convertable):
    Qg: Annotated[
        float,
        Unit("млн м3/сут", 1e6 / (60 * 60 * 24)),
    ] = Field(
        ...,
        description="Расчетный расход топливного газа [млн м3/сут]",
    )


class HydraulicsResults(BaseModel):
    """Результаты гидравлического расчета объектов схемы."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
    )

    pipes: List[ResultsPipe] = Field(
        [],
        title="Результаты расчета газопроводов",
    )

    shops: List[ResultsShop] = Field(
        [],
        title="Результаты расчета компрессорных цехов",
    )

    # valves ...


class DynamicHydraulicsResults(BaseModel):
    """Результаты гидравлического расчета газотранспортной системы
    по временным слоям.


    Examples::
        results_container = tp.transmission.DynamicHydraulicsResults()
        results_container = pack_dynamic_results(
            layer,
            t=t,
            results={**x, **aux},
            container=results_container,
        )

    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
    )

    time_layers: Dict[int, HydraulicsResults] = Field(
        {},
        title="Результаты расчета по временным слоям",
    )

    indexes_pipes: List[int] = Field(
        [],
        title="Порядковые ID газопроводов",
    )

    indexes_shops: List[int] = Field(
        [],
        title="Порядковые ID компрессорных цехов",
    )

    def visualize_results(
        self,
        rtype: Literal["pipe", "shop"] = "pipe",
        res_prop_name: str = "Q0",
        max_graphs: int = 16,
        max_timesteps: int | None = None,
    ) -> None:
        """Выполняет построение графиков указанного параметра

        Args:
            res_prop_name (str, optional): наименование атрибута ResultsPipe.
                По умолчанию "Q0".
            max_graphs (int, optional): максимальное число выводимых графиков.
                По умолчанию 16.

        Raises:
            ValueError: если указанный атрибут `res_prop_name` не существует
        """
        import matplotlib.pyplot as plt

        resClazz = ResultsPipe if rtype == "pipe" else ResultsShop

        if res_prop_name not in resClazz.model_fields.keys():
            raise ValueError(
                f"Модель {resClazz.__name__} не содержит "
                f"атрибут {res_prop_name}. Доступные атрибуты: "
                f"{list(resClazz.model_fields.keys())}"
            )

        description = resClazz.model_fields[res_prop_name].description
        if description is None or len(description) == 0:
            description = f"Динамика параметра '{res_prop_name}'"

        results = defaultdict(list)
        T = max_timesteps or max(list(self.time_layers.keys()))
        for t in self.time_layers.keys():
            if t > T:
                break
            rlist = (
                self.time_layers[t].pipes
                if rtype == "pipe"
                else self.time_layers[t].shops
            )

            for i, res in enumerate(rlist):
                results[i].append(getattr(res, res_prop_name))

        # максимальное число дуг для вывода информации
        max_edges = min(max_graphs, len(results.keys()))

        M = max_edges
        fig, axs = plt.subplots(math.ceil(M / 2), 2, figsize=(15, 8))
        fig.suptitle(description)

        for i, (edge, vals) in enumerate(results.items()):
            if i > M - 1:
                break
            row, col = i // 2, i % 2
            axs[row, col].plot(list(range(T + 1)), vals)
            axs[row, col].set_title(f"Дуга {edge}")
            axs[row, col].set_ylim([min(vals) * 0.95, max(vals) * 1.05])
            axs[row, col].ticklabel_format(useOffset=False)

        for ax in axs.flat:
            ax.set(xlabel="t", ylabel=res_prop_name)

        plt.subplots_adjust(top=0.95, bottom=0.01, hspace=0.5, wspace=0.4)
