from typing import Literal, TypeVar, Generic, cast
from enum import IntEnum
import abc
import inspect
import bisect

from pydantic import BaseModel, Field, ConfigDict

from .geometry import Geometry
from .unit import Unit
from .dynamics import Dynamic
from .cachable import Cachable

TObject = Literal[
    "Pipe",
    "Shop",
    "Valve",
    "ControlValve",
    "In",
    "Out",
    "GIS",
    "Text",
    "Line",
]


class State(IntEnum):
    """Состояние объекта"""

    ACTIVE = 0
    DISABLED = 1


DEFAULT_STATE = State.ACTIVE

# тип граничных условий: задано давление / задан расход / не задано ничего
BOUNDARY_TYPE = Literal["P", "Q", "null"]

TGeom = TypeVar("TGeom")
TPROPS = TypeVar("TPROPS")


class Object(BaseModel, Generic[TGeom, TPROPS], Cachable, abc.ABC):
    """Базовый класс объекта газотранспортной системы."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    json_type: str = Field(
        "Feature",
        description="Служебное поле, указывающее на то, "
        "что информационный блок содержит информацию о сущностях "
        "предметной области (расчетных объектах)",
        serialization_alias="type",
    )

    id: int = Field(..., title="Идентификатор объекта")

    geometry: Geometry[TGeom] | None = Field(
        None,
        title="Геометрические координаты объекта",
    )

    props: TPROPS = Field(
        ...,
        alias="properties",
        title="Атрибуты объекта",
    )

    def get_state_int(self) -> int:
        """Возвращает целое число, характеризующее работу объекта.

        Returns:
            int: целое число, характеризующее работу объекта
        """
        assert hasattr(
            self.props, "state"
        ), f"Класс {self.__class__.__name__} не имеет атрибута `state`"
        return {"on": 0, "off": 1}[self.props.state]  # type: ignore

    def get_prop_value(
        self,
        prop_name: str,
        t: int | None = None,
        value_if_none: float | int = 0,
        to_SI: bool = True,
    ) -> float:
        """Возвращает значение атрибута.

        Если временной слой `t` не указан, возвращает значение атрибута
        (свойства) из статических свойств, в противном случае - из динамики
        значений этого свойства.

        Args:
            prop_name (str): наименование атрибута
            t (int | None): временной слой
            value_if_none (float | int): значение, возвращаемое в случае, если
            значение свойства равно None
            to_SI (bool): выполнить ли перевод значения в СИ.
              True по умолчанию.

        Raises:
            ValueError: если модель не содержит атрибут `prop_name`

        Returns:
            float: значение атрибута `prop_name`
        """
        assert hasattr(
            self.props, "model_fields"
        ), f"Класс {self.__class__.__name__} не имеет атрибута `model_fields`"
        if prop_name not in self.props.model_fields.keys():  # type: ignore
            raise ValueError(
                f"Модель {self.__class__.__name__} не содержит "
                f"атрибут {prop_name}"
            )

        value = (
            getattr(self.props, prop_name)
            if t is None
            else self.get_dyn_value(prop_name, t)
        ) or value_if_none

        # так как у всех типов объектов state имеет значения on и off
        if prop_name == "state":
            return self.get_state_int()

        if to_SI:
            try:
                converter = self.__class__.get_units_converter(prop_name)
                return cast(float, converter.to_SI(value))
            except Exception:
                """Атрибут не содержит аннотацию Unit."""
                pass

        return value

    @classmethod
    def get_units_converter(cls, prop_name: str) -> Unit:
        """Возвращает конвертер единиц измерения для указанного атрибута.

        Args:
            prop_name (str): наименование атрибута

        Returns:
            Unit: конвертер единиц измерения
        """
        props_clazz = cls.model_fields["props"].annotation
        if not (
            inspect.isclass(props_clazz) and issubclass(props_clazz, BaseModel)
        ):
            raise ValueError(f"Класс {cls.__name__} не имеет атрибута props")

        if prop_name not in props_clazz.model_fields.keys():
            raise ValueError(
                f"Модель {cls.__name__} не содержит " f"атрибут {prop_name}"
            )
        try:
            converter = next(
                meta
                for meta in props_clazz.model_fields[prop_name].metadata
                if isinstance(meta, Unit)
            )
            return converter
        except Exception:
            raise ValueError(
                f"Атрибут {prop_name} модели {cls.__name__} "
                f"не содержит аннотации Unit"
            )

    def add_dynamics(
        self, prop_name: str, dynamics: dict[int, float | int]
    ) -> Dynamic:
        """Добавляет значения динамического свойства `prop_name`.

        Args:
            prop_name (str): наименование свойства из набора свойств TPROPS
            dynamics (dict[int, float  |  int]): динамика значений в формате
                <временной слой>: <значение>

        Returns:
            Dynamic: инстанс Dynamic, общий для всего класса TPROPS
        """
        assert hasattr(
            self.props, "model_fields"
        ), f"Класс {self.__class__.__name__} не имеет атрибута `model_fields`"
        if prop_name not in self.props.model_fields.keys():  # type: ignore
            raise ValueError(
                f"Модель {self.__class__.__name__} не содержит "
                f"атрибут {prop_name}"
            )
        try:
            dyn = next(
                meta
                for meta in cast(BaseModel, self.props)
                .model_fields[prop_name]
                .metadata
                if isinstance(meta, Dynamic)
            )
        except Exception as err:
            raise ValueError(
                f"Атрибут {prop_name} модели {self.__class__.__name__} "
                f"не содержит аннотации Dynamic: {err}"
            )

        dyn.add_dynamics(self.id, dynamics)
        return dyn

    def clear_dynamics(self, prop_name: str) -> Dynamic:
        """Очищает информацию о динамике значение параметра `prop_name`
        для текущего объекта.

        Args:
            prop_name (str): наименование своства (атрибута), для которого
            необходимо удалить значения динамики параметра

        Returns:
            Dynamic: инстанс Dynamic
        """
        assert hasattr(
            self.props, "model_fields"
        ), f"Класс {self.__class__.__name__} не имеет атрибута `model_fields`"
        if prop_name not in self.props.model_fields.keys():  # type: ignore
            raise ValueError(
                f"Модель {self.__class__.__name__} не содержит "
                f"атрибут {prop_name}"
            )
        try:
            dyn = next(
                meta
                for meta in cast(BaseModel, self.props)
                .model_fields[prop_name]
                .metadata
                if isinstance(meta, Dynamic)
            )
        except Exception as err:
            raise ValueError(
                f"Атрибут {prop_name} модели {self.__class__.__name__} "
                f"не содержит аннотации Dynamic: {err}"
            )

        dyn.clear(self.id)
        return dyn

    @classmethod
    def get_full_dynamics(cls, prop_name: str) -> Dynamic:
        """Возвращает заданную динамику значений параметров для всех объектов
        данного класса.

        Args:
            prop_name (str): наименование атрибута

        Returns:
            Dynamic | None: инстанс Dynamic, содержищий значения динаики
              параметра для всех объектов класса, или None, если
              для указанного атрибута динамика не задавалась
        """
        props_clazz = cls.model_fields["props"].annotation
        if not (
            inspect.isclass(props_clazz) and issubclass(props_clazz, BaseModel)
        ):
            raise ValueError(f"Класс {cls.__name__} не имеет атрибута props")

        if prop_name not in props_clazz.model_fields.keys():
            raise ValueError(
                f"Модель {cls.__name__} не содержит атрибут {prop_name}"
            )
        try:
            dyn = next(
                meta
                for meta in props_clazz.model_fields[prop_name].metadata
                if isinstance(meta, Dynamic)
            )
        except Exception as err:
            raise ValueError(
                f"Атрибут {prop_name} модели {props_clazz.__name__} "
                f"не содержит аннотации Dynamic: {err}"
            )

        return dyn

    def get_dynamics(
        self, prop_name: str, num_layers: int | None = None, dense: bool = True
    ):
        """Возвращает динамику значений свойства (атрибута) `prop_name`.

        Args:
            prop_name (str): наименование атрибута
            num_layers (int | None): число временных слоев, для которых
              необходимо вернуть значения. None по умолчанию (возвращаются
              все известные значения). Если dense == True, аргумент
              игнорируется и возвращаются все значения
            dense (bool): признак, заполнить ли пробелы в массиве
              (True:  возращается np.ndarray, False: Dict[int, float | int])

        Returns:
            np.ndarray | Dict[int, float | int]: массив динамики значений
            свойства `prop_name`
        """
        assert hasattr(
            self.props, "model_fields"
        ), f"Класс {self.__class__.__name__} не имеет атрибута `model_fields`"
        if prop_name not in self.props.model_fields.keys():  # type: ignore
            raise ValueError(
                f"Модель {self.__class__.__name__} не содержит "
                f"атрибут {prop_name}"
            )

        try:
            dyn = next(
                meta
                for meta in cast(BaseModel, self.props)
                .model_fields[prop_name]
                .metadata
                if isinstance(meta, Dynamic)
            )
        except Exception as err:
            raise ValueError(
                f"Атрибут {prop_name} модели {self.__class__.__name__} "
                f"не содержит аннотации Dynamic {err}"
            )

        return (
            dyn.to_dense_array(self.id, num_layers)
            if dense
            else dyn.get_dynamics(self.id)
        )

    def has_dyn_value(self, prop_name: str, t: int | None = None) -> bool:
        """Проверяет, задавалось ли значение динамики для указанного
        временного слоя.

        Если временной слой не указан, проверяет, задавалось ли какое-либо
        значение для указанного свойства `prop_name`.

        Args:
            prop_name (str): наименование свойства из набора свойств TPROPS
            t (int | None): временной слой или None

        Returns:
            bool: признак, задавалось ли значение для указанного свойства на
            временном слое ``t`
        """
        assert hasattr(
            self.props, "model_fields"
        ), f"Класс {self.__class__.__name__} не имеет атрибута `model_fields`"
        if prop_name not in self.props.model_fields.keys():  # type: ignore
            raise ValueError(
                f"Модель {self.__class__.__name__} не содержит "
                f"атрибут {prop_name}"
            )

        try:
            dyn = next(
                meta
                for meta in cast(BaseModel, self.props)
                .model_fields[prop_name]
                .metadata
                if isinstance(meta, Dynamic)
            )
        except Exception as err:
            raise ValueError(
                f"Атрибут {prop_name} модели {self.__class__.__name__} "
                f"не содержит аннотации Dynamic {err}"
            )

        return dyn.has_value(self.id, t)

    def get_dyn_value(
        self,
        prop_name: str,
        t: int,
        when_not_set: Literal["nearest_left", "interpolate"] = "interpolate",
    ) -> float | int:
        """Возвращает значение из динамики значений параметров объекта.

        Args:
            prop_name (str): наименование свойства из набора свойств TPROPS
            t (int): временной слой
            when_not_set (Literal[&quot;nearest_left&quot;, &quot;interpolate&
            quot;]): возвращаемое значение, когда значение на указанном
            временном слое не задано:
              - `nearest_left`: ближайшее заданное значение на предшествующих
              временных слоях
              - `interpolate`: интерполированное значение

        Returns:
            float | int: значение динамического параметра
        """
        if self.has_dyn_value(prop_name, t):
            dynamics = self.get_dynamics(prop_name, dense=False)
            assert dynamics is not None, "dynamics is None"
            dynamics = cast(dict, dynamics)
            value = dynamics.get((self.id, t))
            assert value is not None, "Внутренняя ошибка"
            return value

        dynamics = self.get_dynamics("state", dense=False)
        assert dynamics is not None, "dynamics is None"
        dynamics = cast(dict, dynamics)
        if when_not_set == "nearest_left":
            # поиск ближайшего заданного состояния
            times = list(dynamics.keys())
            idx = bisect.bisect_left(list(times), t)

            match idx:
                case _ if idx == 0:
                    # Если динамика не задавалась, то возвращается значение
                    # из статических свойств
                    value = getattr(self, prop_name)
                case _ if idx >= len(times):
                    value = dynamics[times[-1]]
                case _:
                    value = dynamics[times[idx - 1]]

            return value

        elif when_not_set == "interpolate":
            dynamics = self.get_dynamics("state", dense=True)
            assert dynamics is not None, "dynamics is None"
            dynamics = cast(dict, dynamics)
            return dynamics[t] if t < len(dynamics) else dynamics[-1]

        raise AttributeError(f"Неизвестный аргумент: {when_not_set}")

    def set_dyn_value(
        self,
        prop_name: str,
        t: int,
        value: float | int,
    ) -> None:
        """Добавляет значение в динамику параметров объекта.

        В случае, если для указанного временного слоя значение
        было задано ранее, перезаписывает его.

        Args:
            prop_name (str): наименование свойства из набора свойств TPROPS
            t (int): временной слой
            value (float | int): значение
        """
        self.add_dynamics(prop_name, {t: value})

    def get_state(self, t: int) -> State:
        """Возвращает состояние объекта (отключен / в работе) для временного
        слоя `t`.

        Args:
            t (int): временной слой

        Returns:
            int: состояние объекта в момент времени t
        """
        assert hasattr(
            self.props, "model_fields"
        ), f"Класс {self.__class__.__name__} не имеет атрибута `model_fields`"
        if "state" not in self.props.model_fields.keys():  # type: ignore
            raise ValueError(
                f"Модель {self.__class__.__name__} не содержит "
                f"атрибут `state` (состояние)"
            )
        dynamics = self.get_dynamics("state", dense=False)
        assert dynamics is not None, "dynamics is None"
        dynamics = cast(dict, dynamics)
        state = dynamics.get(t)

        if state is None:
            # поиск ближайшего заданного состояния
            times = list(dynamics.keys())
            idx = bisect.bisect_left(list(times), t)

            match idx:
                case _ if idx == 0:
                    state = DEFAULT_STATE
                case _ if idx >= len(times):
                    state = dynamics[times[-1]]
                case _:
                    state = dynamics[times[idx - 1]]

        return state

    def set_state(self, t: int, state: int) -> None:
        """Задает состояние объекта (в работе / отключен)
        на временном слое `t`.

        Обертка над функцией set_dyn_value.

        Args:
            t (int): временной слой
            state (int): состояние (0 - отключен, 1 - в работе)
        """
        assert hasattr(
            self.props, "model_fields"
        ), f"Класс {self.__class__.__name__} не имеет атрибута `model_fields`"
        if "state" not in self.props.model_fields.keys():  # type: ignore
            raise ValueError(
                f"Модель {self.__class__.__name__} не содержит "
                f"атрибут `state` (состояние)"
            )
        self.add_dynamics("state", {t: state})

    def is_active(self, t: int) -> bool:
        """Возвращает, является ли объект активным (в работе)
        в момент времени `t`.

        Args:
            t (int): временной слой

        Returns:
            bool: признак, является ли объект активным (в работе)
        """
        return self.get_state(t) == State.ACTIVE


class SimulatedObject(Object[TGeom, TPROPS]):
    """Моделируемый объект."""

    def validate_before_calculation(self):
        """Выполняет проверку объекта перед запуском расчета
        на предмет корректности данных.

        Метод для переопределения в дочерних классах.

        Raises:
            ValueError: при наличии ошибок в данных

        """
        pass


class NodeObject(SimulatedObject[TGeom, TPROPS]):
    """Объект, имеющий один узел подключения.

    Примеры: потребитель, поставщик газа.
    """

    node: int | None = Field(
        None, title="Узел подключения объекта", exclude=True
    )


class EdgeObject(SimulatedObject[TGeom, TPROPS]):
    """Объект, имеющий два узла подключения (входной и выходной).

    Примеры: газопровод, компрессорный цех.
    """

    node_start: int | None = Field(
        None,
        title="Узел, в котором поток входит в объект",
        exclude=True,
    )

    node_end: int | None = Field(
        None,
        title="Узел, в котором поток выходит из объекта",
        exclude=True,
    )
