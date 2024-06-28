"""Классы для указания информации о геометрических координатах
расчетных объектов."""

from typing import (
    Any,
    Literal,
    Generic,
    TypeVar,
    List,
    ClassVar,
)

from pydantic import (
    BaseModel,
    Field,
    model_serializer,
    model_validator,
    ValidationInfo,
)


TGeometry = Literal[
    "Point",
    "LineString",
    "MultiPoint",
]


class Point(BaseModel):
    """Точечный объект, в котором свойство coordinates должно содержать одну
    пару координат, означающую логический (семантический) центр, несущий
    различный смысл в зависимости от типа объекта графической схемы.
    """

    geometry_type: ClassVar[TGeometry] = "Point"
    coordinates: List[float]


class LineString(BaseModel):
    """Линия, в которой свойство «coordinates» должно содержать массив из двух
    и более пар координат, при этом аналогично мультиточечному объекту порядок
    координат задает ориентацию соответствующего ребра (дуги) расчетного графа.
    """

    geometry_type: ClassVar[TGeometry] = "LineString"
    coordinates: List[List[float]]

    @model_validator(mode="after")  # type: ignore
    def validate_input_params(self) -> "LineString":
        if not isinstance((coord := self.coordinates), list) or len(coord) < 2:
            raise ValueError(
                "Объект типа LineString должен содержать массив (список) "
                f"из двух и более пар координат (cм. {self.coordinates})"
            )
        return self


class MultiPoint(BaseModel):
    """Мультиточечный объект, в котором свойство coordinates должно содержать
    массив пар координат, при этом первая координата соответствует «началу»
    расчетного объекта (месту входа газового потока), а последняя – «концу»
    объекта (месту выхода газового потока), тем самым задавая ориентацию
    соответствующего ребра (дуги) расчетного графа.
    """

    geometry_type: ClassVar[TGeometry] = "MultiPoint"
    coordinates: List[List[float]]

    @model_validator(mode="after")  # type: ignore
    def validate_input_params(self):
        if not isinstance((coord := self.coordinates), list) or len(coord) < 2:
            raise ValueError(
                "Объект типа MultiPoint должен содержать массив (список) "
                f"из двух и более пар координат (cм. {self.coordinates})"
            )
        return self


TypesGeometry = Point | LineString | MultiPoint

T = TypeVar("T")


class Geometry(BaseModel, Generic[T]):
    coord: T = Field(
        default_factory=lambda: T(),  # type: ignore
        alias="coordinates",
        description="Геометрические координаты",
    )

    @model_serializer(when_used="json")
    def serialize_model(self):
        return {
            "type": self.coord.geometry_type,  # type: ignore
            "coordinates": self.coord.coordinates,  # type: ignore
        }

    @model_validator(mode="before")
    @classmethod
    def before_transform(cls, input: Any, inf: ValidationInfo):
        if inf.mode == "json":
            return {
                "coordinates": {
                    "geometry_type": input["type"],
                    "coordinates": input["coordinates"],
                }
            }
        return input
