from typing import (
    List,
    Dict,
    Tuple,
    Annotated,
    Union,
    Literal,
    Any,
    get_args,
    get_origin,
    Type,
    Callable,
    cast,
)
import inspect
from warnings import warn

from collections import defaultdict
import numpy as np
import scipy

from pydantic import (
    ConfigDict,
    BaseModel,
    Field,
    Discriminator,
    Tag,
    model_validator,
    model_serializer,
    SerializerFunctionWrapHandler,
    SerializationInfo,
    ValidationInfo,
)
import scipy.sparse

from tensorproxy.domain.midstream.transmission.header import Header
from tensorproxy.domain.midstream.transmission.model import (
    Object,
    TObject,
    NodeObject,
    EdgeObject,
    Pipe,
    Shop,
    ControlValve,
    In,
    Out,
    GIS,
    Unit,
    Cachable,
)
from tensorproxy.domain.midstream.transmission.model.dynamics import Dynamic


TClazzes = Annotated[
    Union[
        Annotated[In, Tag("In")],
        Annotated[Out, Tag("Out")],
        Annotated[Pipe, Tag("Pipe")],
        Annotated[Shop, Tag("Shop")],
        Annotated[ControlValve, Tag("ControlValve")],
        Annotated[GIS, Tag("GIS")],
    ],
    Discriminator(lambda v: v.get("properties").get("type")),
]


class GraphInfo(BaseModel):
    """Информация о графе расчетной схемы.

    Граф может быть задан явно, в этом случае выполняется его
    запись в json при сериализации. Если граф не был задан явно,
    перед сериализацией выполняется его формирование на основе
    информации об узлах подключения объектов модели.

    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
    )

    incidence: List[List[int]] = Field(
        None,
        title="Описание матрицы связей дуг (ребер) и узлов расчетной схемы",
        description="Первый элемент массива в каждой строке означает "
        "идентификатор узла, а последующие элементы этого массива обозначают "
        "дугу (ребро) графа, при этом если дуга (ребро) выходит из узла, то "
        "указывается знак «минус»",
    )

    io: List[List[int]] = Field(
        None,
        title="Описание связей узлов и объектов «вход/выход», где первый "
        "элемент массива на каждой строке означает идентификатор этого "
        "объекта, а второй элемент этого массива означает идентификатор узла.",
        description="[[ID объекта, ID узла], [], ...]",
    )


class GasTransmissionSystem(BaseModel, Cachable):
    """Модель газотранспортной системы."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
    )

    json_type: str = Field(
        "FeatureCollection",
        alias="type",
    )

    json_header: Header = Field(
        default_factory=lambda: Header(),  # type: ignore
        alias="header",
        title="Общая информация о расчете",
    )

    graph: GraphInfo = Field(
        default_factory=lambda: GraphInfo(),  # type: ignore
        title="Описание графа расчетной схемы",
    )

    objects: List[TClazzes] = Field(
        [],
        title="Объекты газотранспортной системы",
        description="Объекты газотранспортной системы",
        alias="features",
    )

    _obj_hash: Dict[int, TClazzes] = {}
    """Словарь id объекта -> Object"""

    _obj_clazzes: Dict[TClazzes, List[TClazzes]] = defaultdict(list)
    """Словарь <class> - <список объектов класса>"""

    _edges: List[EdgeObject] = []
    """Дуги графа (объекты за исключением поставщиков и потребителей)."""

    _ins: List[In] = []
    """Поставщики (притоки) газа."""

    _outs: List[Out] = []
    """Потребители (стоки) газа."""

    _pipes: List[Pipe] = []
    """Газопроводы."""

    _shops: List[Shop] = []
    """Компрессорные цеха."""

    _ctrl_valves: List[ControlValve] = []
    """Краны-регуляторы."""

    _gis: List[GIS] = []
    """Газоизмерительные стации."""

    _nodes: List[int] | None = None
    """Узлы графа."""

    _nodes_objects: Dict[int, List[TClazzes]] = defaultdict(list)
    """Словарь <Id узла> - <список объектов, инцидентных узлу>"""

    @property
    def inlets(self) -> List[In]:
        """Возвращает список постащиков."""
        return self._ins

    @property
    def oulets(self) -> List[Out]:
        """Возвращает список потребителей / стоков."""
        return self._outs

    @property
    def pipes(self) -> List[Pipe]:
        """Возвращает список газопроводов."""
        return self._pipes

    @property
    def shops(self) -> List[Shop]:
        """Возвращает список цехов."""
        return self._shops

    @property
    def control_valves(self) -> List[ControlValve]:
        """Возвращает список кранов-регуляторов."""
        return self._ctrl_valves

    @property
    def gis(self) -> List[GIS]:
        """Возвращает список ГИС."""
        return self._gis

    @property
    def edges(self) -> List[EdgeObject]:
        """Возвращает дуги графа.

        Returns:
            List[TClazzes]: список дуг графа
        """
        return self._edges

    @property
    def n(self) -> int:
        """Возвращает число объектов в модели, включая
        объекты-дуги и поставщики  / потребители

        Returns:
            int: число объектов
        """
        return len(self.objects)

    @property
    def n_edges(self) -> int:
        """Возвращает число дуг в графе.

        Returns:
            int: число дуг
        """
        return len(self._edges)

    def invalidate_cache(self):
        """Обновляет все кэшированные значения."""
        self._obj_clazzes = defaultdict(list)
        self._nodes = None
        self._nodes_objects = defaultdict(list)

    @model_validator(mode="after")  # type: ignore
    def post_load(  # noqa: C901
        self, info: SerializationInfo
    ) -> "GasTransmissionSystem":
        self._obj_hash |= {obj.id: obj for obj in self.objects}

        # присвоение дугам и объектам поставщик/потребитель информации
        # об узлах подключения
        if self.graph.incidence is not None:
            self.infer_connections()

        for obj in self.objects:
            if isinstance(obj, EdgeObject) and (
                obj.node_start is None or obj.node_end is None
            ):
                warn(
                    f"В схеме найден объект {obj.id}, "
                    "который отсутствует в матрице инцидентности"
                )
                continue

            if isinstance(obj, In):
                self._ins.append(obj)
            elif isinstance(obj, Out):
                self._outs.append(obj)
            elif isinstance(obj, Pipe):
                self._pipes.append(obj)
            elif isinstance(obj, Shop):
                self._shops.append(obj)
            elif isinstance(obj, ControlValve):
                self._ctrl_valves.append(obj)
            elif isinstance(obj, GIS):
                self._gis.append(obj)

            if not isinstance(obj, In) and not isinstance(obj, Out):
                self._edges.append(obj)

        return self

    @model_validator(mode="wrap")  # type: ignore
    def parse_model(  # noqa: C901
        data: Dict,  # type: ignore
        handler: Callable,
        info: ValidationInfo,
    ) -> "GasTransmissionSystem":
        """Загружает модель, создает инстанс `GasTransmissionSystem`.

        Выполняет кастомную загрузку динамики значений параметров объектов.

        Args:
            handler (callable): базовый обработчик данных

        Returns:
            GasTransmissionSystem: инстанс GasTransmissionSystem
        """
        dynamics: dict[str, Any] = data.pop("dynamics", {})

        # 0: data -> raw input (dict)
        validated_self = handler(data)
        # 1: validated_self - инстанс GasTransmissionSystem

        # 2: парсинг динамики параметров
        # словарь <class_tag>: <class>, например, <"In">: <tensorpoxy...In>
        types = {}
        for arg in get_args(TClazzes):
            if get_origin(arg) == Union:
                union = get_args(arg)
                for u in union:
                    type, hint = get_args(u)
                    if isinstance(hint, Tag):
                        types[hint.tag] = type

        for clazz_name, clazz in types.items():
            props_clazz: BaseModel = clazz.model_fields["props"].annotation
            if not (
                inspect.isclass(props_clazz)
                and issubclass(props_clazz, BaseModel)
            ):
                continue

            for _, field_info in props_clazz.model_fields.items():
                try:
                    dyn_prop = next(
                        meta
                        for meta in field_info.metadata
                        if isinstance(meta, Dynamic)
                    )
                    dyn_prop.clear()

                    # if info.mode != "json":
                    # после очистки динамики свойств,
                    # кэшированных с предыдущей модели
                    # continue

                    section = dynamics.get(dyn_prop.class_name)
                    if section is None:
                        # warn(
                        #     "В json не найдена секция с данными динамики "
                        #     f"объектов {dyn_prop.class_name}"
                        # )
                        continue

                    dyn_values: dict | None = section.get(dyn_prop.prop_name)
                    if dyn_values is None:
                        continue

                    dyn_values = {
                        tuple(map(int, key.split(","))): value
                        for key, value in dyn_values.items()
                    }
                    dyn_prop.from_dict(dyn_values)

                except StopIteration:
                    pass
                except Exception as err:
                    print(err)
                    pass

        return validated_self

    @model_serializer(mode="wrap", when_used="json")
    def serialize_model(
        self,
        serializer: SerializerFunctionWrapHandler,
        info: SerializationInfo,
    ) -> Dict[str, Any]:
        """Выполняет сериализацию модели."""

        if self.graph.incidence is None:
            self.graph.incidence = self.infer_incidence("lil")

        if self.graph.io is None:
            self.graph.io = self.infer_incidence("io")

        partial_result: Dict[str, Any] = serializer(self)

        # Сериализация динамики параметров
        props_clazzes = list(
            set([object.props.__class__ for object in self.objects])
        )

        dynamics = {}
        for clazz in props_clazzes:
            class_dict = {}
            class_label = None
            for _, field_info in clazz.model_fields.items():
                try:
                    dyn = next(
                        meta
                        for meta in field_info.metadata
                        if isinstance(meta, Dynamic)
                    )
                    class_dict |= dyn.to_dict()
                    class_label = dyn.class_name
                    assert (
                        dyn.class_name == class_label
                        if class_label is not None
                        else True
                    ), f"У {clazz.__name__} динамические атрибуты "
                    "имеют разные метки класса"
                except Exception:
                    pass

            if len(class_dict) > 0:
                dynamics[class_label] = class_dict

        partial_result["dynamics"] = dynamics
        return partial_result

    def validate_before_simulation(self):
        """Выполняет проверку корректности расчетной схемы для проведения
        имитационного моделирования гидравлических режимов.

        Raises:
            ValueError: в случае некорректности схемы
        """
        # валидация объектов
        for o in self.objects:
            o.validate_before_calculation()

        nodes = self.get_nodes()
        node_objects: List = [self.get_node_objects(node) for node in nodes]

        # проверка, что в узлах нет одновременно двух или более
        # поставщиков / потребителей с заданным давлением
        ps = np.array(
            [
                len(
                    list(
                        filter(
                            lambda o: isinstance(o, (In, Out))
                            and o.get_boundary_type() == "P",
                            olist,
                        )
                    )
                )
                > 1
                for olist in node_objects
            ]
        )
        errs = np.where(ps)[0]
        if errs.size > 0:
            raise ValueError(
                f"В узлах {errs} найдены несколько поставщиков / потребителей "
                "с заданным давлением газа. Пожалуйста, замените несколько "
                "поставщиков / потребителей на один."
            )

        # проверка, что в узлах нет одновременно двух или более
        # поставщиков / потребителей с заданным расходом
        ps = np.array(
            [
                len(
                    list(
                        filter(
                            lambda o: isinstance(o, (In, Out))
                            and o.get_boundary_type() == "Q",
                            olist,
                        )
                    )
                )
                > 1
                for olist in node_objects
            ]
        )
        errs = np.where(ps)[0]
        if errs.size > 0:
            raise ValueError(
                f"В узлах {errs} найдены несколько поставщиков / потребителей "
                "с заданным расходом газа. Пожалуйста, замените несколько "
                "поставщиков / потребителей на один агрегированный объект."
            )

    def add_object(self, obj: TClazzes) -> None:
        """Добавляет объект в модель.

        Args:
            obj (TClazzes): объект расчетной схемы
        """
        # assert isinstance(
        #     obj, TClazzes
        # ), f"Не удалось добавить объект класса {obj.__class__.__name__}, "
        # "так как он отсутствует в списке разрешенных типов объектов"

        if obj.id == 0:
            # т.к. в матрице связей дуга с ID=0 не может быть записана,
            # как исходящая из узла (со знаком минус)
            raise AttributeError(
                "Невозможно добавить объект с зарезервированным "
                "идентификатором 0"
            )

        if obj.id in [o.id for o in self.objects]:
            raise ValueError(
                "Невозможно добавить объект, т.к. "
                f"id={obj.id} уже используется"
            )

        self.objects.append(obj)
        self._obj_hash[obj.id] = obj

        if isinstance(obj, In):
            self._ins.append(obj)
        elif isinstance(obj, Out):
            self._outs.append(obj)
        else:
            self._edges.append(obj)
            if isinstance(obj, Pipe):
                self._pipes.append(obj)
            elif isinstance(obj, Shop):
                self._shops.append(obj)
            elif isinstance(obj, ControlValve):
                self._ctrl_valves.append(obj)
            elif isinstance(obj, GIS):
                self._gis.append(obj)

    def add_objects(self, olist: List[TClazzes]) -> None:
        """Добавляет объекты в модель."""
        if any([o.id == 0 for o in olist]):
            # т.к. в матрице связей дуга с ID=0 не может быть записана,
            # как исходящая из узла (со знаком минус)
            raise AttributeError(
                "Невозможно добавить объект с зарезервированным "
                "идентификатором 0"
            )

        exob = [o.id for o in self.objects]
        aod = [o.id for o in olist if o.id in exob]
        if len(aod) > 0:
            raise ValueError(
                "Невозможно добавить объекты, т.к. "
                f"id={aod} уже используются"
            )
        self.objects.extend(olist)
        self._edges += [
            edge
            for edge in olist
            if not isinstance(edge, In) and not isinstance(edge, Out)
        ]
        self._ins += [obj for obj in olist if isinstance(obj, In)]
        self._outs += [obj for obj in olist if isinstance(obj, Out)]
        self._pipes += [obj for obj in olist if isinstance(obj, Pipe)]
        self._shops += [obj for obj in olist if isinstance(obj, Shop)]
        self._ctrl_valves += [
            obj for obj in olist if isinstance(obj, ControlValve)
        ]
        self._gis += [obj for obj in olist if isinstance(obj, GIS)]
        self._obj_hash |= {obj.id: obj for obj in olist}

    def build_dynamics(  # noqa: C901
        self,
        prop_name: str,
        clazz: Type[TClazzes] | None = None,
        olist: List[TClazzes] | None = None,
        to_SI: bool = True,
        dtype: type = np.float64,
        transform_hook: (
            Callable[
                [np.ndarray, np.ndarray, np.ndarray],
                Tuple[np.ndarray, np.ndarray, np.ndarray],
            ]
            | None
        ) = None,
        shape: Tuple | None = None,
        use_cache: bool = True,
    ) -> scipy.sparse.coo_array:
        """Возвращает динамику заданных значений параметров для объектов
        указанного класса.

        В случае, если для временного слоя `t`=0 значения не заданы, задает
        их из статических атрибутов модели.

        Args:
            prop_name (str): наименование атрибута
            clazz (Type[TClazzes]): класс объектов
            olist (List[TClazzes]): список объектов, для которых собрать
              информацию; при указании атрибут clazz игнорируется
            to_SI (bool): выполнить ли перевод значений в СИ.
              True по умолчанию.
            dtype (type): тип значений в возвращаемой разреженной матрице
            transform_hook (Callable): хук для трансформации times, ids, values
              (данных для формирования разреженной матрицы) в новые
              times, ids, values, например, для замены отдельных значений в
              итоговой матрице
            T (int): минимальное количество временных слоев или None для
              автоматического определения размерности возвращаемой матрицы по T
            use_cache (bool, optional): использовать ли кэшированный словарь
              <класс>:<список объектов класса>. По умолчанию True.

        Returns:
            scipy.sparse.coo_array: разреженная матрица в формате
            <id временного слоя>, <id объекта>, <значение>
        """
        if clazz is None and olist is None:
            raise AttributeError(
                "Необходимо указать либо `clazz`, либо `olist`"
            )

        converter: Unit | None = None
        if olist is None:
            assert clazz is not None, "'clazz' is None"
            ocls: dict = defaultdict(list)
            if use_cache and len(self._obj_clazzes) > 0:
                ocls = self._obj_clazzes
            else:
                for o in self.objects:
                    ocls[o.__class__].append(o)
                self._obj_clazzes = ocls

            olist = ocls.get(clazz)
            if olist is None:
                raise AttributeError(
                    f"В схеме отсутствуют расчетные объекты класса {clazz}"
                )

            dyn = clazz.get_full_dynamics(prop_name)
            dynamics = dyn.dynamics.copy() if dyn.dynamics is not None else {}
            if to_SI:
                converter = clazz.get_units_converter(prop_name)
        else:
            clazzes = [o.__class__ for o in olist]
            clazzes = list(set(clazzes))
            dynamics = {}
            for cls in clazzes:
                dynamics |= (
                    d
                    if (d := cls.get_full_dynamics(prop_name).dynamics)
                    is not None
                    else {}
                )
                if converter is None and to_SI:
                    converter = cls.get_units_converter(prop_name)

        # если на нулевом слое значение атрибута не задавалось,
        # получить его из статических свойств
        for o in olist:
            dynamics.setdefault(
                (o.id, 0), o.get_prop_value(prop_name, to_SI=False)
            )

        keys = np.array(list(dynamics.keys()), dtype=np.int32)
        keys = np.atleast_2d(keys)
        values = np.array(list(dynamics.values()), dtype=dtype)
        values = np.asarray(
            (
                converter.to_SI(values)
                if to_SI and converter is not None
                else values
            )
        )

        # идентификаторы объектов и номера временных слоев
        ids, times = (
            (keys[:, 0], keys[:, 1])
            if keys.size > 0
            else (np.array([], dtype=np.int32), np.array([], dtype=np.int32))
        )

        if transform_hook is not None:
            # примеры ожидаемых применений:
            # - трансформация ids объектов в ids вершин графа
            # - трансформация идентификаторов дуг в порядковые идентификаторы
            times, ids, values = transform_hook(times, ids, values)

        if shape is None and len(ids) == 0:
            shape = (0, 0)

        # assert shape is not None, "Static checker helper"

        # shape может быть задан как (None, 100) или (100, None)
        if shape is not None:
            if shape[0] is None and shape[1] is not None:
                shape = (np.max(times) + 1 if len(times) > 0 else 0, shape[1])
            if shape[1] is None and shape[0] is not None:
                shape = (shape[0], np.max(ids) + 1 if len(ids) > 0 else 0)

        return scipy.sparse.coo_array(
            (values, (times, ids)), shape=shape, dtype=dtype
        )

    def dyn_shape(
        self,
        prop_name: str,
        clazz: Type[TClazzes] | None = None,
        olist: List[TClazzes] | None = None,
        transform_hook: (
            Callable[
                [np.ndarray, np.ndarray, np.ndarray],
                Tuple[np.ndarray, np.ndarray, np.ndarray],
            ]
            | None
        ) = None,
        use_cache: bool = True,
    ) -> Tuple[int, int]:
        """Возвращает размерность динамики параметров
        [максимальное число временных слоев, максимальный id объекта].

        Args:
            prop_name (str): наименование атрибута
            clazz (Type[Object]): класс объектов
            olist (List[Object]): список объектов, для которых собрать
              информацию; при указании атрибут clazz игнорируется
              use_cache (bool, optional): использовать ли кэшированный словарь
                <класс>:<список объектов класса>. По умолчанию True.
            transform_hook (Callable): хук для трансформации times, ids, values
              (данных для формирования разреженной матрицы) в новые
              times, ids, values, например, для замены отдельных значений в
              итоговой матрице
            use_cache (bool, optional): использовать ли кэшированный словарь
              <класс>:<список объектов класса>. По умолчанию True.

        Returns:
            Tuple[int, int]: размерность динамики параметров
        """
        dynamics = self.build_dynamics(
            prop_name=prop_name,
            clazz=clazz,
            olist=olist,
            transform_hook=transform_hook,
            use_cache=use_cache,
        )

        return dynamics.shape

    def get_object(self, id: int) -> TClazzes | None:
        """Возвращает объект по идентификатору.

        Args:
            id (int): идентификатор объекта

        Returns:
            Object | None: инстанс объекта или None
        """
        return self._obj_hash.get(id)

    def get_edges_state(self, t: int) -> np.ndarray:
        """Возвращает массив состояний дуг для заданного временного слоя.

        Args:
            t (int): временной слой

        Note:
            Определяет последние заданные состояния для каждой дуги до
            указанного момента времени. Если для какой-то дуги состояние
            не задавалось, то принимается равным 1 (в работе).

        Returns:
            np.array: массив состояний дуг
        """
        return np.array(
            [edge.get_state(t) for edge in self._edges], dtype=np.int32
        )

    def get_nodes(self, use_cache: bool = True) -> List[int]:
        """Возвращает узлы расчетого графа.

        Args:
            use_cache (bool): получить ли из кэша

        Returns:
            List[int]: список узловы
        """
        if use_cache and self._nodes is not None:
            return self._nodes

        nodes = [
            (
                [obj.node]
                if isinstance(obj, NodeObject)
                else (
                    [obj.node_start, obj.node_end]
                    if (
                        isinstance(obj, EdgeObject)
                        and obj.node_start is not None
                        and obj.node_end is not None
                    )
                    else []
                )
            )
            for obj in self.objects
        ]

        nodes: List = [node for ni in nodes for node in ni]
        nodes = list(set(nodes))
        nodes.sort()
        self._nodes = nodes.copy()

        return nodes

    def get_nodes_coords(self) -> dict[int, Any]:
        """Возвращает координаты узлов.

        Returns:
            dict[int, Any]: словарь с координатами
        """
        pos = {}
        for obj in self.objects:
            if obj.geometry is None:
                continue
            if (
                (
                    obj.geometry.coord.geometry_type == "LineString"
                    or obj.geometry.coord.geometry_type == "MultiPoint"
                )
                and isinstance(obj, EdgeObject)
                and (obj.node_start is not None and obj.node_end is not None)
            ):
                pos[obj.node_start] = obj.geometry.coord.coordinates[0]
                pos[obj.node_end] = obj.geometry.coord.coordinates[-1]

            if (
                obj.geometry.coord.geometry_type == "Point"
                and isinstance(obj, NodeObject)
                and obj.node is not None
            ):
                pos[obj.node] = obj.geometry.coord.coordinates

        return pos

    def normalize_nodes(self, nodes: List[int]) -> dict[int, int]:
        """Выполняет нормализацию узлов расчетного графа.

        "Нормализованные" узлы - узлы с нумерацией 0 ... m,
        где m - число узлов в графе.

        Args:
            nodes (List[int]): узлы с произвольной нумерацией

        Returns:
            List[int]: список узлов с нормализованной нумерацией
        """
        return {model_node: i for i, model_node in enumerate(nodes)}

    def get_node_objects(
        self,
        node: int,
        clazzes: Type[TClazzes] | Tuple[Type[TClazzes]] | None = None,
        use_cache: bool = True,
    ) -> List[Object]:
        """Возвращает список объектов, инцидентных узлу,
        включая дуги и объекты типа поставщик / потребитель.

        Args:
            node (int): номер узла
            clazzes (List[TClazzes]): список классов, по которым
            отфильтровать выборку
            use_cache (bool): использовать ли кэшированный список

        Returns:
            List[Object]: список объектов, инцидентных узлу
        """
        if use_cache and len(self._nodes_objects) > 0:
            olist = self._nodes_objects.get(node)
        else:
            self._nodes_objects.clear()
            for o in self.objects:
                if isinstance(o, NodeObject) and o.node is not None:
                    self._nodes_objects[o.node].append(o)
                elif isinstance(o, EdgeObject) and (
                    o.node_start is not None and o.node_end is not None
                ):
                    self._nodes_objects[o.node_start].append(o)
                    self._nodes_objects[o.node_end].append(o)
            olist = self._nodes_objects.get(node)

        return (
            []
            if olist is None
            else [
                o for o in olist if clazzes is None or isinstance(o, clazzes)
            ]
        )

    def get_incidence(self, t: int | None = None) -> scipy.sparse.csr_array:
        """Возвращает матрицу инцидентности.

        Если указан временной слой `t`, формирует матрицу
        инцидентности с учетом состояний объектов (в работе / отключен)
        на этом временном слое. Если `t` == None, возвращает полную
        матрицу инцидентности.

        Args:
            t (int | None): номер временного слоя или None

        Returns:
            scipy.sparse.csr_array: разреженная матрица инцидентности
        """
        # TODO: рассмотреть целесообразность кэширования матрицы
        # для увеличения производительности обработки
        inc = self.infer_incidence("scipy")
        inc = cast(scipy.sparse.coo_array, inc)
        inc = inc.tocsr()
        if t is not None:
            states = self.get_edges_state(t)
            # т.к. принятое соглашение, что 0 - в работе,
            # необходимо инвертировать матрицу
            diag = scipy.sparse.identity(
                states.shape[-1], dtype=np.int32  # type: ignore
            )
            diag -= scipy.sparse.diags(states)
            inc = inc @ diag
        return inc

    def infer_incidence(
        self, format: Literal["scipy", "lil", "io"] = "scipy"
    ) -> scipy.sparse.coo_array | List[List[int]]:
        """Формирует матрицы связей из информации об узлах подключения
        объектов модели.

        Args:
            format (Literal[&quot;scipy&quot;, &quot;lil&quot;], optional):
                формат матрицы:
                - "scipy": возвращает матрицу инцидентности ребер (дуг)
                    и узлов графа в формате scipy.sparse.coo_array
                - "lil": возвращает матрицу связей ребер (дуг)
                    и узлов графа в формате List[List[int]], где первый
                    элемент массива в каждой строке означает идентификатор
                    узла, а последующие элементы этого массива обозначают
                    дугу (ребро) графа, при этом если дуга (ребро) выходит
                    из узла, то указывается знак «минус»
                - io: возвращает матрицу связей объектов типа "вход/выход"
                    (типа поставщик, потребитель) и узлов графа в формате
                    [[ID объекта, ID узла], [], ...]",
                По умолчанию "scipy".

        Returns:
            scipy.sparse.coo_array | List[List[int]]: разреженная матрица
                инцидентности, или при format="io" - матрица связей объектов
                типа "вход/выход" (типа поставщик, потребитель) и узлов графа
        """

        # узлы в нумерации, заданной извне
        nodes = np.array(self.get_nodes(), dtype=np.int32)

        # формирование матрицы инцидентности
        n = len(self._edges)
        m = len(nodes)

        if format == "scipy":
            row = np.array(
                [
                    node
                    for edge in self._edges
                    for node in (
                        np.where(nodes == edge.node_start)[0].item(),
                        np.where(nodes == edge.node_end)[0].item(),
                    )
                ],
                dtype=np.int32,
            )
            col = np.array(
                [
                    e
                    for edge, _ in enumerate(self._edges)
                    for e in (edge, edge)
                ],
                dtype=np.int32,
            )
            data = np.tile(np.array([1, -1], np.int32), len(self._edges))

            inc = scipy.sparse.coo_array(
                (data, (row, col)), shape=(m, n), dtype=np.int32
            )

            return inc

        if format == "lil":
            mat = defaultdict(list)
            for edge in self._edges:
                mat[np.where(nodes == edge.node_start)[0].item()].append(
                    -edge.id
                )
                mat[np.where(nodes == edge.node_end)[0].item()].append(edge.id)
            mat = [
                [id for id in [node, *edges]] for node, edges in mat.items()
            ]
            return mat

        if format == "io":
            io = [[o.id, o.node] for o in self._ins + self._outs]
            return io

        raise AttributeError(f"Неизвестный параметр функции: {format=}")

    def infer_connections(self) -> None:  # noqa: C901
        """Определяет узлы подключения каждого объекта модели из информации
        о графе соединений."""
        if self.graph.incidence is None:
            raise ValueError("В модели не задана матрица связей")
        if self.graph.io is None:
            raise ValueError(
                "В модели не задана матрица связей узлов и объектов вход/выход"
            )

        # определение узлов присоединения объектов поставщик / потребитель
        for io in self.graph.io:
            if len(io) != 2:
                raise ValueError(
                    "Матрица связей узлов и объектов вход/выход содержит "
                    f"некорректную запись: {io}"
                )
            id = io[0]
            o = self.get_object(id)
            if not isinstance(o, (In, Out)):
                raise ValueError(
                    "В матрице связей узлов и объектов вход/выход указан "
                    f"{id=}, но объект с этим ID ({o}) не является объектом "
                    "типа поставщик / потребитель"
                )
            o.node = io[1]

        # определение узлов присоединения дуг
        for node_edges in self.graph.incidence:
            if len(node_edges) < 2:
                raise ValueError(
                    "Матрица связей узлов и дуг содержит "
                    f"некорректную запись: {node_edges}"
                )
            node = node_edges[0]
            for e in node_edges[1:]:
                edge = self.get_object(abs(e))
                if not isinstance(edge, EdgeObject):
                    raise ValueError(
                        "В матрице связей узлов и дуг указан идентификатор "
                        f"дуги {e}, но объект с этим ID ({edge}) не является "
                        "объектом модели"
                    )
                if e < 0:
                    edge.node_start = node
                else:
                    edge.node_end = node

    def get_edges_tensor(
        self,
        etype: TObject,
        props: List[str],
        to_SI: bool = True,
    ) -> np.ndarray:
        """Возвращает тензор значений свойств объектов указанного типа в
        единицах СИ.

        Args:
            etype (Literal[TObject]): тип объектов
            props (List[str]): свойства
            to_SI (bool): выполнить ли перевод в единицы СИ

        Example::
            >>> model.get_edges_tensor("Pipe", ["L", "Di", "E", "k", "Te"])

        Returns:
            np.ndarray: тензор значений свойств в единицах СИ
        """

        olists: Dict[TObject, List] = {
            "Pipe": self._pipes,
            "Shop": self._shops,
            "In": self._ins,
            "Out": self._outs,
        }
        olist = olists.get(etype)
        if olist is None:
            raise AttributeError(f"Неизвестный тип объектов: {etype}")

        # clazz = self._classes.get(etype)
        # if clazz is None:
        #     raise AttributeError(f"Неизвестный тип объектов: {etype}")
        # olist = self._class_lists.get(clazz)
        # if olist is None:
        #     raise AttributeError(
        #         f"Класс {clazz.__name__} не имеет списка объектов"
        #     )

        return np.array(
            [
                [obj.get_prop_value(prop, to_SI=to_SI) for prop in props]
                for obj in olist
            ]
        )

    def to_edges_list(
        self, t: int | None = None, inactive_label: int | str | None = None
    ) -> List[tuple]:
        """Возвращает список кортежей, отвечающих дугам расчетного графа,
        где каждый кортеж (tuple) содержит узел начала и узел конца дуги.

        Если задан временной слой `t`, учитывает состояние матрицы инциденций
        на этом слое. В случае если дуга в момент времени `t` неактивна,
        заполняет кортеж метками, указанными в `inactive_label`.

        Args:
            t (int | None): временной слой
            inactive_label (int | str | None, optional): метка для неактивной
            дуги. None по умолчанию.

        Returns:
            List[tuple]: начальные и конечные узлы дуг
        """
        del t  # не используется
        return [(edge.node_start, edge.node_end) for edge in self.edges]

    def to_networkx(self, t: int):
        """Преобразует модель в оринетированный граф `networkx`.

        Args:
            t (int): временной слой

        Returns:
            Graph: граф networkx
        """

        import networkx as nx

        return nx.from_edgelist(
            self.to_edges_list(t, "из."), create_using=nx.MultiDiGraph
        )

    def draw(
        self,
        t: int,
        draw_type: Literal["from_pos", "planar", "common"] = "common",
        options: dict = {
            "font_size": 10,
            "node_size": 1,
            "shops_size": 80,
            "edgecolors": "black",
            "width": 2,
            "node_color": "white",
            "edge_color": "blue",
            "arrowsize": 6,
        },
        with_labels=True,
        figsize=(16, 12),
    ) -> None:
        """Выполняет отрисовку графа для временного слоя `t`.

        Args:
            t (int): временной, который требуется отрисовать
            options (dict, optional): настройки отрисовки
            with_labels (bool, optional): выводить ли метки

        See also:
          https://networkx.org/documentation/stable/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html#networkx.drawing.nx_pylab.draw_networkx
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import networkx as nx

        G = self.to_networkx(t)
        labeldict = {i: i for i in G.nodes}

        mpl.rcParams["figure.figsize"] = figsize
        match draw_type:
            case "from_pos":
                pos = self.get_nodes_coords()
            case "planar":
                pos = nx.planar_layout(G)
            case _:
                pos = nx.shell_layout(G)

        pipes = [(pipe.node_start, pipe.node_end) for pipe in self._pipes]
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=pipes,
            edge_color="b",
            width=1,
            arrows=False,
        )
        nx.draw_networkx_nodes(
            G,
            pos=(
                {
                    shop.node_start: [
                        (
                            shop.geometry.coord.coordinates[0][0]
                            + shop.geometry.coord.coordinates[-1][0]
                        )
                        / 2,
                        (
                            shop.geometry.coord.coordinates[0][1]
                            + shop.geometry.coord.coordinates[-1][1]
                        )
                        / 2,
                    ]
                    for shop in self._shops
                    if shop.geometry is not None
                }
                if draw_type == "from_pos"
                else pos
            ),
            nodelist=[shop.node_start for shop in self._shops],
            node_size=options.get("shops_size", 100),
            node_color="w",
            edgecolors="#1f78b4",
            label={shop.node_start: shop.id for shop in self._shops},
        )

        if with_labels:
            nx.draw_networkx_labels(
                G,
                pos,
                labels=labeldict,
                font_size=options.get("font_size", 12),
            )

        plt.show()
