"""Класс, отвечающий за сборку задачи."""

from typing import Tuple, Literal

import numpy as np

import jax.numpy as jnp
import jax.experimental.sparse as jsparse

from tensorproxy.domain.midstream.transmission.gas_transmission_system import (
    GasTransmissionSystem,
)
from tensorproxy.domain.midstream.transmission.model import (
    In,
    Out,
    Pipe,
)


def get_hydraulics_params(
    model: GasTransmissionSystem,
    t: int,
    unsteady: bool = False,
) -> Tuple[np.ndarray, dict]:
    """Возвращает тензоры атрибутов, необходимых для проведения
    гидравлического расчета ГТС.

    Args:
        t (int): _description_
        unsteady (bool, optional): _description_. Defaults to False.

    Returns:
        np.ndarray, dict: начальное приближение, параметры модели
    """

    _SI = model._to_SI

    # формирование словаря "ID узла" : "ID объекта (потребителя/поставщика)"
    io_inv = {value: key for key, value in model.graph.io}

    def _is_set_p(node: int) -> bool:
        """Проверяет, что в узле задано давление газа.

        Args:
            node (int): номер узла

        Returns:
            bool: задано ли в узле давление
        """
        id = io_inv.get(node)
        obj = model.get_object(id)
        return (
            obj is not None
            and (isinstance(obj, In) or isinstance(obj, Out))
            and obj.props._btype == "P"
        )

    def _build_nodes_tensor(nodelist: list):
        """Формирует массив распределения параметров
        потоков газа в узлах ГТС.

        TODO: реализовать формирование начального приближения

        Args:
            nodelist (list): узлы ГТС
        """

        def get_setted_params(node: int, param: Literal["P", "Q"]):
            id = io_inv.get(node)
            if id is None:
                return 0.0
            obj = model.get_object(id)

            if obj is None or not (
                isinstance(obj, In) or isinstance(obj, Out)
            ):
                return 0.0

            res = 0
            if param == "P" and obj.props._btype == "P":
                res = obj.props.P

            if param == "Q" and obj.props._btype == "Q":
                res = obj.props.Q

            return res or 0.0

        # Массив с параметрами ["P", "T", "Q"]
        nodes_attr = np.array(
            [
                np.array(
                    [
                        (
                            # TODO: реализовать начальное приближение
                            _SI(p, "P")
                            if (p := get_setted_params(node, "P")) > 0
                            else _SI(5.4, "P")
                        ),
                        _SI(30, "T"),
                        _SI(get_setted_params(node, "Q"), "Q"),
                    ]
                )
                for node in nodelist
            ],
            dtype="float64",
        )
        return nodes_attr

    # получение матрицы инциденций для времени `t`
    At = model.get_incidence(t)

    # формирование списка узлов
    n = At.shape[-1]
    m = At.shape[-2]
    nodes = np.arange(m)

    # формирование списка узлов с заданными давлениями
    nodes_p = list(filter(lambda node: _is_set_p(node), nodes))

    # сортировка узлов по признаку наличия заданного давления
    # (вначале узлы, где давление не задано)
    sorted_nodes = sorted(nodes, key=lambda node: node in nodes_p)

    # формирование списка узлов с незаданными давлениями
    k = len(nodes) - len(nodes_p)
    unknown_p_nodes = sorted_nodes[:k]

    # Формирование тензора атрибутов дуг
    # [L, Di, E, Q0, Q1]
    pipes_attr = np.array(
        [
            [
                _SI(pipe.props.L, "L"),
                _SI(pipe.props.Di, "D"),
                pipe.props.E,
                _SI(pipe.props.k, "k"),
                _SI(pipe.props.Te, "T"),
                _SI(0.0, "Q"),
                _SI(0.0, "Q"),
            ]
            for pipe in model._pipes
        ]
    )

    nodes_attr = _build_nodes_tensor(nodes)

    params = {
        "At": jsparse.BCOO.from_scipy_sparse(At),
        "pipes_attr": jnp.array(pipes_attr),
        "nodes_attr": jnp.array(nodes_attr),
        "unknown_p_nodes": jnp.array(unknown_p_nodes, dtype="int32"),
        "known_p_nodes": jnp.array(
            [node for node in range(m) if node not in unknown_p_nodes],
            dtype="int32",
        ),
        "pipes_nodes": jnp.array(
            [
                nodes
                for i, nodes in enumerate(model.to_edges_list(0))
                if isinstance(model._edges[i], Pipe)
            ]
        ),
        "pj_prev": jnp.zeros(k, dtype="float64"),
        "qs_prev": jnp.zeros(n, dtype="float64"),
        "qf_prev": jnp.zeros(n, dtype="float64"),
    }

    if not unsteady:
        x0 = jnp.array(
            [nodes_attr[node][0] for node in unknown_p_nodes]
            + [pipe[3] for pipe in pipes_attr],
            dtype="float64",
        )

    else:
        raise NotImplementedError(
            "Подготовка данных для нестац.моделирования не реализована"
        )
        # g_prev = model.model.snapshots[t - 1]
        # params |= {
        #     # значения давления газа в узлах на предыдущем временном слое
        #     "pj_prev": jnp.array(
        #         [g_prev.nodes[node]["p"] for node in unknown_p_nodes],
        #         dtype="float64",
        #     ),
        #     # значения расхода газа в началах дуг на предыдущем
        #     # временном слое
        #     "qs_prev": jnp.array(
        #         [attr["qs"] for u, v, attr in g_prev.edges(data=True)],
        #         dtype="float64",
        #     ),
        #     # значения расхода газа в конце дуг на предыдущем временном слое
        #     "qf_prev": jnp.array(
        #         [attr["qf"] for u, v, attr in g_prev.edges(data=True)],
        #         dtype="float64",
        #     ),
        # }
        # x0 = jnp.concat(
        #     [
        #         params["pj_prev"].copy(),
        #         params["qs_prev"].copy(),
        #         params["qf_prev"].copy(),
        #     ]
        # )

    return x0, params
