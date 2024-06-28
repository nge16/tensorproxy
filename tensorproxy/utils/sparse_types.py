from typing import Annotated

from pydantic import BeforeValidator, PlainSerializer, WithJsonSchema

import scipy
import numpy as np


def scipy_sparse_before_validator(x):
    if isinstance(x, dict):
        try:
            data: dict = x["data"]
            row: list = x["row"]
            col: list = x["col"]
            shape: tuple = x["shape"]

            x = scipy.sparse.coo_array(
                (data, (row, col)), shape=shape, dtype=np.int32
            )
        except KeyError:
            raise ValueError("Неверно задан формат разреженной матрицы")

    return x


def scipy_sparse_serializer(x: scipy.sparse.coo_array):
    return {
        "data": x.data.tolist(),
        "row": x.row.tolist(),
        "col": x.col.tolist(),
        "shape": x.shape,
    }


TScipySparseCOO = Annotated[
    scipy.sparse.coo_array,
    BeforeValidator(scipy_sparse_before_validator),
    PlainSerializer(scipy_sparse_serializer, return_type=dict),
    WithJsonSchema({"type": "str"}, mode="serialization"),
    # WithJsonSchema({"type": "str"}, mode="validation"),
    WithJsonSchema({"type": "dict"}, mode="validation"),
]
