from tensorproxy.utils.log import LOG
from .stack import inspect_stack

import warnings

from .sparse_types import TScipySparseCOO
from .sparse_utils import csr_vappend


def check_unknown_options(unknown_options: dict):
    if unknown_options:
        msg = ", ".join(map(str, unknown_options.keys()))
        warnings.warn(f"Неизвестная опция: {msg}")


__all__ = [
    "LOG",
    "check_unknown_options",
    "inspect_stack",
    # "TSparseDOK",
    "TScipySparseCOO",
    # sparse utils
    "csr_vappend",
]
