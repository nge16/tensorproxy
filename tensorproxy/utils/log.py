"""Модуль логирования.
"""
from typing import List, Dict
import functools
import inspect
import logging
from logging import Logger
import os
import sys
from logging.handlers import RotatingFileHandler
from os.path import join


class LOG:
    """Кастомный логгер.

    Имя логгера автоматически определяется при вызове.

    """

    base_path = "stdout"
    fmt = "%(asctime)s.%(msecs)03d - " "%(name)s - %(levelname)s - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt)
    max_bytes = 50000000
    backup_count = 3
    name = os.getenv("SIM_DEFAULT_LOG_NAME") or "SIM"
    level = os.getenv("SIM_DEFAULT_LOG_LEVEL") or "INFO"
    diagnostic_mode = False
    _loggers: Dict[str, Logger] = {}

    @classmethod
    def __init__(cls, name: str = "PYHYDROSIM"):
        cls.name = name

    @classmethod
    def init(cls, config: dict = None):
        cls.max_bytes = config.get("max_bytes", 50000000)
        cls.backup_count = config.get("backup_count", 3)
        cls.level = config.get("level") or LOG.level

    @classmethod
    def create_logger(cls, name: str, tostdout: bool = True) -> Logger:
        if name in cls._loggers:
            return cls._loggers[name]
        logger = logging.getLogger(name)
        logger.propagate = False
        # логирование также в stdout
        if tostdout or cls.base_path == "stdout":
            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_handler.setFormatter(cls.formatter)
            logger.addHandler(stdout_handler)
        # логирование в файл
        if cls.base_path != "stdout":
            os.makedirs(cls.base_path, exist_ok=True)
            path = join(cls.base_path, cls.name.lower().strip() + ".log")
            handler = RotatingFileHandler(
                path, maxBytes=cls.max_bytes, backupCount=cls.backup_count
            )
            handler.setFormatter(cls.formatter)
            logger.addHandler(handler)
        logger.setLevel(cls.level)
        cls._loggers[name] = logger
        return logger

    @classmethod
    def set_level(cls, level):
        cls.level = level
        for l in cls._loggers:  # noqa: E741
            cls._loggers[l].setLevel(level)

    @classmethod
    def _get_real_logger(cls):
        name = ""
        if cls.name is not None:
            name = cls.name + " - "

        # Stack:
        # [0] - _log()
        # [1] - debug(), info(), warning(), or error()
        # [2] - caller
        stack = inspect.stack()

        # Record:
        # [0] - frame object
        # [1] - filename
        # [2] - line number
        # [3] - function
        # ...
        record = stack[2]
        mod = inspect.getmodule(record[0])
        module_name = mod.__name__ if mod else ""
        name += module_name + ":" + record[3] + ":" + str(record[2])

        logger = cls.create_logger(name, tostdout=True)
        return logger

    @classmethod
    def info(cls, *args, **kwargs):
        cls._get_real_logger().info(*args, **kwargs)

    @classmethod
    def debug(cls, *args, **kwargs):
        cls._get_real_logger().debug(*args, **kwargs)

    @classmethod
    def warning(cls, *args, **kwargs):
        cls._get_real_logger().warning(*args, **kwargs)

    @classmethod
    def error(cls, *args, **kwargs):
        cls._get_real_logger().error(*args, **kwargs)

    @classmethod
    def exception(cls, *args, **kwargs):
        cls._get_real_logger().exception(*args, **kwargs)


def log_deprecation(
    log_message: str = "DEPRECATED",
    deprecation_version: str = "Unknown",
    func_name: str = None,
    func_module: str = None,
    excluded_package_refs: List[str] = None,
):
    """
    Логирует информацию об использовании устаревшей функции.

    Args:

        log_message (str): сообщение
        deprecation_version (str): версия пакета, в котором метод будет удален
        func_name: имя декорируемой функции (или читается из стека)
        func_module: модуль с декорируемой функцией (или читается из стека)
        @param excluded_package_refs: список пакетов для исключения
    """
    import inspect

    stack = inspect.stack()[1:]  # [0] is this method
    call_info = "Unknown Origin"
    origin_module = func_module
    log_name = (
        f"{LOG.name} - {func_module}:{func_name}"
        if func_module and func_name
        else LOG.name
    )
    for call in stack:
        module = inspect.getmodule(call.frame)
        name = module.__name__ if module else call.filename
        if any(
            (
                name if name.startswith(x) else None
                for x in ("ovos_utils.log", "<")
            )
        ):
            continue
        if not origin_module:
            origin_module = name
            log_name = f"{LOG.name} - {name}:{func_name or call[3]}:{call[2]}"
            continue
        if excluded_package_refs and any(
            (name.startswith(x) for x in excluded_package_refs)
        ):
            continue
        if not name.startswith(origin_module):
            call_info = f"{name}:{call.lineno}"
            break
    LOG.create_logger(log_name).warning(
        f"Deprecation version={deprecation_version}. Caller={call_info}. "
        f"{log_message}"
    )


def deprecated(log_message: str, deprecation_version: str = "Unknown"):
    """Декоратор для логирования сообщения о вызове устаревшей функции."""

    def wrapped(func):
        @functools.wraps(func)
        def log_wrapper(*args, **kwargs):
            log_deprecation(
                log_message=log_message,
                func_name=func.__qualname__,
                func_module=func.__module__,
                deprecation_version=deprecation_version,
            )
            return func(*args, **kwargs)

        return log_wrapper

    return wrapped
