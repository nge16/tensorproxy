"""Модуль с основными расчетными функциями для моделирования стационарных
и нестационарных режимов работы газотранспортной системы."""

from .pipelines_networks import solve_steady_state

__all__ = ["solve_steady_state"]
