"""Единица измерения."""

from dataclasses import dataclass
import numpy as np


@dataclass
class Unit:
    """Единица измерения.

    Перевод в СИ: coeff_a_SI * value + coeff_b_SI
    """

    name: str
    """Наименование"""

    coeff_a_SI: float = 1.0
    """Переводной коэффициент в СИ"""

    coeff_b_SI: float = 0.0
    """Переводной коэффициент в СИ"""

    def to_SI(self, value: float | np.ndarray) -> float | np.ndarray:
        """Переводит значение или массив значений в единицы СИ.

        Args:
            value (float | np.ndarray): значение (или массив значений)
              для перевода в СИ

        Returns:
            float | np.ndarray: значение или массив значений в единицах СИ
        """
        return self.coeff_a_SI * value + self.coeff_b_SI

    def from_SI(self, value: float | np.ndarray) -> float | np.ndarray:
        """Переводит из единиц СИ во внутреннюю размерность `name`.

        Args:
            value (float): значение для перевода из СИ

        Returns:
            float: значение во внутреннй размерности `name`
        """
        div = self.coeff_a_SI if self.coeff_a_SI != 0 else 1.0
        return (value - self.coeff_b_SI) / div
