from typing import Any, Tuple, Dict, Callable

import numpy as np

from smt.surrogate_models import KRG
from smt.utils.misc import compute_rms_error

from tensorproxy.surrogate import SurrogateModel


class KrigingSurrogate(SurrogateModel):
    """Суррогатная модель на основе кригинга (обертка SMT)."""

    def __init__(self, *args, **kwargs) -> None:
        print_global = kwargs.pop("print_global", True)

        super().__init__(*args, **kwargs)
        self._model = KRG(print_prediction=False, print_global=print_global)
        self._model.options["corr"] = "abs_exp"

    def fit(
        self,
        x: Any,
        y: Any,
        validation_data: Tuple | None = None,
        verbose: int = 0,
    ) -> None:
        self._model.set_training_values(x, y)
        self._model.train()

        if validation_data:
            val_x, val_y = validation_data
            val_logs = self.evaluate(
                x=val_x, y=val_y, metrics={"RMS_error": compute_rms_error}
            )
            val_logs = {"val_" + name: val for name, val in val_logs.items()}

            if verbose > 0:
                print(val_logs)

    def predict(self, x: np.ndarray, args=()) -> np.ndarray:
        x = np.atleast_2d(x)
        return self._model.predict_values(x).squeeze()

    def predict_gradient(self, x: np.ndarray, args=()) -> np.ndarray:
        x = np.atleast_2d(x)
        return np.array(
            [
                self._model.predict_derivatives(x, i).squeeze()
                for i in range(self._model.nx)
            ]
        )

    def evaluate(
        self, x: Any, y: Any, metrics: Dict[str, Callable]
    ) -> Dict[str, float]:
        val_logs = {}
        for name, fmetric in metrics.items():
            val_logs[name] = fmetric(self._model, x, y)
        return val_logs

    def load(self, path: str) -> bool:
        raise NotImplementedError("Метод подлежит реализации")

    def save(self, path: str) -> bool:
        raise NotImplementedError("Метод подлежит реализации")
