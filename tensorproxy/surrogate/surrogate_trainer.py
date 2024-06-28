from typing import Dict, List, Tuple
import logging
from warnings import warn

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from tensorproxy.simulate import (
    SimulationService,
    SimulationModel,
    SimulationSynthesizer,
)
from .surrogate_model import SurrogateModel


class SurrogateTrainer:
    """Класс для обучения суррогатных моделей."""

    def __init__(
        self,
        surrogate_models: Dict[str, SurrogateModel],
        simulation_service: SimulationService,
    ) -> None:
        self.surrogate_models = surrogate_models
        self.logger = logging.getLogger(__name__)

        mset = set(
            [model.simulation_model for model in surrogate_models.values()]
        )
        if len(mset) != 1:
            raise AttributeError(
                "SurrogateTrainer предназначен для обучения суррогатных "
                "моделей, аппроксимирующих общую полноразмерную модель, "
                "но surrogate_models имеют разные родительские модели"
            )

        self.__simulation_model: SimulationModel = next(iter(mset))
        self.simulation_service = simulation_service

        # словарь для хранения обучающей выборки
        # имя модели -> (x, y)
        self._training_data: Dict[str, Tuple[List, List]] = {
            self.surrogate_models[key].model_name: ([], [])
            for key in self.surrogate_models.keys()
        }
        self._validation_data: Dict[str, Tuple[List, List]] = {
            self.surrogate_models[key].model_name: ([], [])
            for key in self.surrogate_models.keys()
        }

    @property
    def training_data(self) -> Dict[str, Tuple[List, List]]:
        return self._training_data

    def train(  # noqa: C901
        self,
        budget: int = None,
        filepath: str = None,
        test_size: float = None,
        csv_savefile: str | None = None,
        verbose: int = 0,
    ) -> None:
        """Обучает суррогатные модели.

        Args:
            budget (int): бюджет вычислений (число симуляций). Если не задан,
            предусматривается, что обучение будет выполнено на основе данных
            из файла `filepath`.
            filename (str): путь до файла с предварительно сохраненными
            результатами моделирования (сэмплами)
            test_size (float | None): размер тестовой выборки (доля от общего
            размера выборки) или None, если обучение требуется выполнить на
            всей выборке
            csv_savefile (str | None): путь до файла для сохранения результатов
            синтезирования обучающей выборки
        """

        if budget is not None and filepath is not None:
            warn(
                "Одновременно задан бюджет вычислений и путь до файла. "
                "Будет использоваться файл.",
                RuntimeWarning,
            )
            budget = None

        results = {
            self.surrogate_models[name].model_name: model.results
            for name, model in self.surrogate_models.items()
        }

        # синтезирование обучающей выборки
        if budget is not None:
            synthesizer = SimulationSynthesizer(
                budget, self.simulation_service
            )

            if verbose > 0:
                print("Синтезирование обучающей выборки...")

            # запись в файл
            file = (
                open(csv_savefile, mode="a", encoding="utf-8")
                if csv_savefile
                else None
            )
            labels = [p.label for p in self.__simulation_model.domain] + [
                result.name
                for model in self.surrogate_models.values()
                for result in model.results
            ]
            line = ",".join(labels)
            file.write(line)
            file.write("\n")

            # синтезирование выборки
            i = 0
            for x, y in synthesizer.synth(
                simulation_model=self.__simulation_model, results=results
            ):
                print(
                    f"#{i+1} из {budget}: "
                    f"{', '.join([f'{key}: {values}' for key, values in y.items()])}"  # noqa: E501
                )
                for key in y.keys():
                    self.training_data[key][0].append(x)
                    self.training_data[key][1].append(y[key])
                i += 1

                # запись в файл
                values = list(x) + [
                    result.fres()
                    for model in self.surrogate_models.values()
                    for result in model.results
                ]
                line = ",".join([str(lp) for lp in values])
                file.write(line)
                file.write("\n")

            if file:
                file.close()

        # чтение выборки из файла
        if filepath is not None:
            if verbose > 0:
                print(f"Чтение данных из файла {filepath}")
            df = None
            if filepath.endswith(".pkl"):
                df = pd.read_pickle(filepath)
            elif filepath.endswith(".csv"):
                df = pd.read_csv(filepath)

            if df is None:
                raise AttributeError(
                    f"При попытке загрузки файла {filepath} произошла ошибка. "
                    "Файл должен быть в формате pkl или csv."
                )

            for name, model in self.surrogate_models.items():
                features = [param.label for param in model.domain]
                targets = [res.name for res in model.results]
                self.training_data[model.model_name] = (
                    df[features].values,
                    df[targets].values,
                )

        # Обучение моделей
        if verbose > 0:
            print("Обучение суррогатных моделей...")
        for name, model in self.surrogate_models.items():
            model_name = model.model_name
            X = np.array(self.training_data[model_name][0])
            Y = np.array(self.training_data[model_name][1])

            X_train, X_test, Y_train, Y_test = (
                train_test_split(X, Y, test_size=test_size, random_state=41)
                if test_size is not None
                else (X, None, Y, None)
            )

            self.training_data[model_name] = (X_train, Y_train)
            self._validation_data[model_name] = (
                (X_test, Y_test) if test_size is not None else None
            )

            model.fit(
                self.training_data[model_name][0],
                self.training_data[model_name][1],
                validation_data=self._validation_data[model_name],
                verbose=verbose,
            )

    def further_train(self, x: np.ndarray, y: Dict[str, List[float]]):
        """Дообучает суррогатные модели.

        Args:
            x (List[float]): значения управляемых переменных
            y (Dict[str, List[float]]): словарь, содержащий имя суррогатной
            функции и истинные значения в точке `x`

        """
        print("Дообучение суррогатных моделей")

        for name, model in self.surrogate_models.items():
            model_name = model.model_name
            self.training_data[model_name] = (
                np.append(
                    self.training_data[model_name][0], np.atleast_2d(x), axis=0
                ),
                np.append(
                    self.training_data[model_name][1],
                    np.atleast_2d(y[model_name]),
                    axis=0,
                ),
            )

            model.fit(
                self.training_data[model_name][0],
                self.training_data[model_name][1],
                validation_data=self._validation_data[model_name],
            )
