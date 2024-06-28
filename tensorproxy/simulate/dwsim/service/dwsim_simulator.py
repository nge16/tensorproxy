from typing import List
import os
import clr
import glob


from tensorproxy.simulate.simulator import Simulator
from tensorproxy.simulate.flowsheet import Flowsheet

from .dwsim_flowsheet import DWSIMFlowsheet

DWSIM_DLL_LIST = [
    "CapeOpen.dll",
    "DWSIM.Automation.dll",
    "DWSIM.Interfaces.dll",
    "DWSIM.GlobalSettings.dll",
    "DWSIM.SharedClasses.dll",
    "DWSIM.Thermodynamics.dll",
    "DWSIM.UnitOperations.dll",
    "System.Buffers.dll",
    "DWSIM.Inspector.dll",
    "DWSIM.MathOps.dll",
    "TcpComm.dll",
    "Microsoft.ServiceBus.dll",
    "SkiaSharp.dll",
    "OxyPlot",
    "ThermoCS/ThermoCS.dll",
]


class DWSIMSimulator(Simulator):
    """Класс для взаимодействия с API симулятора DWSIM.

    Args:
        dwsim_path (str): путь до каталога с установленным DWSIM

    """

    def __init__(
        self, dwsim_path: str = "/usr/local/lib/dwsim", verbose: bool = False
    ) -> None:
        super().__init__()

        self.verbose = verbose
        self.dwsim_path = dwsim_path

        self.__dwsim_flowsheet = None
        self.__flowsheet: DWSIMFlowsheet | None = None

        self.flowsheet_filepath = None

        self._fsetters = []
        self.__load_automation()

    @property
    def active_flowsheet(self) -> DWSIMFlowsheet:
        return self.__flowsheet

    def __load_automation(self):
        """Инициализирует симулятор"""

        self.__load_dlls(
            [os.path.join(self.dwsim_path, dll) for dll in DWSIM_DLL_LIST]
        )

        try:
            from DWSIM.Automation import Automation3 as Automation
        except ImportError as e:
            self.logger.error(
                "Не удалось загрузить DWSIM"
                "Пожалуйста, проверьте наличие доступа "
                "к библиотекам DWSIM "
                f"({repr(e)})"
            )
            raise

        self.interf = Automation()
        assert self.interf, "Не удалось создать объект Automation"

    def __load_dlls(self, dlls: List[str] = []) -> None:
        """Добавляет dll DWSIM.

        Если список пустой, добавляются все dll.

        Args:
            dlls (List[str]): список dll, которые требуется открыть
        """
        dlls = dlls or glob.glob(os.path.join(self.dwsim_path, "*.dll"))
        for dll in dlls:
            try:
                clr.AddReference(dll)
                if self.verbose:
                    self.logger.info(f"Загружена dll: {dll}")
            except BaseException as e:
                self.logger.error(
                    f"Не удалось загрузить dll: {dll} ({repr(e)})"
                )

    def load_flowsheet(self, flowsheet_filepath: str) -> Flowsheet | None:
        """Загружает технологическую схему из файла формата `dwxmz`.

        Args:
            flowsheet_filepath (str): путь до файла с описанием
                технологической схемы

        Returns:
            Flowsheet | None: загруженная схема или None
        """
        if self.__flowsheet is None:
            try:
                self.__dwsim_flowsheet = self.interf.LoadFlowsheet(
                    flowsheet_filepath
                )
                self.__flowsheet = DWSIMFlowsheet(self.__dwsim_flowsheet)
            except Exception as e:
                self.logger.error(
                    "При попытке загрузки технологической схемы "
                    "произошла ошибка: "
                    f"{flowsheet_filepath} ({repr(e)})"
                )

        if self.__dwsim_flowsheet is not None:
            self.flowsheet_filepath = flowsheet_filepath
            if self.verbose:
                self.logger.info(
                    f"Загружена технологическая схема: {flowsheet_filepath}"
                )
        else:
            self.flowsheet_filepath = None
            self.__flowsheet = None
            self.logger.warning(
                "Не удалось загрузить технологическую схему: "
                f"{flowsheet_filepath}"
            )

        return self.__flowsheet

    def save_flowsheet(self, flowsheet: Flowsheet, filepath: str) -> None:
        """Сохраняет технологическую схему."""

        if flowsheet and not isinstance(flowsheet, DWSIMFlowsheet):
            raise AttributeError(
                "Для сохранения техсхемы симулятором DWSIM она должна быть "
                "унаследована от DWSIMFlowsheet"
            )

        flowsheet = (
            flowsheet.DWSIM_flowsheet or self.active_flowsheet.DWSIM_flowsheet
        )
        if not flowsheet:
            self.logger.warning(
                "Не удалось сохранить технологическую схему, "
                "flowsheet is None"
            )
        self.interf.SaveFlowsheet2(flowsheet, filepath)
        self.logger.info(f"Технологическая схема сохранена: {filepath}")

    def calculate_flowsheet(
        self, flowsheet: DWSIMFlowsheet | None = None
    ) -> List[str] | None:
        errors = self.interf.CalculateFlowsheet4(
            flowsheet.DWSIM_flowsheet if flowsheet else self.__dwsim_flowsheet
        )
        return list(errors)

    def reset(self):
        if self.active_flowsheet is not None:
            self.active_flowsheet.reset_calculations()

    def shutdown(self):
        """Завершает все процессы моделирования, выполняет очистку."""
        self.interf.ReleaseResources()
        del self.interf
