from typing import Callable, Any

from tensorproxy.simulate.flowsheet import Flowsheet
from tensorproxy.domain import ObjId, AttrId, AttrUnit, Parameter


class DWSIMFlowsheet(Flowsheet):
    def __init__(self, dwsim_flowsheet: Any, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__dwsim_flowsheet = dwsim_flowsheet

    @property
    def DWSIM_flowsheet(self):
        return self.__dwsim_flowsheet

    def fget_obj_attr(
        self, obj_id: ObjId, attr_id: AttrId, unit: AttrUnit = None, **kwargs
    ) -> Callable[[], Any]:
        try:
            import DWSIM
            from DWSIM.SharedClasses.SystemsOfUnits import Converter
        except ImportError as e:
            self.logger.error(
                "Не удалось загрузить DWSIM. "
                "Пожалуйста, проверьте наличие доступа "
                "к библиотекам DWSIM "
                f"({repr(e)})"
            )
            raise

        obj = self.DWSIM_flowsheet.GetFlowsheetSimulationObject(obj_id.sid)
        if obj is None:
            raise ValueError(
                f"В технологической схеме отсутствует объект {obj_id.sid}"
            )
        obj = obj.GetAsObject()

        if obj is None:
            raise ValueError(f"Не удалось получить объект {obj_id.sid}")

        fget = None

        def from_si(x, unit: AttrUnit):
            return Converter.ConvertFromSI(unit.unit, x) if unit else x

        if attr_id.param == Parameter.OUTPUT_VAR:
            # OutputVariables возвращаются в исходной размерности,
            # т.к. неизвестно, в какой единице измерения они представлены
            fget = lambda: from_si(  # noqa: E731
                obj.OutputVariables[attr_id.extra], None
            )

        # UnitOperations
        if isinstance(obj, DWSIM.UnitOperations.Streams.EnergyStream):
            match attr_id.param:
                case Parameter.ENERGY_FLOW:
                    fget = lambda: from_si(obj.EnergyFlow, unit)  # noqa: E731
                case _:
                    raise ValueError(f"Неизвестный параметр: {attr_id.param}")

        elif isinstance(obj, DWSIM.Thermodynamics.Streams.MaterialStream):
            match attr_id.param:
                case Parameter.MASS_FLOW:
                    fget = lambda: from_si(
                        obj.GetMassFlow(), unit
                    )  # noqa: E731
                case _:
                    raise ValueError(f"Неизвестный параметр: {attr_id.param}")

        if fget is None:
            raise ValueError(
                f"Неизвестный параметр: {attr_id.param} у объекта типа "
                f"{type(obj)}"
            )

        return fget

    def fset_obj_attr(
        self, obj_id: ObjId, attr_id: AttrId, unit: AttrUnit = None, **kwargs
    ) -> Callable[[float], str] | None:
        try:
            import DWSIM
            from DWSIM.SharedClasses.SystemsOfUnits import Converter
        except ImportError as e:
            self.logger.error(
                "Не удалось загрузить DWSIM"
                "Пожалуйста, проверьте наличие доступа "
                "к библиотекам DWSIM "
                f"({repr(e)})"
            )
            raise

        obj = self.DWSIM_flowsheet.GetFlowsheetSimulationObject(obj_id.sid)
        if obj is None:
            raise ValueError(
                f"В технологической схеме отсутствует объект {obj_id.sid}"
            )
        obj = obj.GetAsObject()

        def _si(x, unit: AttrUnit):
            return Converter.ConvertToSI(unit.unit, x) if unit else x

        fset = None
        # MaterialStream
        if isinstance(obj, DWSIM.Thermodynamics.Streams.MaterialStream):
            match attr_id.param:
                case Parameter.T:
                    fset = lambda x: obj.SetTemperature(_si(x, unit))
                case Parameter.P:
                    fset = lambda x: obj.SetPressure(_si(x, unit))
                case Parameter.MASS_FLOW:
                    fset = lambda x: obj.SetMassFlow(_si(x, unit))
                case Parameter.MOLAR_FLOW:
                    fset = lambda x: obj.SetMolarFlow(_si(x, unit))
                case Parameter.COMPOUND_MASS_FLOW:
                    fset = lambda x: obj.SetOverallCompoundMassFlow(
                        attr_id.extra, _si(x, unit)
                    )
                case Parameter.COMPOUND_MOLAR_FLOW:
                    fset = lambda x: obj.SetOverallCompoundMolarFlow(
                        attr_id.extra, _si(x, unit)
                    )
                case _:
                    raise ValueError(
                        "Присвоение значения не может быть реализовано. "
                        f"Неизвестный параметр: {attr_id.param}"
                    )

        # UnitOperations
        if isinstance(obj, DWSIM.UnitOperations.Streams.EnergyStream):
            match attr_id.param:
                case Parameter.ENERGY_FLOW:
                    fset = lambda x: setattr(obj, "EnergyFlow", _si(x, unit))

                case _:
                    raise ValueError(
                        "Присвоение значения не может быть реализовано. "
                        f"Неизвестный параметр: {attr_id.param}"
                    )

        #  UnitOpBaseClass
        if isinstance(
            obj, DWSIM.UnitOperations.UnitOperations.UnitOpBaseClass
        ):
            match attr_id.param:
                case Parameter.P_OUT:
                    attr_name = (
                        "POut"
                        if isinstance(
                            obj,
                            DWSIM.UnitOperations.UnitOperations.Compressor,  # noqa: E501
                        )
                        else "OutletPressure"
                    )
                    fset = lambda x: setattr(obj, attr_name, _si(x, unit))
                case Parameter.T_OUT:
                    fset = lambda x: setattr(
                        obj, "OutletTemperature", _si(x, unit)
                    )
                case _:
                    raise ValueError(
                        "Присвоение значения не может быть реализовано. "
                        f"Неизвестный параметр: {attr_id.param}"
                    )

        return fset

    def reset_calculations(self) -> None:
        self.DWSIM_flowsheet.ResetCalculationStatus()
