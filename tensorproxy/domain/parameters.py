from enum import Enum, IntEnum


class Parameter(str, Enum):
    T = "Temperature"
    T_OUT = "OutletTemperature"
    P = "Pressure"
    P_OUT = "OutletPressure"
    MASS_FLOW = "MassFlow"
    MOLAR_FLOW = "MolarFlow"
    COMPOUND_MASS_FLOW = "CompoundMassFlow"
    COMPOUND_MOLAR_FLOW = "CompoundMolarFlow"
    ENERGY_FLOW = "EnergyFlow"
    OUTPUT_VAR = "OutputVariable"


class Phases(IntEnum):
    OVERALL = 0
    LIQUID = 1
    VAPOR = 2
