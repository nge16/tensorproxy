"""Пакет для описания предментной области. """

from .domain_param import DomainParam
from .interface import ObjId, AttrId, AttrUnit
from .parameters import Parameter

__all__ = ["DomainParam", "ObjId", "AttrId", "AttrUnit", "Parameter"]
