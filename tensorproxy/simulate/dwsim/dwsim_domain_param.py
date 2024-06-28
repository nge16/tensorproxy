from tensorproxy import Parameter
from tensorproxy.domain import DomainParam, ObjId, AttrId, AttrUnit


def get_dwsim_domain_param(
    label: str,
    obj_id: str,
    attr_id: Parameter,
    extra: str = None,
    unit: str = None,
    lower: float = None,
    upper: float = None,
    extra_params: dict = {},
) -> DomainParam:
    """Обертка для удобной работы с параметрами симулятора DWSIM.

    Args:
        label (str): Имя параметра для нужд отладки
        obj_id (str): Идентификатор объекта
        attr_id (Parameter): Идентификатор атрибута объекта
        extra (str, optional): Дополнительные сведения об атрибуте
        unit (str, optional): Единица измерения
        lower (float, optional): Нижняя граница значения атрибута
        upper (float, optional): Верхняя граница значения атрибута
        extra_params (dict, optional): Дополнительная информация

    Returns:
        _type_: DesignParam
    """
    return DomainParam(
        label=label,
        obj_id=ObjId(sid=obj_id),
        attr_id=AttrId(param=attr_id, extra=extra),
        unit=AttrUnit(unit=unit),
        lower=lower,
        upper=upper,
        extra_params=extra_params,
    )
