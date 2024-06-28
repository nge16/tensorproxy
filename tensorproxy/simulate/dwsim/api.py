from typing import Callable, Any, List

from tensorproxy.domain import ObjId, AttrId, AttrUnit
from tensorproxy.simulate import SimulationService

import requests


class DWSIMSimulationService(SimulationService):
    r"""Сервис для работы с API симулятора DWSIM.

    See also:
        https://dwsim.org/
    """

    def __init__(self, url="http://127.0.0.1:8000") -> None:
        super().__init__()
        self.url = url

    def load_flowsheet(self, filepath: str) -> bool:
        url = f"{self.url}/load_flowsheet"

        try:
            response = requests.post(
                url,
                params={
                    "filepath": filepath,
                },
            )

            status = response.json()["status"]
            if status == "success":
                # result_collector = Collector(**response.json()["result"])
                return True

            if status == "failed":
                print(response.json()["cause"])
                return False

        except (requests.RequestException, ValueError) as e:
            print(f"Ошибка на сервере: {e}")
            return False

    def calculate(self) -> List[str] | None:
        url = f"{self.url}/calculate"
        response = requests.post(
            url,
        )

        res = response.json()
        status = res["status"]
        if status == "success":
            return None

        return ["Errors (see in service log...)"]

    def reset_calculations(self) -> None:
        url = f"{self.url}/reset"
        response = requests.post(
            url,
        )

        res = response.json()
        status = res["status"]
        if status == "success":
            return

        print("Не удалось сбросить результаты вычислений")

    def fget_obj_attr(
        self, obj_id: ObjId, attr_id: AttrId, unit: AttrUnit = None, **kwargs
    ) -> Callable[[], Any]:
        url = f"{self.url}/object/attributes/"

        def fget():
            try:
                response = requests.get(
                    url,
                    headers={"Content-type": "application/json"},
                    json={
                        "obj_id": obj_id.sid,
                        "attr_id": attr_id.param.value,
                        "attr_extra": attr_id.extra,
                        "unit": unit.unit if unit is not None else None,
                    },
                )

                res = response.json()
                status = res["status"]
                if status == "success":
                    return float(res["result"])

                if status == "failed":
                    print(response.json()["cause"])
                    return None

            except (requests.RequestException, ValueError) as e:
                print(f"Ошибка на сервере: {e}")
                return None

        return fget

    def fset_obj_attr(
        self,
        obj_id: ObjId,
        attr_id: AttrId,
        unit: AttrUnit = None,
        **kwargs,
    ) -> Callable[[float], str] | None:
        url = f"{self.url}/object/attributes"

        def fset(x: float):
            try:
                payload = {
                    "x": x,
                    "obj_id": obj_id.sid,
                    "attr_id": attr_id.param.value,
                    "attr_extra": attr_id.extra,
                    "unit": unit.unit,
                }
                response = requests.post(
                    url,
                    headers={"Content-type": "application/json"},
                    json=payload,
                )

                res = response.json()
                status = res["status"]

                if status == "failed":
                    print(response.json()["cause"])

            except (requests.RequestException, ValueError) as e:
                print(f"Ошибка на сервере: {e}")

        return fset
