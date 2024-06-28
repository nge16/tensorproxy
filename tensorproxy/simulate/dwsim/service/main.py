from starlette.responses import JSONResponse

from fastapi import FastAPI, Body

from tensorproxy.simulate.dwsim.service.dwsim_simulator import DWSIMSimulator
from tensorproxy.simulate.flowsheet import Flowsheet
from tensorproxy.domain import ObjId, AttrId, AttrUnit

app = FastAPI()
simulator = DWSIMSimulator()
loaded_flowsheet: Flowsheet = None


@app.get("/")
def read_root() -> JSONResponse:
    return JSONResponse({"simulations": "DWSIM"})


@app.post("/calculate")
def calculate() -> JSONResponse:
    global simulator
    errors = simulator.calculate_flowsheet()
    if errors:
        print(errors)

    return JSONResponse(
        {
            "status": "success" if len(errors) == 0 else "failed",
            # "errors": list(errors),
        }
    )


@app.post("/reset")
def reset() -> JSONResponse:
    global simulator
    simulator.reset()
    return JSONResponse({"status": "success"})


@app.post("/load_flowsheet/")
def load_flowsheet(filepath: str) -> JSONResponse:
    global simulator
    global loaded_flowsheet
    loaded_flowsheet = simulator.load_flowsheet(filepath)
    return JSONResponse(
        {
            "status": "success",
            "loaded": loaded_flowsheet is not None,
        }
    )


@app.get("/object/attributes/")
def get_obj_attr(
    obj_id: str = Body(..., embed=True),
    attr_id: str = Body(..., embed=True),
    attr_extra: str = Body(default=None, embed=True),
    unit: str = Body(default=None, embed=True),
) -> JSONResponse:
    global loaded_flowsheet
    value = loaded_flowsheet.fget_obj_attr(
        obj_id=ObjId(sid=obj_id),
        attr_id=AttrId(param=attr_id, extra=attr_extra),
        unit=AttrUnit(unit=unit),
    )()

    return JSONResponse(
        {
            "status": "success",
            "result": value,
        }
    )


@app.post("/object/attributes")
def set_obj_attr(
    x: float = Body(..., embed=True),
    obj_id: str = Body(..., embed=True),
    attr_id: str = Body(..., embed=True),
    attr_extra: str = Body(default=None, embed=True),
    unit: str = Body(default=None, embed=True),
) -> JSONResponse:
    global loaded_flowsheet
    loaded_flowsheet.fset_obj_attr(
        obj_id=ObjId(sid=obj_id),
        attr_id=AttrId(param=attr_id, extra=attr_extra),
        unit=AttrUnit(unit=unit),
    )(x)

    return JSONResponse(
        {
            "status": "success",
        }
    )
