from typing import Any, List

import peewee
from fastapi import APIRouter, HTTPException, Response, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import PlainTextResponse,FileResponse
from models.wine import Wine
from pydantic import BaseModel
from pydantic.utils import GetterDict
from utils import utils
router_model = APIRouter(
    prefix="/api/model",
    tags=["model"]
)

templates = Jinja2Templates(directory="templates")


class PeeweeGetterDict(GetterDict):
    def get(self, key: Any, default: Any = None):
        res = getattr(self._obj, key, default)
        if isinstance(res, peewee.ModelSelect):
            return list(res)
        return res


class WineModel(BaseModel):
    id: int
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: int
    total_sulfur_dioxide: int
    density: float
    pH: float
    sulphates: float
    alcohol: float

    class Config:
        orm_mode = True
        getter_dict = PeeweeGetterDict


@router_model.get("/", summary="Get model",
                    description="Return serialized model",response_class=FileResponse)
def get_model():
    return FileResponse("knn_model.sav", media_type='application/octet-stream',filename="model.sav")

@router_model.get("/description", response_class=PlainTextResponse, summary="Get model description",
                    description="Return all model parameters")
def get_wine():
    return utils.model_description()

@router_model.put("/", response_model=WineModel, summary="Put a new wine in db")
async def create(fixed_acidity: float, volatile_acidity: float, citric_acid: float,
                 residual_sugar: float, chlorides: float, free_sulfur_dioxide: int, total_sulfur_dioxide: int,
                 density: float, ph: float, sulphates: float, alcohol: float):
    return await Wine().create_wine(fixed_acidity=fixed_acidity,
                                  volatile_acidity=volatile_acidity,
                                  citric_acid=citric_acid,
                                  residual_sugar=residual_sugar,
                                  chlorides=chlorides,
                                  free_sulfur_dioxide=free_sulfur_dioxide,
                                  total_sulfur_dioxide=total_sulfur_dioxide,
                                  density=density,
                                  ph=ph,
                                  sulphates=sulphates,
                                  alcohol=alcohol)

@router_model.post("/retrain", summary="Retrain model")
def retrain_model():
    return("hello")