from typing import Any, List

import peewee
from fastapi import APIRouter, HTTPException, Response, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from models.wine import Wine
from pydantic import BaseModel
from pydantic.utils import GetterDict
from utils import utils

router_predict = APIRouter(
    prefix="/api/predict",
    tags=["predict"]
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


@router_predict.get("/", summary="Get the perfect wine",
                    description="Return the exepected values to have the perfect wine")
def get_wine():
    return utils.determine_perfect_wine()


"""@router_predict.get("/view/{id}", response_model=WineModel, summary="Returns a single user")
async def view(uid: int):
    """"""
        To view all details related to a single user

        - **id**: The integer id of the user you want to view details.
    """"""
    user = Wine.get_wine(uid=uid)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    return user
"""


@router_predict.post("/", summary="Test a wine quality", description="Generate a grade "
                                                                                               "regarding the model "
                                                                                               "values")
async def create(fixed_acidity: float, volatile_acidity: float, citric_acid: float,
                 residual_sugar: float, chlorides: float, free_sulfur_dioxide: int, total_sulfur_dioxide: int,
                 density: float, ph: float, sulphates: float, alcohol: float):
    new_wine = [fixed_acidity,
                volatile_acidity,
                citric_acid,
                residual_sugar,
                chlorides,
                free_sulfur_dioxide,
                total_sulfur_dioxide,
                density,
                ph,
                sulphates,
                alcohol]
    return utils.predict_quality(utils.load_model(), new_wine)
