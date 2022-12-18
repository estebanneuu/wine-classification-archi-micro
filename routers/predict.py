from typing import Any, List
from fastapi import APIRouter, HTTPException, Response, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from utils import utils

router_predict = APIRouter(
    prefix="/api/predict",
    tags=["predict"]
)

templates = Jinja2Templates(directory="templates")

@router_predict.get("/", summary="Get the perfect wine",
                    description="Return the exepected values to have the perfect wine")
def get_wine():
    return utils.determine_perfect_wine()

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
