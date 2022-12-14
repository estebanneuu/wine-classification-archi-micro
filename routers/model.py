from typing import Any, List
from fastapi import APIRouter, HTTPException, Response, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import PlainTextResponse,FileResponse
from utils import utils
router_model = APIRouter(
    prefix="/api/model",
    tags=["model"]
)

templates = Jinja2Templates(directory="templates")

@router_model.get("/", summary="Get model",
                    description="Return serialized model",response_class=FileResponse)
def get_model():
    return FileResponse("knn_model.sav", media_type='application/octet-stream',filename="model.sav")

@router_model.get("/description", response_class=PlainTextResponse, summary="Get model description",
                    description="Return all model parameters")
def get_wine():
    return utils.model_description()

@router_model.put("/", summary="Put a new wine in db")
async def create(fixed_acidity: float, volatile_acidity: float, citric_acid: float,
                 residual_sugar: float, chlorides: float, free_sulfur_dioxide: float, total_sulfur_dioxide: float,
                 density: float, ph: float, sulphates: float, alcohol: float, quality: float):
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
                    alcohol,
                    quality]
        return utils.data_enrichment(new_wine, "Wines.csv")

@router_model.post("/retrain", summary="Retrain model")
def retrain_model():
    return utils.retrain_model("Wines.csv")