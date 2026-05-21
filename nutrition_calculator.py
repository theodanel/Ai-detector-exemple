from backend.recommendation_engine import generate_recommendation
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from backend.ai_demo import simulate_food_detection
from fastapi.middleware.cors import CORSMiddleware

from backend.nutrition_db import get_all_foods, get_food_by_name
from backend.nutrition_calculator import calculate_meal
from pathlib import Path
from fastapi import UploadFile, File
from backend.ai_inference import predict_foods


app = FastAPI()
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FoodQuantity(BaseModel):
    aliment: str
    quantity_g: float | None = None


class MealRequest(BaseModel):
    foods: list[FoodQuantity]

class AnalyzeMealRequest(BaseModel):
    foods: list[FoodQuantity]
    objective: str = "sante"    


@app.get("/")
def root():
    return {"message": "Nutrition AI API fonctionne"}


@app.get("/api/nutrition")
def api_get_all_foods():
    return get_all_foods()


@app.get("/api/nutrition/{food_name}")
def api_get_food(food_name: str):
    food = get_food_by_name(food_name)

    if food is None:
        raise HTTPException(status_code=404, detail="Aliment introuvable")

    return food


@app.post("/api/calculate-meal")
def api_calculate_meal(meal: MealRequest):
    try:
        foods_as_dict = [food.model_dump() for food in meal.foods]
        return calculate_meal(foods_as_dict)

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

@app.post("/api/analyze-meal")
def api_analyze_meal(meal: AnalyzeMealRequest):
    try:
        foods_as_dict = [food.model_dump() for food in meal.foods]

        nutrition_result = calculate_meal(foods_as_dict)

        recommendation_result = generate_recommendation(
            total_calories=nutrition_result["total_calories"],
            total_proteines=nutrition_result["total_proteines"],
            total_glucides=nutrition_result["total_glucides"],
            total_lipides=nutrition_result["total_lipides"],
            objective=meal.objective
        )

        return {
            "nutrition": nutrition_result,
            "recommendation": recommendation_result
        }

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))        

@app.post("/api/analyze-image-demo")
def api_analyze_image_demo():

    detection_result = simulate_food_detection()

    foods_for_calculation = []

    for food in detection_result["detected_foods"]:
        foods_for_calculation.append({
            "aliment": food["aliment"],
            "quantity_g": food["estimated_quantity_g"]
        })

    nutrition_result = calculate_meal(foods_for_calculation)

    recommendation_result = generate_recommendation(
        total_calories=nutrition_result["total_calories"],
        total_proteines=nutrition_result["total_proteines"],
        total_glucides=nutrition_result["total_glucides"],
        total_lipides=nutrition_result["total_lipides"],
        objective="sante"
    )

    return {
        "detection": detection_result,
        "nutrition": nutrition_result,
        "recommendation": recommendation_result
    }        

@app.post("/api/predict-image")
async def api_predict_image(file: UploadFile = File(...)):
    upload_path = UPLOADS_DIR / file.filename

    with open(upload_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    prediction_result = predict_foods(str(upload_path))

    return {
        "filename": file.filename,
        "analysis": prediction_result
    }

@app.post("/api/full-analyze-image")
async def api_full_analyze_image(file: UploadFile = File(...)):
    upload_path = UPLOADS_DIR / file.filename

    with open(upload_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    prediction_result = predict_foods(str(upload_path))
    predictions = prediction_result["predictions"]

    foods_for_calculation = []

    for prediction in predictions:
        foods_for_calculation.append({
            "aliment": prediction["aliment"],
            "quantity_g": 100
        })

    nutrition_result = calculate_meal(foods_for_calculation)

    recommendation_result = generate_recommendation(
        total_calories=nutrition_result["total_calories"],
        total_proteines=nutrition_result["total_proteines"],
        total_glucides=nutrition_result["total_glucides"],
        total_lipides=nutrition_result["total_lipides"],
        objective="sante"
    )

    return {
        "filename": file.filename,
        "analysis": prediction_result,
        "predictions": predictions,
        "estimated_portions": foods_for_calculation,
        "nutrition": nutrition_result,
        "recommendation": recommendation_result
    }   