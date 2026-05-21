from backend.nutrition_db import get_food_by_name


def calculate_food_values(food_data: dict, quantity_g: float) -> dict:
    factor = quantity_g / 100

    return {
        "aliment": food_data["aliment"],
        "quantity_g": quantity_g,
        "calories": round(food_data["calories_100g"] * factor, 2),
        "proteines": round(food_data["proteines_100g"] * factor, 2),
        "glucides": round(food_data["glucides_100g"] * factor, 2),
        "lipides": round(food_data["lipides_100g"] * factor, 2),
    }


def calculate_meal(foods: list[dict]) -> dict:
    details = []

    totals = {
        "total_calories": 0,
        "total_proteines": 0,
        "total_glucides": 0,
        "total_lipides": 0,
    }

    for item in foods:
        aliment = item.get("aliment")
        quantity_g = item.get("quantity_g")

        if not aliment:
            raise ValueError("Un aliment est manquant.")

        food_data = get_food_by_name(aliment)

        if food_data is None:
            raise ValueError(f"Aliment introuvable : {aliment}")

        if quantity_g is None:
            quantity_g = food_data["portion_defaut_g"]

        quantity_g = float(quantity_g)

        if quantity_g <= 0:
            raise ValueError(f"Quantité invalide pour : {aliment}")

        calculated = calculate_food_values(food_data, quantity_g)
        details.append(calculated)

        totals["total_calories"] += calculated["calories"]
        totals["total_proteines"] += calculated["proteines"]
        totals["total_glucides"] += calculated["glucides"]
        totals["total_lipides"] += calculated["lipides"]

    return {
        "total_calories": round(totals["total_calories"], 2),
        "total_proteines": round(totals["total_proteines"], 2),
        "total_glucides": round(totals["total_glucides"], 2),
        "total_lipides": round(totals["total_lipides"], 2),
        "details": details,
    }