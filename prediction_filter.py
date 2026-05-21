from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
NUTRITION_FILE = BASE_DIR / "data" / "nutrition.csv"


def load_nutrition_data():
    if not NUTRITION_FILE.exists():
        raise FileNotFoundError(f"Fichier introuvable : {NUTRITION_FILE}")

    df = pd.read_csv(NUTRITION_FILE)
    return df


def get_all_foods():
    df = load_nutrition_data()
    return df.to_dict(orient="records")


def get_food_by_name(food_name: str):
    df = load_nutrition_data()

    food_name = food_name.lower().strip()
    result = df[df["aliment"].str.lower() == food_name]

    if result.empty:
        return None

    return result.iloc[0].to_dict()