from pathlib import Path
from backend.prediction_filter import filter_predictions

import torch
from PIL import Image
from torchvision import transforms

from ai_training.model import create_food_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "food_model.pth"

LABELS = [
    "riz",
    "pates",
    "poulet",
    "steak",
    "oeuf",
    "salade",
    "tomate",
    "brocoli",
    "pain",
    "fromage",
    "frites",
    "pomme_de_terre",
    "saumon",
    "banane",
    "yaourt"
]

IMAGE_SIZE = 224


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = create_food_model(num_classes=len(LABELS))

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)

model.to(device)

model.eval()


transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


def predict_foods(image_path: str) -> list:

    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)

        probabilities = torch.sigmoid(outputs)[0]

    results = []

    for label, probability in zip(LABELS, probabilities):

        confidence = float(probability)

        if confidence > 0.3:
            results.append({
                "aliment": label,
                "confidence": round(confidence, 3)
            })

    filtered_result = filter_predictions(results)

    return filtered_result