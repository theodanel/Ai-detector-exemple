from pathlib import Path
import csv

import torch
from torch.utils.data import DataLoader

from ai_training.dataset import FoodDataset
from ai_training.model import create_food_model


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "models" / "food_model.pth"
DATASET_DIR = PROJECT_DIR / "dataset"
LABELS_CSV = PROJECT_DIR / "ai_training" / "labels.csv"

BATCH_SIZE = 32
THRESHOLD = 0.35
MAX_ERRORS_TO_SHOW = 80


def load_trained_model(device, num_classes):
    model = create_food_model(num_classes=num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model


def labels_to_names(label_tensor, label_columns):
    names = []

    for index, value in enumerate(label_tensor):
        if int(value.item()) == 1:
            names.append(label_columns[index])

    return names


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device utilisé : {device}")

    test_dataset = FoodDataset(
        images_dir=str(DATASET_DIR),
        labels_csv=str(LABELS_CSV),
        split="test",
        augment=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = load_trained_model(
        device=device,
        num_classes=len(test_dataset.label_columns),
    )

    total_images = 0
    exact_match_count = 0

    true_positive = torch.zeros(len(test_dataset.label_columns))
    false_positive = torch.zeros(len(test_dataset.label_columns))
    false_negative = torch.zeros(len(test_dataset.label_columns))
    true_negative = torch.zeros(len(test_dataset.label_columns))

    detailed_errors = []

    REPORTS_DIR = PROJECT_DIR / "reports"
    REPORTS_DIR.mkdir(exist_ok=True)
    ERRORS_CSV = REPORTS_DIR / "evaluation_errors.csv"

    with torch.no_grad():
        image_global_index = 0

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities >= THRESHOLD).float()

            total_images += labels.size(0)
            exact_match_count += (predictions == labels).all(dim=1).sum().item()

            true_positive += ((predictions == 1) & (labels == 1)).sum(dim=0).cpu()
            false_positive += ((predictions == 1) & (labels == 0)).sum(dim=0).cpu()
            false_negative += ((predictions == 0) & (labels == 1)).sum(dim=0).cpu()
            true_negative += ((predictions == 0) & (labels == 0)).sum(dim=0).cpu()

            for batch_index in range(labels.size(0)):
                expected = labels[batch_index].cpu()
                predicted = predictions[batch_index].cpu()
                probs = probabilities[batch_index].cpu()

                if not torch.equal(expected, predicted):
                    row = test_dataset.dataframe.iloc[image_global_index]
                    image_name = row["image"]

                    expected_names = labels_to_names(
                        expected,
                        test_dataset.label_columns,
                    )

                    predicted_names = labels_to_names(
                        predicted,
                        test_dataset.label_columns,
                    )

                    missed = []
                    extra = []

                    for label_index, label_name in enumerate(test_dataset.label_columns):
                        if expected[label_index] == 1 and predicted[label_index] == 0:
                            missed.append(
                                f"{label_name} ({probs[label_index].item():.3f})"
                            )

                        if expected[label_index] == 0 and predicted[label_index] == 1:
                            extra.append(
                                f"{label_name} ({probs[label_index].item():.3f})"
                            )

                    detailed_errors.append({
                        "image": image_name,
                        "source_folder": row["source_folder"],
                        "source_file": row["source_file"],
                        "expected": expected_names,
                        "predicted": predicted_names,
                        "missed": missed,
                        "extra": extra,
                    })

                image_global_index += 1

    exact_match_accuracy = exact_match_count / total_images

    print()
    print("===== ÉVALUATION GLOBALE =====")
    print(f"Images testées : {total_images}")
    print(f"Exact match accuracy : {exact_match_accuracy:.3f}")

    print()
    print("===== RÉSULTATS PAR ALIMENT =====")

    for index, label_name in enumerate(test_dataset.label_columns):
        tp = true_positive[index].item()
        fp = false_positive[index].item()
        fn = false_negative[index].item()
        tn = true_negative[index].item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + fp + fn + tn)

        print(
            f"{label_name:15s} | "
            f"accuracy: {accuracy:.3f} | "
            f"precision: {precision:.3f} | "
            f"recall: {recall:.3f} | "
            f"TP:{int(tp):3d} FP:{int(fp):3d} FN:{int(fn):3d}"
        )

    print()
    print("===== ERREURS DÉTAILLÉES =====")
    print(f"Nombre total d'images avec erreur : {len(detailed_errors)}")

    for error in detailed_errors[:MAX_ERRORS_TO_SHOW]:
        print()
        print(f"Image : dataset/test/{error['image']}")
        print(f"Attendu : {', '.join(error['expected']) if error['expected'] else 'rien'}")
        print(f"Prédit  : {', '.join(error['predicted']) if error['predicted'] else 'rien'}")

        if error["missed"]:
            print(f"Manqué  : {', '.join(error['missed'])}")

        if error["extra"]:
            print(f"En trop : {', '.join(error['extra'])}")

        with open(ERRORS_CSV, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "image",
                "source_folder",
                "source_file",
                "expected",
                "predicted",
                "missed",
                "extra",
            ]

            writer = csv.DictWriter(
                csvfile,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for error in detailed_errors:
                writer.writerow({
                    "image": error["image"],
                    "source_folder": error["source_folder"],
                    "source_file": error["source_file"],
                    "expected": ";".join(error["expected"]),
                    "predicted": ";".join(error["predicted"]),
                    "missed": ";".join(error["missed"]),
                    "extra": ";".join(error["extra"]),
                })

        print()
        print(f"Rapport d'erreurs sauvegardé ici : {ERRORS_CSV}")

    if len(detailed_errors) > MAX_ERRORS_TO_SHOW:
        print()
        print(
            f"... {len(detailed_errors) - MAX_ERRORS_TO_SHOW} erreurs non affichées."
        )


if __name__ == "__main__":
    main()