from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ai_training.dataset import FoodDataset
from ai_training.model import create_food_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "dataset"
LABELS_CSV = PROJECT_ROOT / "ai_training" / "labels.csv"
MODELS_DIR = PROJECT_ROOT / "models"

LAST_MODEL_PATH = MODELS_DIR / "food_model.pth"
BEST_MODEL_PATH = MODELS_DIR / "food_model_best.pth"

BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.0003
PATIENCE = 5


def run_epoch(model, dataloader, criterion, device, optimizer=None):
    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0

    with torch.set_grad_enabled(is_training):
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

    return total_loss / len(dataloader)


def train():
    MODELS_DIR.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device utilisé : {device}")

    train_dataset = FoodDataset(
        images_dir=str(DATASET_DIR),
        labels_csv=str(LABELS_CSV),
        split="train",
        augment=True,
    )

    val_dataset = FoodDataset(
        images_dir=str(DATASET_DIR),
        labels_csv=str(LABELS_CSV),
        split="val",
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = create_food_model(num_classes=len(train_dataset.label_columns))
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    print()
    print("===== DÉBUT ENTRAÎNEMENT =====")
    print(f"Images train : {len(train_dataset)}")
    print(f"Images val   : {len(val_dataset)}")
    print(f"Batch size   : {BATCH_SIZE}")
    print(f"Epochs max   : {EPOCHS}")
    print()

    for epoch in range(EPOCHS):
        train_loss = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )

        val_loss = run_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1}/{EPOCHS} - "
            f"Train loss : {train_loss:.4f} - "
            f"Val loss : {val_loss:.4f} - "
            f"LR : {current_lr:.7f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            torch.save(model.state_dict(), BEST_MODEL_PATH)

            print(f"  Nouveau meilleur modèle sauvegardé : {BEST_MODEL_PATH}")
        else:
            epochs_without_improvement += 1

            print(
                f"  Pas d'amélioration validation "
                f"({epochs_without_improvement}/{PATIENCE})"
            )

        if epochs_without_improvement >= PATIENCE:
            print()
            print("Early stopping : la validation ne progresse plus.")
            break

    torch.save(model.state_dict(), LAST_MODEL_PATH)

    print()
    print(f"Dernier modèle sauvegardé ici : {LAST_MODEL_PATH}")
    print(f"Meilleur modèle sauvegardé ici : {BEST_MODEL_PATH}")
    print(f"Meilleure val loss : {best_val_loss:.4f}")


if __name__ == "__main__":
    train()