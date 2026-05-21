from pathlib import Path

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

from torchvision import transforms


class FoodDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        labels_csv: str,
        split: str | None = None,
        image_size: int = 224,
        augment: bool = False,
    ):
        self.images_dir = Path(images_dir)
        self.dataframe = pd.read_csv(labels_csv)

        if split is not None:
            if "split" not in self.dataframe.columns:
                raise ValueError("Le fichier labels.csv ne contient pas de colonne 'split'.")

            self.dataframe = self.dataframe[
                self.dataframe["split"] == split
            ].reset_index(drop=True)

        if self.dataframe.empty:
            raise ValueError(f"Aucune image trouvée pour le split : {split}")

        self.label_columns = [
            column
            for column in self.dataframe.columns
            if column not in ["image", "split", "source_folder", "source_file"]
        ]

        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((image_size + 20, image_size + 20)),
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.85, 1.0),
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=12),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.15,
                    hue=0.03,
                ),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image_name = row["image"]

        if "split" in self.dataframe.columns:
            image_path = self.images_dir / row["split"] / image_name
        else:
            image_path = self.images_dir / image_name

        if not image_path.exists():
            raise FileNotFoundError(f"Image introuvable : {image_path}")

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)

        labels = row[self.label_columns].values.astype("float32")
        labels_tensor = torch.tensor(labels)

        return image_tensor, labels_tensor