import torch.nn as nn
from torchvision import models


def create_food_model(num_classes: int) -> nn.Module:
    model = models.mobilenet_v3_small(weights="DEFAULT")

    in_features = model.classifier[3].in_features

    model.classifier[3] = nn.Linear(in_features, num_classes)

    return model