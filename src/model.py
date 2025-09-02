import torch.nn as nn
from torchvision import models

def get_resnet18(num_classes: int = 10):
    """
    Returns a ResNet18 model modified for MNIST (1 channel input).
    """
    model_resnet18 = models.resnet18(pretrained=False)

    # Adjust first conv layer to accept 1-channel MNIST images
    model_resnet18.conv1 = nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )

    # Adjust final layer to number of classes
    model_resnet18.fc = nn.Linear(model_resnet18.fc.in_features, num_classes)

    return model_resnet18