import torch
import argparse
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

from dataset import get_dataloaders
from model import get_resnet18
from utils import get_device

import yaml
import os



with open("../config.yaml", "r") as f:
    config = yaml.safe_load(f)


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=config["batch_size"])
parser.add_argument("--num_epochs", type=int, default=config["num_epochs"])
parser.add_argument("--lr", type=float, default=config["lr"])
args = parser.parse_args()

def train(args):
    device = get_device()
    loaders = get_dataloaders(batch_size=args.batch_size,num_workers=10, root="../data")

    # Model, loss, optimizer
    model = get_resnet18(num_classes=10).to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    total_step = len(loaders["train"])
    model.train()

    for epoch in range(args.num_epochs):
        for i, (images, labels) in enumerate(loaders["train"]):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            output = model(images)
            loss = loss_func(output, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{args.num_epochs}], "
                    f"Step [{i+1}/{total_step}], "
                    f"Loss: {loss.item():.4f}"
                )

    # Save trained model
    torch.save(model.state_dict(), "../experiments/resnet18_mnist.pth")
    print("Model saved at experiments/resnet18_mnist.pth")


    # Save model to SageMaker default output directory
    os.makedirs("/opt/ml/model/", exist_ok=True)
    torch.save(model.state_dict(), "/opt/ml/model/resnet18_mnist.pth")

if __name__ == "__main__":
    train(args)