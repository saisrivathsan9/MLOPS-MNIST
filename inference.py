import torch
import torch.nn as nn
from model import get_resnet18
import json


# Initialize model 
def model_fn(model_dir):
    model = get_resnet18(num_classes=10)
    model.load_state_dict(torch.load(f"{model_dir}/resnet18_mnist.pth", map_location="cpu"))
    model.eval()
    return model

# Handle input request (JSON -> tensor)
def input_fn(request_body, content_type):
    if content_type == "application/json":
        data = json.loads(request_body)
        tensor = torch.tensor(data["input"], dtype=torch.float32)
        return tensor
    raise ValueError(f"Unsupported content type: {content_type}")

# Make prediction
def predict_fn(input_object, model):
    with torch.no_grad():
        outputs = model(input_object)
        _, pred = torch.max(outputs, 1)
        return pred.item()

# Format response
def output_fn(prediction, content_type):
    if content_type == "application/json":
        return json.dumps({"prediction": prediction})
    raise ValueError(f"Unsupported content type: {content_type}")
