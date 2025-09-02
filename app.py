import streamlit as st
import boto3
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Load secrets from Streamlit (set in cloud)
AWS_REGION = st.secrets["AWS_REGION"]
ENDPOINT_NAME = st.secrets["SAGEMAKER_ENDPOINT"]

# SageMaker runtime client
runtime = boto3.client(
    "sagemaker-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
)

st.title("MNIST Digit Classifier (SageMaker + Streamlit 🚀)")

# Load MNIST test dataset
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

# User picks an index
index = st.slider("Pick a test image index:", 0, len(test_dataset)-1, 0)

# Get image & label
image, label = test_dataset[index]
img_np = image.squeeze().numpy()

# Show image
st.write(f"Ground truth label: {label}")
st.image(img_np, width=150, caption=f"MNIST index {index}", clamp=True)

# Preprocess for SageMaker
payload = image.numpy().reshape(1, 1, 28, 28).tolist()  # batch size 1

if st.button("Predict with SageMaker"):
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=str(payload)  # SageMaker expects JSON
    )
    result = response["Body"].read().decode("utf-8")
    st.success(f"Predicted: {result}")