import streamlit as st
import boto3
import json
import numpy as np

# --- Streamlit Secrets ---
AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
REGION = st.secrets["AWS_REGION"]
ENDPOINT_NAME = st.secrets["SAGEMAKER_ENDPOINT"]

# --- SageMaker clients ---
sm_client = boto3.client(
    "sagemaker",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=REGION
)

runtime_client = boto3.client(
    "sagemaker-runtime",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=REGION
)

# --- Helper Functions ---
def deploy_endpoint():
    """Check if endpoint exists. Deploy model if not."""
    existing = [ep["EndpointName"] for ep in sm_client.list_endpoints()["Endpoints"]]
    if ENDPOINT_NAME in existing:
        st.info(f"Endpoint '{ENDPOINT_NAME}' already exists. Using it...")
        return
    st.warning(f"Endpoint '{ENDPOINT_NAME}' does not exist! Please create it in SageMaker first.")

def delete_endpoint():
    """Delete the endpoint to stop billing."""
    try:
        sm_client.delete_endpoint(EndpointName=ENDPOINT_NAME)
        st.info(f"Endpoint '{ENDPOINT_NAME}' deleted successfully.")
    except Exception as e:
        st.warning(f"Failed to delete endpoint: {e}")

def predict(number_array):
    """Send MNIST input to SageMaker endpoint and get prediction."""
    payload = json.dumps({"input": number_array.tolist()})
    response = runtime_client.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=payload
    )
    result = json.loads(response["Body"].read().decode())
    return result["prediction"]

# --- Streamlit UI ---
st.title("MNIST SageMaker Demo")

# Deploy endpoint info
with st.spinner("Checking SageMaker endpoint..."):
    deploy_endpoint()

# Slider for number selection
number = st.slider("Select a number (0-9)", 0, 9, 0)

if st.button("Predict"):
    # Convert number to MNIST-like input (28x28 placeholder)
    img = np.zeros((1, 1, 28, 28), dtype=np.float32)
    img[0, 0, :, :] = number / 9.0  # scale 0-1
    prediction = predict(img)
    st.write(f"Predicted number: {prediction}")

# Optional: Delete endpoint to avoid charges
st.button("Delete Endpoint (Stop Billing)", on_click=delete_endpoint)