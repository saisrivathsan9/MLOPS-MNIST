import sagemaker
from sagemaker.pytorch import PyTorch
import os



# Configuration
sagemaker_session = sagemaker.Session()

# Replace with your S3 bucket
bucket = os.environ.get("S3_BUCKET")    
prefix = "mnist-mlops"

# SageMaker IAM role
role = os.environ.get("SAGEMAKER_ROLE_ARN") 

# Local source directory 
source_dir = "./src"
entry_point = "train.py"

# Hyperparameters
hyperparameters = {
    "batch_size": 100,
    "num_epochs": 5,
    "lr": 0.01
}

# Create PyTorch Estimator
estimator = PyTorch(
    entry_point=entry_point,
    source_dir=source_dir,
    role=role,
    framework_version="2.2.0",  
    py_version="py310",
    instance_count=1,
    instance_type="ml.t2.medium",
    hyperparameters=hyperparameters,
    output_path=f"s3://{bucket}/{prefix}/model", 
)

# Launch Training Job
print("Launching SageMaker training job...")
estimator.fit()
print(f"Training complete! Model saved to s3://{bucket}/{prefix}/model")