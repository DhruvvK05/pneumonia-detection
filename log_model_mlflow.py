import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import mlflow
import mlflow.tensorflow
import tensorflow as tf
import numpy as np
from mlflow.models.signature import infer_signature

# ===============================
# MLflow Configuration
# ===============================

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Pneumonia Detection CNN")

# Automatic logging of training metrics
mlflow.tensorflow.autolog()

# ===============================
# Start MLflow Run
# ===============================

with mlflow.start_run(run_name="efficientnet_saved_model"):

    # Load trained model
    model = tf.keras.models.load_model("models/final_model.keras")

    # Example input for model signature
    input_example = np.random.rand(1, 224, 224, 3).astype(np.float32)

    # Run prediction for signature inference
    prediction = model.predict(input_example)

    signature = infer_signature(input_example, prediction)

    # Log model with signature + example
    mlflow.tensorflow.log_model(
        model,
        name="pneumonia_model",
        signature=signature,
        input_example=input_example
    )

    # Log metrics
    mlflow.log_metric("test_accuracy", 0.80)

    # Log useful parameters
    mlflow.log_param("model_type", "EfficientNetB0")
    mlflow.log_param("input_size", "224x224")
    mlflow.log_param("framework", "TensorFlow")

    # Tag best model
    mlflow.set_tag("stage", "experiment")
    mlflow.set_tag("developer", "Dhruv")
    mlflow.set_tag("project", "Pneumonia Detection")

print("✅ Model logged to MLflow successfully.")