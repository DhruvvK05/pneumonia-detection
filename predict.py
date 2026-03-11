import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

MODEL_PATH = "models/final_model.keras"
IMG_SIZE = 224

CLASS_NAMES = ['NORMAL', 'bacterial', 'viral']

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

def predict_image(img_path):

    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)

    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    print("\nPrediction:", predicted_class)
    print("Confidence: {:.2f}%".format(confidence))


if __name__ == "__main__":

    # Expect image path as command-line argument
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]

    if os.path.exists(img_path):
        predict_image(img_path)
    else:
        print("Image path not found!")
