import os
from urllib import request
import tensorflow as tf
import numpy as np
import cv2

MODEL_PATH = 'gender_recognition/gender_recognition.h5'
MODEL_URL = "https://storage.googleapis.com/ai-models-faceswap/gender_recognition.h5"

model = None

def ensure_model_downloaded():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        print(f"Downloading model from {MODEL_URL}...")
        request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")


def load_model_if_needed():
    global model
    if model is None:
        ensure_model_downloaded()
        model = tf.keras.models.load_model(MODEL_PATH)

# model = tf.keras.models.load_model(MODEL_PATH)

def predict_gender(input_image):
    load_model_if_needed()
    input_image = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
    input_image = cv2.resize(input_image, (178, 218))
    input_image = np.array(input_image).astype(np.float32) / 255.0
    input_image = np.expand_dims(input_image, axis=0)

    prediction = model.predict(input_image)

    labels = ['Female', 'Male']
    threshold = 0.5
    predicted_gender = 'Male' if prediction[0][1] > threshold else 'Female'
    prediction_probability = prediction[0][1] if predicted_gender == 'Male' else prediction[0][0]

    return {
        'gender': predicted_gender,
        'probability': float(prediction_probability)
    }
