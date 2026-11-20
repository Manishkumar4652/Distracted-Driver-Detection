import os
import sys
import json
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model_loader import predict_driver_state, CLASS_MAP

def return_prediction(image_input):
    """
    Takes an image (path, PIL Image, or numpy array) and returns the predicted activity.
    Matches original project API signature for full backward compatibility.
    """
    res = predict_driver_state(image_input, model_type="efficientnet")
    # Returns the readable class string, e.g. 'TEXTING_RIGHT'
    return res["class_name"]

def predict_with_details(image_input):
    """
    Extended prediction returning full dictionary including confidence and probabilities.
    """
    return predict_driver_state(image_input, model_type="efficientnet")

if __name__ == '__main__':
    test_path = os.path.join(PROJECT_ROOT, "images", "img_12_TEXTING_RIGHT.jpg")
    if os.path.exists(test_path):
        result = return_prediction(test_path)
        print(f"Test prediction: {result}")
    else:
        print("Model loaded successfully.")
