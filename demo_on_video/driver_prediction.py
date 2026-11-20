import os
import sys
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model_loader import predict_driver_state

def predict_result(frame, model_type="efficientnet"):
    """
    Takes a video frame (numpy array or PIL image) and returns the predicted activity string.
    Matches original project API signature.
    """
    res = predict_driver_state(frame, model_type=model_type)
    return res["class_name"]

def predict_result_with_conf(frame, model_type="efficientnet"):
    """
    Returns (label_string, confidence_float, is_distracted_bool).
    """
    res = predict_driver_state(frame, model_type=model_type)
    return res["class_name"], res["confidence"], res["is_distracted"]

if __name__ == "__main__":
    test_path = os.path.join(PROJECT_ROOT, "images", "img_22_SAFE_DRIVING.jpg")
    if os.path.exists(test_path):
        import cv2
        img = cv2.imread(test_path)
        label = predict_result(img)
        print("Test prediction:", label)
