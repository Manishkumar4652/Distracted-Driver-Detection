"""
Model loader and inference engine for Distracted Driver Detection.
Supports both:
  1. EfficientNet-B0 (10 State Farm classes: c0 to c9)
  2. YOLO-based Distraction Classifier (fast real-time inference)
"""

import os
import sys
import time
import json
import tempfile
import torch

# Ensure Ultralytics config is writable in sandboxed environments
os.environ["YOLO_CONFIG_DIR"] = os.path.join(tempfile.gettempdir(), "Ultralytics")
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

# Base directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "model")
JSON_DIR = os.path.join(BASE_DIR, "json_files")

# Class dictionary
CLASS_MAP = {
    "c0": {"name": "SAFE_DRIVING", "label": "Safe Driving", "distracted": False, "desc": "Driver is driving attentively with hands on the wheel."},
    "c1": {"name": "TEXTING_RIGHT", "label": "Texting (Right Hand)", "distracted": True, "desc": "Driver is texting using their right hand."},
    "c2": {"name": "TALKING_PHONE_RIGHT", "label": "Talking on Phone (Right)", "distracted": True, "desc": "Driver is holding phone to right ear."},
    "c3": {"name": "TEXTING_LEFT", "label": "Texting (Left Hand)", "distracted": True, "desc": "Driver is texting using their left hand."},
    "c4": {"name": "TALKING_PHONE_LEFT", "label": "Talking on Phone (Left)", "distracted": True, "desc": "Driver is holding phone to left ear."},
    "c5": {"name": "OPERATING_RADIO", "label": "Operating Radio", "distracted": True, "desc": "Driver is reaching for / adjusting dashboard controls or radio."},
    "c6": {"name": "DRINKING", "label": "Drinking / Eating", "distracted": True, "desc": "Driver is consuming a drink or food while driving."},
    "c7": {"name": "REACHING_BEHIND", "label": "Reaching Behind", "distracted": True, "desc": "Driver is turned around reaching into the backseat."},
    "c8": {"name": "HAIR_AND_MAKEUP", "label": "Hair and Makeup", "distracted": True, "desc": "Driver is grooming hair or applying makeup in the mirror."},
    "c9": {"name": "TALKING_TO_PASSENGER", "label": "Talking to Passenger", "distracted": True, "desc": "Driver is turned towards passenger and not watching the road."}
}

CLASS_KEYS = [f"c{i}" for i in range(10)]

# Below this confidence (%) the prediction is flagged as unreliable in the UI
LOW_CONFIDENCE_THRESHOLD = 50.0

# Preprocessing transform for EfficientNet
EFFICIENTNET_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

_cached_efficientnet = None
_cached_yolo = None

def get_efficientnet_model(weights_path=None):
    """Loads and caches the 10-class EfficientNet-B0 model."""
    global _cached_efficientnet
    if _cached_efficientnet is not None:
        return _cached_efficientnet
    
    if weights_path is None:
        weights_path = os.path.join(MODEL_DIR, "best_efficientnet_b0.pth")
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at: {weights_path}")
    
    model = models.efficientnet_b0()
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 10)
    
    checkpoint = torch.load(weights_path, map_location="cpu")
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    _cached_efficientnet = model
    return model

def get_yolo_model(weights_path=None):
    """Loads and caches the YOLO distraction classification model."""
    global _cached_yolo
    if _cached_yolo is not None:
        return _cached_yolo
    
    from ultralytics import YOLO
    if weights_path is None:
        weights_path = os.path.join(MODEL_DIR, "yolo_distraction_s.pt")
        if not os.path.exists(weights_path):
            weights_path = os.path.join(MODEL_DIR, "yolo_distraction_n.pt")
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"YOLO weights not found at: {weights_path}")
        
    model = YOLO(weights_path)
    _cached_yolo = model
    return model

def to_pil_image(image_input):
    """Converts various image representations into a PIL RGB Image."""
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 2:
            return Image.fromarray(image_input).convert("RGB")
        elif image_input.shape[2] == 3:
            # Assume BGR if from OpenCV, convert to RGB
            return Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        elif image_input.shape[2] == 4:
            return Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGRA2RGB))
    elif isinstance(image_input, str):
        if os.path.exists(image_input):
            return Image.open(image_input).convert("RGB")
        raise FileNotFoundError(f"Image file does not exist: {image_input}")
    raise ValueError(f"Unsupported image type: {type(image_input)}")

def predict_efficientnet(image_input):
    """
    Predict driver state using 10-Class EfficientNet-B0 model.
    Returns structured dictionary with predictions and probabilities.
    """
    model = get_efficientnet_model()
    pil_img = to_pil_image(image_input)
    
    t0 = time.time()
    tensor = EFFICIENTNET_TRANSFORM(pil_img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        
    inference_time = (time.time() - t0) * 1000.0
    pred_idx = int(np.argmax(probs))
    pred_code = CLASS_KEYS[pred_idx]
    info = CLASS_MAP[pred_code]
    
    probabilities = {
        CLASS_MAP[c]["label"]: float(probs[i])
        for i, c in enumerate(CLASS_KEYS)
    }
    
    return {
        "model": "EfficientNet-B0 (10 Classes)",
        "class_code": pred_code,
        "class_name": info["name"],
        "class_label": info["label"],
        "description": info["desc"],
        "confidence": float(probs[pred_idx] * 100.0),
        "is_distracted": info["distracted"],
        "probabilities": probabilities,
        "low_confidence": float(probs[pred_idx] * 100.0) < LOW_CONFIDENCE_THRESHOLD,
        "inference_ms": round(inference_time, 2)
    }

def predict_yolo(image_input):
    """
    Predict driver state using YOLO distraction classifier.
    Returns structured dictionary with predictions and probabilities.
    """
    model = get_yolo_model()
    pil_img = to_pil_image(image_input)
    
    t0 = time.time()
    results = model.predict(pil_img, imgsz=224, verbose=False)[0]
    inference_time = (time.time() - t0) * 1000.0
    
    top1 = int(results.probs.top1)
    label = model.names[top1]
    conf = float(results.probs.top1conf) * 100.0
    
    is_distracted = (label.lower() != "safe")
    
    # Map to 4 general states
    probabilities = {
        model.names[i]: float(results.probs.data[i].item())
        for i in range(len(model.names))
    }
    
    return {
        "model": "YOLO-Distraction (Real-Time)",
        "class_code": label.lower(),
        "class_name": label.upper(),
        "class_label": label.capitalize(),
        "description": f"Driver state categorized as {label}.",
        "confidence": round(conf, 2),
        "is_distracted": is_distracted,
        "probabilities": probabilities,
        "low_confidence": conf < LOW_CONFIDENCE_THRESHOLD,
        "inference_ms": round(inference_time, 2)
    }

def predict_driver_state(image_input, model_type="efficientnet"):
    """Unified entry point for predictions."""
    if model_type.lower() == "yolo":
        return predict_yolo(image_input)
    return predict_efficientnet(image_input)
