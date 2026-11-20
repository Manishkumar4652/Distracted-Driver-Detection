# 🚗 Distracted Driver Detection System

[![Python 3.10 - 3.13](https://img.shields.io/badge/Python-3.10%20--%203.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-orange.svg)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Status-Fully%20Working%20%E2%9C%85-brightgreen.svg)](#)

An intelligent in-cabin driver monitoring computer vision system that detects and classifies distracted driving behaviors in real time from dashboard camera feeds and images.

---

## 📌 Project Overview & Motivation
According to the **CDC (Centers for Disease Control and Prevention)**, approximately **3,000 people are killed** and over **425,000 individuals are injured** each year in crashes involving a distracted driver. 

While conventional vehicle safety systems monitor exterior factors (lane drift, distance from other cars), this project introduces an **interior computer vision perception system**. By analyzing the driver's posture, hands, and facial orientation, the system proactively detects unsafe habits (like texting, calling, or turning away) and triggers safety warnings.

---

## 🏷️ The 10 Driver Behavior Classes
Trained on the **State Farm Distracted Driver Detection Dataset** (Kaggle), the system classifies driver behavior into **10 distinct categories**:

| Class ID | Behavior Name | Category | Description |
|---|---|---|---|
| **c0** | **SAFE_DRIVING** | ✅ Safe | Driver is attentive with eyes on the road and hands on the steering wheel |
| **c1** | **TEXTING_RIGHT** | ⚠️ Distracted | Driver is typing/texting using right hand |
| **c2** | **TALKING_PHONE_RIGHT** | ⚠️ Distracted | Driver is holding phone to right ear |
| **c3** | **TEXTING_LEFT** | ⚠️ Distracted | Driver is typing/texting using left hand |
| **c4** | **TALKING_PHONE_LEFT** | ⚠️ Distracted | Driver is holding phone to left ear |
| **c5** | **OPERATING_RADIO** | ⚠️ Distracted | Reaching for dashboard controls, GPS, or radio |
| **c6** | **DRINKING** | ⚠️ Distracted | Consuming a drink or food while driving |
| **c7** | **REACHING_BEHIND** | ⚠️ Distracted | Turned around looking/reaching into the backseat |
| **c8** | **HAIR_AND_MAKEUP** | ⚠️ Distracted | Grooming hair or looking into the vanity mirror |
| **c9** | **TALKING_TO_PASSENGER** | ⚠️ Distracted | Head turned toward passenger, distracted from road |

---

## 🧠 System Architecture & Models

1. **10-Class Deep Classifier (EfficientNet-B0)**
   - Pre-trained on ImageNet and fine-tuned on the State Farm dataset.
   - Evaluates all 10 granular driver behavior classes with full softmax probability distributions.
   - Weights stored locally at: `model/best_efficientnet_b0.pth`.

2. **Ultra-Fast Real-Time Classifier (YOLO Distraction)**
   - Delivers **25–40 FPS on standard CPU** (100+ FPS on GPU).
   - Ideal for live streaming feeds, dashboard camera video files, and edge devices.
   - Weights stored locally at: `model/yolo_distraction_s.pt`.

3. **Temporal Rolling Smoothing**
   - Implements a rolling deque buffer with majority voting over consecutive frames to prevent frame-to-frame flicker and false alarms.

---

## 🚀 Quick Start Guide (How to Run)

### 1. Run the Interactive Web Dashboard (Recommended for College Demo)
Simply run the launcher script:
```bash
python run_app.py
```
*(Alternatively: `streamlit run app.py`)*

The web application opens automatically in your browser with 4 features:
1. **🖼️ Image Classification**: Test single images using the sample gallery or upload any custom photo.
2. **🎥 Video File Analysis**: Run automated frame-by-frame analysis on `input_video.mp4` or your own video.
3. **📹 Live Webcam Snapshot**: Take a photo using your laptop webcam and get an instant detection.
4. **📚 Project Viva Guide**: Built-in architecture details, class tables, and college viva questions.

---

### 2. Run Video Processing via Command Line
To process a video file directly and generate an annotated video with an alert HUD:
```bash
python demo_on_video/predict_distracted.py --input demo_on_video/input_video.mp4 --output output_annotated.mp4
```
Options:
- `--model yolo` or `--model efficientnet`
- `--max-frames 100` (limits frames for quick demo)
- `--display` (shows live preview window while processing)

---

### 3. Run Automated Tests
To verify all models and pipelines:
```bash
python test_all.py
```
All 6 tests should pass with green checkmarks (`[PASSED] ✅`).

---

## 📂 Project Directory Structure

```text
Distracted-Driver-Detection/
├── app.py                      # Main Streamlit web dashboard
├── run_app.py                  # One-click application launcher
├── test_all.py                 # Automated test suite
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── src/
│   └── model_loader.py         # Unified model loading & inference engine
│
├── model/
│   ├── best_efficientnet_b0.pth # 10-Class EfficientNet weights
│   ├── yolo_distraction_s.pt   # YOLO distraction classifier
│   └── yolo_distraction_n.pt   # Lightweight nano YOLO
│
├── demo_on_image/
│   ├── predictionOnImage.py    # Image prediction module
│   ├── streamlit_demo.py       # Legacy demo launcher
│   ├── class_name_map.json     # Class mapping definitions
│   └── test_images/            # Test driver photos
│
├── demo_on_video/
│   ├── predict_distracted.py   # Full video processing pipeline
│   ├── driver_prediction.py    # Frame prediction wrapper
│   └── input_video.mp4         # Sample dashboard video
│
├── pickle_files/
│   └── labels_list.pkl         # Serialized class labels mapping
│
├── json_files/
│   └── class_name_map.json     # JSON class configuration
│
└── Training Notebooks/         # Jupyter training experiments
    ├── Custom CNN Batchwise.ipynb
    ├── Custom CNN Non Batch.ipynb
    ├── VGG16 Shallow Non Batch.ipynb
    └── VGG16 Fine Tuned Layer Freezing.ipynb
```

---

## 🎓 College Viva / Presentation Talking Points

If your professor or examiner asks questions during your evaluation, here are strong, technical answers:

1. **Why is Transfer Learning used instead of training from scratch?**
   - *Answer:* Training a deep CNN from scratch requires millions of parameters and often overfits on smaller domain-specific datasets. Transfer learning leverages pre-trained feature extractors (like ImageNet weights) that already understand fundamental visual patterns (edges, shapes, textures), allowing faster convergence and higher accuracy.

2. **How does the system distinguish safe driving from adjusting the radio?**
   - *Answer:* The network detects driver hand position and gaze direction. In safe driving, hands remain at the 10-and-2 or 9-and-3 positions on the wheel with head facing forward. For operating the radio, the right arm is extended toward the center console and the head is tilted downward.

3. **How do you prevent false positives in video processing?**
   - *Answer:* We apply **temporal smoothing** using a sliding window (`deque`). Rather than relying on a single isolated frame which might suffer from motion blur or transient gestures, predictions are averaged over consecutive frames using majority voting before raising an alarm.

4. **What are the real-world deployment challenges?**
   - *Answer:* Lighting variations (night driving, shadows), camera placement angle, and low-latency requirements. These can be addressed using infrared cabin illuminators, multi-angle camera training, and model quantization (INT8/TFLite/ONNX) on edge devices like NVIDIA Jetson or Raspberry Pi.

---
**Author:** College Academic Project  
**Status:** Tested & Fully Functional ✅
