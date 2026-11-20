"""
Distracted Driver Detection - Professional White & Red Automotive Dashboard
Built with Streamlit, PyTorch, and OpenCV
"""

import os
import sys
import time
import random
import tempfile
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import streamlit as st

# Setup paths
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model_loader import (
    predict_driver_state,
    CLASS_MAP,
    CLASS_KEYS
)
from demo_on_video.predict_distracted import process_video

# Dataset paths
DATASET_ROOT = os.path.join(PROJECT_ROOT, "state-farm-distracted-driver-detection")
DATASET_AVAILABLE = os.path.exists(DATASET_ROOT)
TRAIN_DIR = os.path.join(DATASET_ROOT, "imgs", "train") if DATASET_AVAILABLE else None
DRIVER_CSV_PATH = os.path.join(DATASET_ROOT, "driver_imgs_list.csv") if DATASET_AVAILABLE else None

# Streamlit Page Config
st.set_page_config(
    page_title="GuardianEye | Distracted Driver Detection",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Professional White & Red Automotive Theme)
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
    }

    /* Top Hero Banner */
    .hero-card {
        background: #FFFFFF;
        border: 1px solid #FEE2E2;
        border-left: 6px solid #DC2626;
        border-radius: 14px;
        padding: 22px 28px;
        margin-bottom: 24px;
        box-shadow: 0 4px 15px -3px rgba(220, 38, 38, 0.07), 0 2px 6px -2px rgba(0, 0, 0, 0.04);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 15px;
    }
    .hero-title {
        font-size: 1.95rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .hero-title span {
        color: #DC2626;
    }
    .hero-subtitle {
        font-size: 0.96rem;
        color: #64748B;
        margin-top: 6px;
        margin-bottom: 0;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: #FFF1F2;
        color: #BE123C;
        border: 1px solid #FECDD3;
        padding: 8px 16px;
        border-radius: 30px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .status-dot {
        width: 9px;
        height: 9px;
        background-color: #E11D48;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 0 3px rgba(225, 29, 72, 0.25);
    }

    /* Content Cards */
    .ui-card {
        background: #FFFFFF;
        border: 1px solid #F1F5F9;
        border-top: 3px solid #DC2626;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }

    /* Alert Badges */
    .danger-alert-box {
        background: linear-gradient(135deg, #DC2626 0%, #B91C1C 100%);
        color: #FFFFFF !important;
        padding: 18px 24px;
        border-radius: 12px;
        font-weight: 800;
        font-size: 1.35rem;
        letter-spacing: -0.01em;
        box-shadow: 0 8px 20px -4px rgba(220, 38, 38, 0.35);
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .safe-alert-box {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        color: #FFFFFF !important;
        padding: 18px 24px;
        border-radius: 12px;
        font-weight: 800;
        font-size: 1.35rem;
        letter-spacing: -0.01em;
        box-shadow: 0 8px 20px -4px rgba(5, 150, 105, 0.35);
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #FEE2E2;
        border-radius: 12px;
        padding: 14px 18px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
    }
    div[data-testid="stMetricLabel"] {
        color: #64748B !important;
        font-weight: 600;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    div[data-testid="stMetricValue"] {
        color: #0F172A !important;
        font-weight: 800;
        font-size: 1.6rem !important;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #DC2626 0%, #B91C1C 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        border: none !important;
        padding: 10px 24px !important;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.25) !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(220, 38, 38, 0.35) !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #FAFAFA !important;
        border-right: 1px solid #FEE2E2 !important;
    }

    /* Custom Section Headers */
    .section-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-title::before {
        content: "";
        width: 4px;
        height: 18px;
        background-color: #DC2626;
        border-radius: 2px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Header & Brand
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 10px 0 15px 0;">
        <div style="font-size: 2.8rem; line-height: 1;">🚨</div>
        <div style="font-size: 1.35rem; font-weight: 800; color: #0F172A; letter-spacing: -0.02em; margin-top: 5px;">
            Guardian<span style="color: #DC2626;">Eye</span>
        </div>
        <div style="font-size: 0.78rem; font-weight: 600; color: #DC2626; text-transform: uppercase; letter-spacing: 0.08em;">
            Driver Safety AI System
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr style='margin: 10px 0 20px 0; border: none; border-top: 1px solid #FEE2E2;'>", unsafe_allow_html=True)

    app_mode = st.radio(
        "NAVIGATION",
        [
            "🖼️ Image Classification",
            "🎥 Video File Analysis",
            "📹 Live Webcam Snapshot",
            "📊 Dataset Explorer & Stats",
            "📚 Project Viva & Tech Guide"
        ],
        label_visibility="visible"
    )

    st.markdown("<hr style='margin: 20px 0; border: none; border-top: 1px solid #FEE2E2;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.78rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 8px;'>AI Core Architecture</div>", unsafe_allow_html=True)
    model_choice = st.selectbox(
        "Select Active Backbone:",
        ["EfficientNet-B0 (10 State Farm Classes)", "YOLO Distraction (Ultra-Fast ~35 FPS)"],
        index=0,
        label_visibility="collapsed"
    )
    model_type = "yolo" if "YOLO" in model_choice else "efficientnet"

    st.markdown("<hr style='margin: 20px 0; border: none; border-top: 1px solid #FEE2E2;'>", unsafe_allow_html=True)
    
    if DATASET_AVAILABLE:
        st.markdown("""
        <div style='background: #FFF1F2; border: 1px solid #FECDD3; border-radius: 8px; padding: 12px; margin-bottom: 10px;'>
            <div style='font-size: 0.85rem; font-weight: 700; color: #991B1B;'>📁 Kaggle Dataset Linked</div>
            <div style='font-size: 0.76rem; color: #64748B; margin-top: 3px;'>22,424 Training Images Online</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size: 0.75rem; color: #94A3B8; text-align: center; margin-top: 20px;'>
        Academic Project Submission<br>
        <strong>Computer Vision & Deep Learning</strong>
    </div>
    """, unsafe_allow_html=True)

# Sample images list
SAMPLE_IMAGES = {
    "Texting Right Hand (img_12)": os.path.join(PROJECT_ROOT, "images", "img_12_TEXTING_RIGHT.jpg"),
    "Operating Radio / Dashboard (img_10)": os.path.join(PROJECT_ROOT, "images", "img_10_OPERATING_RADIO.jpg"),
    "Reaching Behind (img_30)": os.path.join(PROJECT_ROOT, "images", "img_30_REACHING_BEHIND.jpg"),
    "Drinking Beverage (img_124)": os.path.join(PROJECT_ROOT, "demo_on_image", "test_images", "img_124.jpg"),
    "Talking on Phone Right (img_186)": os.path.join(PROJECT_ROOT, "demo_on_image", "test_images", "img_186.jpg"),
    "Safe Driving (img_22)": os.path.join(PROJECT_ROOT, "images", "img_22_SAFE_DRIVING.jpg"),
}

# ==========================================
# 1. IMAGE CLASSIFICATION MODE
# ==========================================
if app_mode == "🖼️ Image Classification":
    # Top Hero Card
    st.markdown("""
    <div class="hero-card">
        <div>
            <h1 class="hero-title">🚗 In-Cabin <span>Driver State</span> Classifier</h1>
            <p class="hero-subtitle">Evaluate static camera frames to detect cognitive, visual, and manual distractions in real time.</p>
        </div>
        <div class="status-badge">
            <span class="status-dot"></span> Vision Engine Active
        </div>
    </div>
    """, unsafe_allow_html=True)

    input_options = ["Use Preset Sample Image"]
    if DATASET_AVAILABLE:
        input_options.append("Browse Full Dataset by Class (c0 - c9)")
    input_options.append("Upload Your Own Image")

    input_source = st.radio("SELECT INPUT SOURCE", input_options, horizontal=True)
    
    selected_image = None
    image_name = ""
    ground_truth = None

    if input_source == "Use Preset Sample Image":
        chosen_sample = st.selectbox("Select Scenario:", list(SAMPLE_IMAGES.keys()))
        sample_path = SAMPLE_IMAGES[chosen_sample]
        if os.path.exists(sample_path):
            selected_image = Image.open(sample_path)
            image_name = chosen_sample

    elif input_source == "Browse Full Dataset by Class (c0 - c9)":
        c_choice = st.selectbox(
            "Select Target Driving Class to Test:",
            CLASS_KEYS,
            format_func=lambda k: f"{k}: {CLASS_MAP[k]['name']} ({CLASS_MAP[k]['label']})"
        )
        ground_truth = CLASS_MAP[c_choice]["name"]
        class_folder = os.path.join(TRAIN_DIR, c_choice)
        
        if os.path.exists(class_folder):
            all_files = os.listdir(class_folder)
            col_b1, col_b2 = st.columns([2, 1])
            with col_b1:
                selected_filename = st.selectbox(f"Select Image from {c_choice} ({len(all_files):,} images):", all_files[:200])
            with col_b2:
                st.write("")
                st.write("")
                if st.button("🎲 Pick Random Image"):
                    selected_filename = random.choice(all_files)
            
            chosen_path = os.path.join(class_folder, selected_filename)
            selected_image = Image.open(chosen_path)
            image_name = f"Kaggle [{c_choice}]: {selected_filename}"

    else:
        uploaded_file = st.file_uploader("Upload Driver Image (.jpg, .png, .jpeg)", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            selected_image = Image.open(uploaded_file)
            image_name = uploaded_file.name

    if selected_image is not None:
        st.markdown("<hr style='margin: 20px 0; border: none; border-top: 1px solid #F1F5F9;'>", unsafe_allow_html=True)
        col1, col2 = st.columns([1.05, 1.15], gap="large")
        
        with col1:
            st.markdown('<div class="section-title">Driver Camera Capture</div>', unsafe_allow_html=True)
            st.image(selected_image, caption=f"Source: {image_name}", use_container_width=True)
            if ground_truth:
                st.markdown(f"""
                <div style='background: #F8FAFC; border: 1px solid #E2E8F0; padding: 10px 14px; border-radius: 8px; margin-top: 10px;'>
                    <span style='font-size: 0.8rem; font-weight: 700; color: #64748B; text-transform: uppercase;'>Dataset Ground Truth:</span>
                    <strong style='color: #0F172A; margin-left: 6px;'>{ground_truth}</strong>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="section-title">Neural Classification Output</div>', unsafe_allow_html=True)
            with st.spinner("Processing through deep neural network..."):
                res = predict_driver_state(selected_image, model_type=model_type)

            # Professional High-Contrast Alert Banner
            if res["is_distracted"]:
                st.markdown(f"""
                <div class="danger-alert-box">
                    <span style="font-size: 1.8rem;">🚨</span>
                    <div>
                        <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.9;">Attention Warning</div>
                        <div>DISTRACTED: {res['class_label']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="safe-alert-box">
                    <span style="font-size: 1.8rem;">🛡️</span>
                    <div>
                        <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.9;">Attentive State</div>
                        <div>SAFE DRIVING: Attentive Focus</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Metrics row
            m1, m2, m3 = st.columns(3)
            m1.metric("Confidence", f"{res['confidence']:.1f}%")
            m2.metric("Latency", f"{res['inference_ms']} ms")
            m3.metric("State", "Distracted ❌" if res["is_distracted"] else "Attentive ✅")

            # Activity Description Box
            st.markdown(f"""
            <div style='background: #FFF5F5; border-left: 4px solid #DC2626; padding: 12px 16px; border-radius: 6px; margin: 15px 0;'>
                <strong style='color: #991B1B;'>Activity Assessment:</strong>
                <span style='color: #475569;'> {res['description']}</span>
            </div>
            """, unsafe_allow_html=True)

            # Probability Distribution Chart
            st.markdown('<div class="section-title">Class Probabilities Telemetry</div>', unsafe_allow_html=True)
            probs = res["probabilities"]
            df_probs = pd.DataFrame({
                "Activity": list(probs.keys()),
                "Probability (%)": [round(v * 100, 2) for v in probs.values()]
            }).sort_values(by="Probability (%)", ascending=True)

            st.bar_chart(df_probs.set_index("Activity"), horizontal=True, color="#DC2626")

# ==========================================
# 2. VIDEO FILE ANALYSIS MODE
# ==========================================
elif app_mode == "🎥 Video File Analysis":
    st.markdown("""
    <div class="hero-card">
        <div>
            <h1 class="hero-title">🎥 Video <span>Stream Analysis</span> & HUD Alert</h1>
            <p class="hero-subtitle">Process continuous cabin camera footage with temporal voting smoothing and instant distraction telemetry.</p>
        </div>
        <div class="status-badge">
            <span class="status-dot"></span> Stream Mode Ready
        </div>
    </div>
    """, unsafe_allow_html=True)

    default_video_path = os.path.join(PROJECT_ROOT, "demo_on_video", "input_video.mp4")
    video_source = st.radio("SELECT VIDEO SOURCE", ["Included Sample Video (input_video.mp4)", "Upload Custom Video"], horizontal=True)

    target_video_path = None
    if video_source == "Included Sample Video (input_video.mp4)":
        if os.path.exists(default_video_path):
            target_video_path = default_video_path
            st.info(f"Loaded bundled dashboard video: `input_video.mp4` ({os.path.getsize(default_video_path)/(1024*1024):.2f} MB)")
        else:
            st.error("Default video file not found.")
    else:
        uploaded_video = st.file_uploader("Upload Video File (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])
        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            target_video_path = tfile.name
            st.success(f"Video uploaded: {uploaded_video.name}")

    if target_video_path:
        st.markdown('<div class="section-title">Processing Configuration</div>', unsafe_allow_html=True)
        col_v1, col_v2, col_v3 = st.columns(3)
        with col_v1:
            max_frames_to_process = st.slider("Frames to Analyze:", min_value=15, max_value=300, value=60, step=15)
        with col_v2:
            stride_val = st.slider("Frame Stride (Skip):", min_value=1, max_value=4, value=1)
        with col_v3:
            vid_model = st.selectbox("Inference Model:", ["yolo", "efficientnet"], index=0, format_func=lambda x: "YOLO (Fast ~35 FPS)" if x=="yolo" else "EfficientNet (~8 FPS)")

        if st.button("🚀 Start Safety Telemetry Analysis", type="primary"):
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            preview_container = st.empty()

            output_temp = os.path.join(tempfile.gettempdir(), f"detected_driver_{int(time.time())}.mp4")

            def update_ui(frame_idx, total_frames, annotated_frame, stats):
                progress = min(1.0, frame_idx / total_frames)
                progress_bar.progress(progress)
                status_text.text(f"Processing Frame: {frame_idx}/{total_frames} ({progress*100:.1f}%) | Safe: {stats['safe_frames']} | Distracted: {stats['distracted_frames']}")
                if frame_idx % 3 == 0:
                    rgb_preview = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    preview_container.image(rgb_preview, caption=f"Real-Time HUD Feed (Frame {frame_idx})", use_container_width=True)

            try:
                stats = process_video(
                    input_path=target_video_path,
                    output_path=output_temp,
                    model_type=vid_model,
                    stride=stride_val,
                    max_frames=max_frames_to_process,
                    show_display=False,
                    progress_callback=update_ui
                )

                st.success("🎉 Video Stream Analysis Completed!")
                progress_bar.progress(1.0)

                # Telemetry summary
                st.markdown('<div class="section-title">Telemetry & Incident Metrics</div>', unsafe_allow_html=True)
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Analyzed Frames", stats["total_frames"])
                s2.metric("Processing Time", f"{stats['elapsed_seconds']}s")
                s3.metric("Throughput", f"{stats['avg_fps']} FPS")
                
                safe_pct = (stats["safe_frames"] / stats["total_frames"] * 100) if stats["total_frames"] > 0 else 0
                dist_pct = (stats["distracted_frames"] / stats["total_frames"] * 100) if stats["total_frames"] > 0 else 0
                s4.metric("Distraction Ratio", f"{dist_pct:.1f}%", delta=f"{safe_pct:.1f}% Safe", delta_color="inverse")

                # Activity Breakdown Chart
                if stats["activities"]:
                    st.markdown('<div class="section-title">Incident Frequency Chart</div>', unsafe_allow_html=True)
                    df_acts = pd.DataFrame(list(stats["activities"].items()), columns=["Activity", "Frames Count"])
                    st.bar_chart(df_acts.set_index("Activity"), color="#DC2626")

                if os.path.exists(stats["output_file"]):
                    st.markdown(f"**Annotated Output Video:** `{stats['output_file']}`")
                    with open(stats["output_file"], "rb") as f:
                        st.download_button(
                            label="⬇️ Download Processed Video with HUD",
                            data=f,
                            file_name="annotated_distracted_driver.mp4",
                            mime="video/mp4"
                        )

            except Exception as e:
                st.error(f"Error during video processing: {e}")

# ==========================================
# 3. LIVE WEBCAM SNAPSHOT MODE
# ==========================================
elif app_mode == "📹 Live Webcam Snapshot":
    st.markdown("""
    <div class="hero-card">
        <div>
            <h1 class="hero-title">📹 Live <span>Webcam</span> Safety Check</h1>
            <p class="hero-subtitle">Capture a live photo directly from your webcam to test the driver monitoring AI model in real time.</p>
        </div>
        <div class="status-badge">
            <span class="status-dot"></span> Camera Sensor
        </div>
    </div>
    """, unsafe_allow_html=True)

    camera_image = st.camera_input("Take a snapshot facing the camera:")
    if camera_image is not None:
        pil_cam = Image.open(camera_image)
        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            st.markdown('<div class="section-title">Captured Frame</div>', unsafe_allow_html=True)
            st.image(pil_cam, caption="Webcam Input", use_container_width=True)
        with col2:
            st.markdown('<div class="section-title">Classification Result</div>', unsafe_allow_html=True)
            with st.spinner("Analyzing driver state..."):
                res = predict_driver_state(pil_cam, model_type=model_type)

            if res["is_distracted"]:
                st.markdown(f"""
                <div class="danger-alert-box">
                    <span style="font-size: 1.8rem;">🚨</span>
                    <div>
                        <div style="font-size: 0.8rem; text-transform: uppercase;">Safety Incident</div>
                        <div>DISTRACTED: {res['class_label']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="safe-alert-box">
                    <span style="font-size: 1.8rem;">🛡️</span>
                    <div>
                        <div style="font-size: 0.8rem; text-transform: uppercase;">Attentive Driver</div>
                        <div>SAFE DRIVING: Eyes on Road</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            m_cam1, m_cam2 = st.columns(2)
            m_cam1.metric("Confidence", f"{res['confidence']:.1f}%")
            m_cam2.metric("Latency", f"{res['inference_ms']} ms")
            
            st.markdown(f"""
            <div style='background: #FFF5F5; border-left: 4px solid #DC2626; padding: 12px 16px; border-radius: 6px; margin-top: 15px;'>
                <strong style='color: #991B1B;'>Activity Description:</strong>
                <span style='color: #475569;'> {res['description']}</span>
            </div>
            """, unsafe_allow_html=True)

# ==========================================
# 4. DATASET EXPLORER & STATISTICS MODE
# ==========================================
elif app_mode == "📊 Dataset Explorer & Stats":
    st.markdown("""
    <div class="hero-card">
        <div>
            <h1 class="hero-title">📊 State Farm <span>Dataset Analytics</span></h1>
            <p class="hero-subtitle">Comprehensive exploratory analysis and class distributions from the official Kaggle dataset.</p>
        </div>
        <div class="status-badge">
            <span class="status-dot"></span> 22,424 Images
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not DATASET_AVAILABLE:
        st.warning("State Farm Dataset folder not detected in the workspace root.")
    else:
        if os.path.exists(DRIVER_CSV_PATH):
            df_drivers = pd.read_csv(DRIVER_CSV_PATH)
            
            # Key Metric Cards
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Labeled Training Images", f"{len(df_drivers):,}")
            m2.metric("Unique Driver Subjects", df_drivers['subject'].nunique())
            m3.metric("Monitored Behaviors", df_drivers['classname'].nunique())
            test_count = len(os.listdir(os.path.join(DATASET_ROOT, "imgs", "test"))) if os.path.exists(os.path.join(DATASET_ROOT, "imgs", "test")) else 79726
            m4.metric("Test Holdout Set", f"{test_count:,}")

            st.markdown("<hr style='margin: 20px 0; border: none; border-top: 1px solid #F1F5F9;'>", unsafe_allow_html=True)
            col_d1, col_d2 = st.columns(2, gap="large")
            with col_d1:
                st.markdown('<div class="section-title">Distribution by Driving Behavior (c0 - c9)</div>', unsafe_allow_html=True)
                class_counts = df_drivers['classname'].value_counts().sort_index().reset_index()
                class_counts.columns = ["Class Code", "Image Count"]
                class_counts["Activity"] = class_counts["Class Code"].map(lambda c: CLASS_MAP.get(c, {}).get("label", c))
                st.bar_chart(class_counts.set_index("Activity")["Image Count"], color="#DC2626")

            with col_d2:
                st.markdown('<div class="section-title">Image Balance Across 26 Driver Subjects</div>', unsafe_allow_html=True)
                driver_counts = df_drivers['subject'].value_counts().sort_index().reset_index()
                driver_counts.columns = ["Driver ID", "Image Count"]
                st.bar_chart(driver_counts.set_index("Driver ID")["Image Count"], color="#B91C1C")

            # Data Table Preview
            st.markdown('<div class="section-title">Ground Truth Manifest (driver_imgs_list.csv)</div>', unsafe_allow_html=True)
            st.dataframe(df_drivers.head(15), use_container_width=True)

# ==========================================
# 5. PROJECT VIVA & ARCHITECTURE GUIDE
# ==========================================
else:
    st.markdown("""
    <div class="hero-card">
        <div>
            <h1 class="hero-title">📚 Project Viva & <span>Technical Blueprint</span></h1>
            <p class="hero-subtitle">Comprehensive architectural summary, dataset taxonomy, and examiner question-and-answers.</p>
        </div>
        <div class="status-badge">
            <span class="status-dot"></span> College Reference
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 🎯 1. Problem Statement & Road Safety Urgency
    - According to the **CDC (Centers for Disease Control and Prevention)**, distracted driving claims approximately **3,000 lives** and injures over **425,000 individuals** every single year.
    - Traditional systems only monitor vehicle telemetry (speed, lane departure). This project implements an **In-Cabin Computer Vision Monitoring System** that monitors the driver directly to detect cognitive and manual distractions before an accident occurs.

    ---
    ### 🏷️ 2. The 10 Target Driving Classes (State Farm Kaggle Dataset)
    """)

    classes_data = []
    for c_id in CLASS_KEYS:
        info = CLASS_MAP[c_id]
        classes_data.append({
            "Class ID": c_id,
            "Behavior Name": info["name"],
            "Label": info["label"],
            "Safety Classification": "❌ Distracted" if info["distracted"] else "✅ Safe Driving",
            "Description": info["desc"]
        })
    st.table(pd.DataFrame(classes_data))

    st.markdown("""
    ---
    ### 🧠 3. Machine Learning Architectures Explored
    
    1. **EfficientNet-B0 (10-Class Classifier)**:
       - Uses Compound Scaling (depth, width, and resolution balanced).
       - Pre-trained on ImageNet and fine-tuned on the State Farm dataset.
       - Delivers high Top-1 accuracy with low parameter footprint (~4M parameters).
    
    2. **YOLO-Distraction (Real-Time Classifier)**:
       - High throughput (~25-35 FPS on standard CPU, 100+ FPS on GPU).
       - Optimized for live video streams and in-vehicle edge devices.
    
    3. **Custom CNN & VGG16 (Exploration Notebooks)**:
       - Included in the `Training Notebooks/` folder demonstrating model comparison, layer freezing, and data augmentation experiments.
    
    ---
    ### 🎓 4. High-Yield Viva Questions & Examiner Answers
    - **Q: Why is transfer learning preferred over training from scratch?**
      - *A: ImageNet backbones (VGG, ResNet, EfficientNet) have already learned rich low-level features (edges, textures, shapes). Fine-tuning on domain-specific driver images converges much faster and achieves higher accuracy without overfitting.*
    - **Q: How does the system handle real-time video flicker?**
      - *A: We implemented a temporal rolling window (deque) with majority voting across consecutive frames to prevent single-frame false alerts.*
    - **Q: What is the real-world deployment mechanism?**
      - *A: The model can be converted to ONNX or TensorFlow Lite (TFLite) for embedded hardware like Raspberry Pi, NVIDIA Jetson, or integrated into modern vehicle ADAS (Advanced Driver Assistance Systems).*
    """)

st.markdown("<hr style='margin: 30px 0 15px 0; border: none; border-top: 1px solid #FEE2E2;'>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #94A3B8; font-size: 0.8rem;'>
    <strong>GuardianEye In-Cabin Driver Monitoring System</strong> | College Project Edition | Powered by PyTorch & Streamlit
</div>
""", unsafe_allow_html=True)
