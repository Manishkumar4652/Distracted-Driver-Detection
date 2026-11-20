"""
Process input video to detect distracted driver behaviors frame-by-frame.
Can be executed from CLI or imported as a module.
"""

import os
import sys
import time
import argparse
from collections import deque, Counter
import cv2
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from demo_on_video.driver_prediction import predict_result_with_conf

def annotate_frame(frame, label, confidence, is_distracted, frame_idx, total_frames, fps):
    """Draws a professional, clean alert HUD on the output frame."""
    h, w = frame.shape[:2]
    annotated = frame.copy()
    
    # Color scheme
    if is_distracted:
        banner_color = (30, 30, 220)       # Red (BGR)
        status_text = f"DISTRACTED: {label} ({confidence:.1f}%)"
    else:
        banner_color = (40, 180, 40)       # Green (BGR)
        status_text = f"SAFE DRIVING ({confidence:.1f}%)"
        
    # Top Banner
    banner_height = int(max(50, h * 0.10))
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_height), banner_color, -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)
    
    # White border below banner
    cv2.line(annotated, (0, banner_height), (w, banner_height), (255, 255, 255), 2)
    
    # Status text
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = max(0.6, w / 1000.0)
    thickness = 2
    cv2.putText(annotated, status_text, (20, int(banner_height * 0.68)), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    # Bottom Telemetry Bar
    bar_h = 35
    bottom_y = h - bar_h
    overlay2 = annotated.copy()
    cv2.rectangle(overlay2, (0, bottom_y), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay2, 0.7, annotated, 0.3, 0, annotated)
    
    telemetry = f"Frame: {frame_idx}/{total_frames} | Speed: {fps:.1f} FPS | Driver Safety System"
    cv2.putText(annotated, telemetry, (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    
    return annotated

def process_video(
    input_path="input_video.mp4",
    output_path="output_video.mp4",
    model_type="yolo",
    queue_size=7,
    stride=1,
    max_frames=None,
    show_display=False,
    progress_callback=None
):
    """
    Reads input video, predicts activity for frames, annotates HUD, and saves output video.
    """
    if not os.path.isabs(input_path):
        candidate_paths = [
            input_path,
            os.path.join(os.path.dirname(__file__), input_path),
            os.path.join(PROJECT_ROOT, "demo_on_video", input_path),
            os.path.join(PROJECT_ROOT, input_path)
        ]
        for p in candidate_paths:
            if os.path.exists(p):
                input_path = p
                break

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")

    vs = cv2.VideoCapture(input_path)
    if not vs.isOpened():
        raise RuntimeError(f"Could not open video file: {input_path}")

    total_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = vs.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if max_frames and max_frames < total_frames:
        total_frames = max_frames

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))
    if not writer.isOpened():
        # Fallback to AVI
        alt_output = os.path.splitext(output_path)[0] + ".avi"
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(alt_output, fourcc, video_fps, (width, height))
        output_path = alt_output

    history = deque(maxlen=queue_size)
    frame_idx = 0
    start_time = time.time()
    last_label = "SAFE_DRIVING"
    last_conf = 95.0
    last_distracted = False
    
    stats = {"safe_frames": 0, "distracted_frames": 0, "activities": {}}

    print(f"[INFO] Processing {input_path} ({total_frames} frames) with model '{model_type}'...")
    
    try:
        while True:
            if max_frames and frame_idx >= max_frames:
                break
            grabbed, frame = vs.read()
            if not grabbed:
                break
                
            frame_idx += 1
                
            if frame_idx % stride == 0:
                label, conf, is_distracted = predict_result_with_conf(frame, model_type=model_type)
                history.append((label, conf, is_distracted))
                
                # Smooth prediction using majority vote from rolling queue
                votes = [item[0] for item in history]
                most_common_label = Counter(votes).most_common(1)[0][0]
                
                # Average confidence for the winning class
                confs = [item[1] for item in history if item[0] == most_common_label]
                avg_conf = float(np.mean(confs)) if confs else conf
                
                last_label = most_common_label
                last_conf = avg_conf
                last_distracted = (last_label != "SAFE_DRIVING" and last_label != "SAFE")
            
            # Tally statistics
            if last_distracted:
                stats["distracted_frames"] += 1
            else:
                stats["safe_frames"] += 1
            stats["activities"][last_label] = stats["activities"].get(last_label, 0) + 1

            # Calculate current processing FPS
            elapsed = time.time() - start_time
            current_fps = frame_idx / elapsed if elapsed > 0 else 0.0

            annotated = annotate_frame(
                frame,
                last_label,
                last_conf,
                last_distracted,
                frame_idx,
                total_frames,
                current_fps
            )
            
            writer.write(annotated)
            
            if show_display:
                try:
                    cv2.imshow("Distracted Driver Detection Preview", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("[INFO] Processing interrupted by user.")
                        break
                except Exception:
                    show_display = False

            if progress_callback:
                progress_callback(frame_idx, total_frames, annotated, stats)

            if frame_idx % 25 == 0 or frame_idx == total_frames:
                pct = (frame_idx / total_frames) * 100.0 if total_frames > 0 else 0
                print(f"[PROGRESS] Frame {frame_idx}/{total_frames} ({pct:.1f}%) - Activity: {last_label} ({last_conf:.1f}%)")

    finally:
        vs.release()
        writer.release()
        if show_display:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    total_time = time.time() - start_time
    stats["total_frames"] = frame_idx
    stats["elapsed_seconds"] = round(total_time, 2)
    stats["avg_fps"] = round(frame_idx / total_time, 2) if total_time > 0 else 0.0
    stats["output_file"] = output_path
    
    print(f"[INFO] Completed in {total_time:.1f}s ({stats['avg_fps']} FPS). Output saved to: {output_path}")
    return stats

def main():
    parser = argparse.ArgumentParser(description="Distracted Driver Detection - Video Processor")
    parser.add_argument("-i", "--input", default="input_video.mp4", help="Path to input video file")
    parser.add_argument("-o", "--output", default="output_video.mp4", help="Path to output video file")
    parser.add_argument("-m", "--model", default="yolo", choices=["efficientnet", "yolo"], help="Model type to use")
    parser.add_argument("-s", "--stride", type=int, default=1, help="Frame step stride (e.g. 2 to skip every other frame)")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument("--display", action="store_true", help="Display live preview window while processing")
    args = parser.parse_args()

    process_video(
        input_path=args.input,
        output_path=args.output,
        model_type=args.model,
        stride=args.stride,
        max_frames=args.max_frames,
        show_display=args.display
    )

if __name__ == "__main__":
    main()
