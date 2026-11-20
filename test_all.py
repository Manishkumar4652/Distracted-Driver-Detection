"""
End-to-End Test Suite for Distracted Driver Detection Project.
Verifies all modules, models, pipelines, and configurations.
"""

import os
import sys
import time

def run_tests():
    print("=" * 65)
    print(" 🧪 RUNNING VERIFICATION SUITE: DISTRACTED DRIVER DETECTION")
    print("=" * 65)
    
    passed_count = 0
    total_tests = 6

    # Test 1: Configuration files
    print("\n[Test 1/6] Checking configuration and pickle files...")
    assert os.path.exists("pickle_files/labels_list.pkl"), "Missing labels_list.pkl"
    assert os.path.exists("json_files/class_name_map.json"), "Missing json_files/class_name_map.json"
    assert os.path.exists("demo_on_image/class_name_map.json"), "Missing demo_on_image/class_name_map.json"
    print("  -> Configuration & mapping files: [PASSED] ✅")
    passed_count += 1

    # Test 2: Model Files Exist
    print("\n[Test 2/6] Checking model weights...")
    assert os.path.exists("model/best_efficientnet_b0.pth"), "Missing best_efficientnet_b0.pth"
    assert os.path.exists("model/yolo_distraction_s.pt"), "Missing yolo_distraction_s.pt"
    print(f"  -> EfficientNet weights: {os.path.getsize('model/best_efficientnet_b0.pth')/(1024*1024):.1f} MB [PASSED] ✅")
    print(f"  -> YOLO weights: {os.path.getsize('model/yolo_distraction_s.pt')/(1024*1024):.1f} MB [PASSED] ✅")
    passed_count += 1

    # Test 3: EfficientNet Prediction
    print("\n[Test 3/6] Testing 10-Class EfficientNet Inference...")
    from src.model_loader import predict_driver_state
    test_img = "images/img_12_TEXTING_RIGHT.jpg"
    res = predict_driver_state(test_img, model_type="efficientnet")
    assert res["class_code"] == "c1", f"Expected c1, got {res['class_code']}"
    assert res["is_distracted"] is True, "Expected is_distracted=True"
    print(f"  -> Prediction on {test_img}: {res['class_name']} ({res['confidence']:.1f}%) in {res['inference_ms']}ms [PASSED] ✅")
    passed_count += 1

    # Test 4: YOLO Real-Time Prediction
    print("\n[Test 4/6] Testing YOLO Real-Time Inference...")
    res_yolo = predict_driver_state(test_img, model_type="yolo")
    assert res_yolo["is_distracted"] is True
    print(f"  -> YOLO Prediction: {res_yolo['class_name']} ({res_yolo['confidence']:.1f}%) in {res_yolo['inference_ms']}ms [PASSED] ✅")
    passed_count += 1

    # Test 5: Legacy Image and Video Modules
    print("\n[Test 5/6] Testing Legacy Module Signatures...")
    from demo_on_image.predictionOnImage import return_prediction
    from demo_on_video.driver_prediction import predict_result
    
    label_img = return_prediction(test_img)
    label_vid = predict_result(test_img)
    assert label_img == "TEXTING_RIGHT"
    assert label_vid == "TEXTING_RIGHT"
    print(f"  -> return_prediction(): {label_img} [PASSED] ✅")
    print(f"  -> predict_result(): {label_vid} [PASSED] ✅")
    passed_count += 1

    # Test 6: Video Processing Pipeline
    print("\n[Test 6/6] Testing Video Processing Pipeline on 10 frames...")
    from demo_on_video.predict_distracted import process_video
    input_vid = "demo_on_video/input_video.mp4"
    temp_out = "temp_test_video.mp4"
    if os.path.exists(temp_out):
        os.remove(temp_out)
        
    stats = process_video(
        input_path=input_vid,
        output_path=temp_out,
        model_type="yolo",
        max_frames=10,
        show_display=False
    )
    assert stats["total_frames"] == 10
    assert os.path.exists(temp_out)
    os.remove(temp_out)
    print(f"  -> Video pipeline: Processed 10 frames at {stats['avg_fps']} FPS [PASSED] ✅")
    passed_count += 1

    print("\n" + "=" * 65)
    print(f" 🎉 ALL {passed_count}/{total_tests} TESTS PASSED SUCCESSFULLY! PROJECT IS 100% READY!")
    print("=" * 65)

if __name__ == "__main__":
    run_tests()
