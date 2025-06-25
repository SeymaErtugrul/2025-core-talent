import cv2
import numpy as np
import tempfile
import os
from movement_detector import LucasKanadeDetector, ObjectMovementDetector, CameraMovementDetector

def test_with_real_video():
    """Test with real video files from TestFolder"""
    print("Testing with real video files from TestFolder...")
    
    # Test video paths
    test_video_path = "TestFolder/test.mp4"
    video_path = "TestFolder/video.mp4"
    
    results = {}
    
    # Test with test.mp4
    if os.path.exists(test_video_path):
        print(f"\n📹 Testing with {test_video_path}...")
        
        # Test Camera Movement Detection
        print("🔍 Testing Camera Movement Detection...")
        camera_detector = CameraMovementDetector(method="ORB", threshold=0.5)
        camera_results = camera_detector.analyze_video(test_video_path, max_frames=50, frame_skip=1)
        results['test_camera'] = camera_results
        print(f"✅ Camera Movement: {len(camera_results['movement_frames'])} movement frames detected")
        
        # Test Lucas-Kanade Object Detection
        print("🎯 Testing Lucas-Kanade Object Detection...")
        lk_detector = LucasKanadeDetector(max_corners=100, quality_level=0.3, min_distance=7)
        lk_results = lk_detector.detect_objects(test_video_path)
        results['test_lucas_kanade'] = lk_results
        print(f"✅ Lucas-Kanade: {len(lk_results.get('object_frames', []))} object movement frames detected")
        
        # Test Farneback Object Detection
        print("🌊 Testing Farneback Object Detection...")
        fb_detector = ObjectMovementDetector(threshold=0.5, flow_threshold=0.5)
        fb_results = fb_detector.detect_objects_video(test_video_path, max_frames=50)
        results['test_farneback'] = fb_results
        print(f"✅ Farneback: {len(fb_results.get('object_frames', []))} object movement frames detected")
    
    # Test with video.mp4
    if os.path.exists(video_path):
        print(f"\n📹 Testing with {video_path}...")
        
        # Test Camera Movement Detection
        print("🔍 Testing Camera Movement Detection...")
        camera_detector = CameraMovementDetector(method="ORB", threshold=0.5)
        camera_results = camera_detector.analyze_video(video_path, max_frames=50, frame_skip=1)
        results['video_camera'] = camera_results
        print(f"✅ Camera Movement: {len(camera_results['movement_frames'])} movement frames detected")
        
        print("🎯 Testing Lucas-Kanade Object Detection...")
        lk_detector = LucasKanadeDetector(max_corners=100, quality_level=0.3, min_distance=7)
        lk_results = lk_detector.detect_objects(video_path)
        results['video_lucas_kanade'] = lk_results
        print(f"✅ Lucas-Kanade: {len(lk_results.get('object_frames', []))} object movement frames detected")
        
        print("🌊 Testing Farneback Object Detection...")
        fb_detector = ObjectMovementDetector(threshold=0.5, flow_threshold=0.5)
        fb_results = fb_detector.detect_objects_video(video_path, max_frames=50)
        results['video_farneback'] = fb_results
        print(f"✅ Farneback: {len(fb_results.get('object_frames', []))} object movement frames detected")
    
    return results

def test_lucas_kanade():
    """Test Lucas-Kanade object detection with synthetic video"""
    print("Testing Lucas-Kanade object detection with synthetic video...")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_file.close()
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_file.name, fourcc, 20.0, (640, 480))
    
    for i in range(50):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        center_x = 320 + int(50 * np.sin(i * 0.2))
        center_y = 240 + int(30 * np.cos(i * 0.3))
        cv2.circle(frame, (center_x, center_y), 30, (0, 255, 0), -1)
        out.write(frame)
    
    out.release()
    
    detector = LucasKanadeDetector(max_corners=100, quality_level=0.3, min_distance=7)
    results = detector.detect_objects(temp_file.name)
    
    print(f"Lucas-Kanade Results: {results}")
    
    # Clean up
    os.unlink(temp_file.name)
    return results

def test_farneback():
    """Test Farneback object detection with synthetic video"""
    print("Testing Farneback object detection with synthetic video...")
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_file.close()
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_file.name, fourcc, 20.0, (640, 480))
    
    for i in range(50):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a moving rectangle
        x = 100 + int(20 * np.sin(i * 0.2))
        y = 200 + int(15 * np.cos(i * 0.3))
        cv2.rectangle(frame, (x, y), (x+100, y+80), (0, 0, 255), -1)
        out.write(frame)
    
    out.release()

    detector = ObjectMovementDetector(threshold=0.5, flow_threshold=0.5)
    results = detector.detect_objects_video(temp_file.name, max_frames=50)
    
    print(f"Farneback Results: {results}")
    
    os.unlink(temp_file.name)
    return results

if __name__ == "__main__":
    print("🎬 Starting comprehensive movement detection tests...")
    print("=" * 60)
    
    print("📹 PHASE 1: Testing with real video files...")
    real_results = test_with_real_video()
    
    print("\n" + "=" * 60)
    print("🎭 PHASE 2: Testing with synthetic videos...")
    
    lk_results = test_lucas_kanade()
    fb_results = test_farneback()
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY:")
    print("=" * 60)
    
    if 'test_camera' in real_results:
        print(f"📹 test.mp4 - Camera Movement: {len(real_results['test_camera']['movement_frames'])} frames")
    if 'test_lucas_kanade' in real_results:
        print(f"📹 test.mp4 - Lucas-Kanade: {len(real_results['test_lucas_kanade'].get('object_frames', []))} frames")
    if 'test_farneback' in real_results:
        print(f"📹 test.mp4 - Farneback: {len(real_results['test_farneback'].get('object_frames', []))} frames")
    
    if 'video_camera' in real_results:
        print(f"📹 video.mp4 - Camera Movement: {len(real_results['video_camera']['movement_frames'])} frames")
    if 'video_lucas_kanade' in real_results:
        print(f"📹 video.mp4 - Lucas-Kanade: {len(real_results['video_lucas_kanade'].get('object_frames', []))} frames")
    if 'video_farneback' in real_results:
        print(f"📹 video.mp4 - Farneback: {len(real_results['video_farneback'].get('object_frames', []))} frames")
    
    print(f"🎭 Synthetic - Lucas-Kanade: {'SUCCESS' if lk_results and 'object_frames' in lk_results else 'FAILED'}")
    print(f"🎭 Synthetic - Farneback: {'SUCCESS' if fb_results and 'object_frames' in fb_results else 'FAILED'}")
    
    print("\n✅ All tests completed!") 