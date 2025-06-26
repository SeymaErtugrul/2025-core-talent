import os
import cv2
import numpy as np
from typing import List
import time


def to_grayscale(frame: np.ndarray) -> np.ndarray:
    if len(frame.shape) == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def open_video(video_path: str):
    """Open video file and return capture object"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    return cap


def read_first_frame(cap):
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return None, None
    return cap, frame


def create_result_dict(object_frames, object_scores, frames, total_frames, details_list=None, annotated_frames=None, method=None, result_type="object"):
    """Create standardized result dictionary"""
    if result_type == "camera":
        result = {
            'movement_frames': object_frames,
            'movement_scores': object_scores,
            'frames': frames,
            'total_frames': total_frames
        }
    else:  
        result = {
            'object_frames': object_frames,
            'object_scores': object_scores,
            'frames': frames,
            'total_frames': total_frames
        }
    
    if details_list is not None:
        result['details_list'] = details_list
    if annotated_frames is not None:
        result['annotated_frames'] = annotated_frames
    if method is not None:
        result['method'] = method
    
    if not object_frames:
        result['message'] = "No Movement Detected"
    
    return result


def get_adaptive_lk_params(frame_area):
    if frame_area > 1000000:
        return (25, 25), 4, 11
    elif frame_area > 500000:
        return (21, 21), 3, 9
    else:
        return (17, 17), 2, 7


class CameraMovementDetector:

    def __init__(self, method="SIFT", threshold=0.5, min_match_count=10, debug=False):
        self.threshold = threshold
        self.min_match_count = min_match_count
        self.debug = debug
        self.method = method.upper()
        if self.method == "SIFT":
            self.detector = cv2.SIFT_create()
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
            self.norm_type = None
        elif self.method == "ORB":
            self.detector = cv2.ORB_create(nfeatures=1000)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            self.norm_type = cv2.NORM_HAMMING
        else:
            raise ValueError("method must be 'SIFT' or 'ORB'")
        self.last_movement_info = {}

    def detect(self, frame1: np.ndarray, frame2: np.ndarray) -> tuple[bool, float, dict[str, any]]:
        gray1 = to_grayscale(frame1)
        gray2 = to_grayscale(frame2)

        kp1, des1 = self.detector.detectAndCompute(gray1, None)
        kp2, des2 = self.detector.detectAndCompute(gray2, None)

        if des1 is None or des2 is None or len(kp1) < self.min_match_count or len(kp2) < self.min_match_count:
            return False, 0.0, {"error": "Not enough keypoints"}
        
        if self.method == "SIFT":
            matches = self.matcher.knnMatch(des1, des2, k=2)
            good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
        else:
            matches = self.matcher.match(des1, des2)
            good_matches = sorted(matches, key=lambda x: x.distance)[:self.min_match_count * 5]

        if len(good_matches) < self.min_match_count:
            return False, 0.0, {"error": f"Not enough good matches: {len(good_matches)}"}
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        if len(src_pts) < 4 or len(dst_pts) < 4:
            return False, 0.0, {"error": f"Not enough points for homography: {len(src_pts)}"}
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            return False, 0.0, {"error": "Homography computation failed"}
        
        dx, dy = H[0, 2], H[1, 2]
        translation = np.sqrt(dx ** 2 + dy ** 2)
        det = H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]
        identity_diff = np.sum(np.abs(H - np.eye(3)))

        inliers = mask.ravel().sum() if mask is not None else 0
        inlier_ratio = inliers / len(good_matches) if good_matches else 0

        movement_score = (0.5 * translation + 0.3 * abs(1 - det) + 0.2 * identity_diff) * inlier_ratio
        is_movement = movement_score > self.threshold

        self.last_movement_info = {
            "translation": translation,
            "determinant": det,
            "identity_diff": identity_diff,
            "movement_score": movement_score,
            "num_matches": len(good_matches),
            "inlier_ratio": inlier_ratio
        }

        return is_movement, movement_score, self.last_movement_info

    def analyze_video(self, video_path, max_frames=120, frame_skip=1, enable_live_viz=False, live_viz_placeholder=None):
        """Analyze camera movement in a video file"""
        cap = open_video(video_path)
        if cap is None:
            return None
        
        cap, prev_frame = read_first_frame(cap)
        if prev_frame is None:
            return None
        
        movement_frames = []
        movement_scores = []
        details_list = []
        frames = []
        
        frame_count = 0
        processed_frames = 0
        
        while True:
            if frame_count > max_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue
            
            is_movement, score, details = self.detect(prev_frame, frame)
            if is_movement:
                movement_frames.append(frame_count)
                movement_scores.append(score)
                details_list.append(details)
            
            # Live visualization
            if enable_live_viz and live_viz_placeholder is not None:
                annotated_frame = frame.copy()
                
                if is_movement:
                    text_color = (0, 255, 0)  # Green for movement
                    movement_text = "CAMERA MOVEMENT DETECTED"
                else:
                    text_color = (0, 0, 255)  # Red for no movement
                    movement_text = "No Camera Movement"
                
                cv2.putText(annotated_frame, movement_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f"Score: {score:.3f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                if details and 'translation' in details:
                    cv2.putText(annotated_frame, f"Translation: {details['translation']:.3f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                if details and 'num_matches' in details:
                    cv2.putText(annotated_frame, f"Matches: {details['num_matches']}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                h, w = annotated_frame.shape[:2]
                border_color = text_color if is_movement else (100, 100, 100)
                cv2.rectangle(annotated_frame, (0, 0), (w-1, h-1), border_color, 3)
                
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                live_viz_placeholder.image(rgb_frame, caption=f"Frame {frame_count} - Camera Movement Detection", use_container_width=True, width=350)
            
            frames.append(frame)
            prev_frame = frame
            frame_count += 1
            processed_frames += 1
        
        cap.release()
        return create_result_dict(movement_frames, movement_scores, frames, frame_count, details_list, result_type="camera")


class LucasKanadeAnalyzer:
    
    def __init__(self, max_corners=150, quality_level=0.1, min_distance=7):
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        
    def analyze_video(self, video_path, max_frames=120, frame_skip=1, enable_live_viz=False, live_viz_placeholder=None):
        cap = open_video(video_path)
        if cap is None:
            return None
        
        cap, prev_frame = read_first_frame(cap)
        if prev_frame is None:
            return None
        
        prev_gray = to_grayscale(prev_frame)
        frame_area = prev_gray.shape[0] * prev_gray.shape[1]
        
        win_size, max_level, block_size = get_adaptive_lk_params(frame_area)
        
        lk_params = dict(
            winSize=win_size,
            maxLevel=max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.03)
        )
        
        feature_params = dict(
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=block_size,
            useHarrisDetector=True,
            k=0.04
        )
        
        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
        if p0 is None:
            cap.release()
            return None
        
        object_frames = []
        object_scores = []
        frames = []
        movement_history = []
        max_history = 7
        flow_history = []
        max_flow_history = 5
        base_threshold = 0.06
        adaptive_threshold = base_threshold
        
        frame_count = 0
        processed_frames = 0
        
        # Show initial frame
        if enable_live_viz and live_viz_placeholder is not None:
            annotated_frame = prev_frame.copy()
            cv2.putText(annotated_frame, "Initializing Lucas-Kanade...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Frame: 0/{max_frames}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Points: {len(p0) if p0 is not None else 0}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            live_viz_placeholder.image(rgb_frame, caption="Frame 0 - Initializing", use_container_width=True, width=350)
            
        
        while True:
            if frame_count > max_frames:
                break
            
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue
            
            curr_gray = to_grayscale(frame)
            
            if enable_live_viz and live_viz_placeholder is not None:
                annotated_frame = frame.copy()
                
                cv2.putText(annotated_frame, f"Frame: {frame_count}/{max_frames}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f"Processing...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f"Progress: {int((frame_count/max_frames)*100)}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                live_viz_placeholder.image(rgb_frame, caption=f"Frame {frame_count} - Processing", use_container_width=True, width=350)
                
            
            if p0 is None or len(p0) < self.max_corners // 3:
                p0 = cv2.goodFeaturesToTrack(curr_gray, mask=None, **feature_params)
                if p0 is None:
                    if enable_live_viz and live_viz_placeholder is not None:
                        annotated_frame = frame.copy()
                        cv2.putText(annotated_frame, "No tracking points found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(annotated_frame, f"Frame: {frame_count}/{max_frames}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(annotated_frame, "Reinitializing...", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        live_viz_placeholder.image(rgb_frame, caption=f"Frame {frame_count} - No Points", use_container_width=True, width=350)
                        
                    
                    frames.append(frame)
                    frame_count += 1
                    continue
            
            p1, st_flow, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **lk_params)
            
            if p1 is not None:
                good_new = p1[st_flow == 1]
                good_old = p0[st_flow == 1]
                
                if len(good_new) > 0 and len(good_old) > 0:
                    flow_vectors = good_new - good_old
                    flow_magnitudes = np.sqrt(flow_vectors[:, 0]**2 + flow_vectors[:, 1]**2)
                    
                    mean_flow = np.mean(flow_magnitudes)
                    std_flow = np.std(flow_magnitudes)
                    max_flow = np.max(flow_magnitudes)
                    median_flow = np.median(flow_magnitudes)
                    q75_flow = np.percentile(flow_magnitudes, 75)
                    q25_flow = np.percentile(flow_magnitudes, 25)
                    
                    flow_angles = np.arctan2(flow_vectors[:, 1], flow_vectors[:, 0])
                    angle_consistency = 1.0 - (std_flow / (mean_flow + 1e-6))
                    
                    flow_iqr = q75_flow - q25_flow
                    flow_cv = std_flow / (mean_flow + 1e-6)
                    
                    flow_history.append(mean_flow)
                    if len(flow_history) > max_flow_history:
                        flow_history.pop(0)
                    
                    if len(flow_history) >= 3:
                        recent_avg_flow = np.mean(flow_history[-3:])
                        adaptive_threshold = max(base_threshold, recent_avg_flow * 0.3)
                    
                    criteria_1 = mean_flow > adaptive_threshold
                    criteria_2 = len(good_new) > 1
                    criteria_3 = max_flow > mean_flow * 1.1
                    criteria_4 = median_flow > adaptive_threshold * 0.4
                    criteria_5 = angle_consistency < 0.85
                    criteria_6 = flow_cv > 0.3
                    criteria_7 = flow_iqr > mean_flow * 0.5
                    
                    score = sum([criteria_1, criteria_2, criteria_3, criteria_4, criteria_5, criteria_6, criteria_7])
                    
                    is_object_movement = score >= 6
                    confidence = min(1.0, score / 7.0) if is_object_movement else 0.5
                    
                    if is_object_movement:
                        movement_history.append(mean_flow * confidence)
                        
                        if len(movement_history) > max_history:
                            movement_history.pop(0)
                        
                        sustained_movement = len(movement_history) >= 3 and np.mean(movement_history[-3:]) > adaptive_threshold * 0.8
                        
                        if sustained_movement:
                            is_object_movement = True
                            confidence *= 1.2
                        else:
                            is_object_movement = False
                    
                    movement_score = (mean_flow * 2.0 + max_flow * 0.5 + std_flow * 1.5) * (confidence if is_object_movement else 0.5)
                    
                    if is_object_movement:
                        object_frames.append(frame_count)
                        object_scores.append(movement_score)
                    
                    if enable_live_viz and live_viz_placeholder is not None:
                        annotated_frame = frame.copy()
                        
                        for new, old in zip(good_new, good_old):
                            a, b = new.ravel()
                            c, d = old.ravel()
                            
                            x1, y1 = int(c), int(d)
                            x2, y2 = int(a), int(b)
                            
                            h, w = frame.shape[:2]
                            if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                                flow_mag = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                                
                                if is_object_movement:
                                    if flow_mag > 10:
                                        color = (0, 255, 0)
                                        thickness = 4
                                    elif flow_mag > 6:
                                        color = (0, 220, 0)
                                        thickness = 3
                                    elif flow_mag > 3:
                                        color = (0, 180, 0)
                                        thickness = 2
                                    else:
                                        color = (0, 140, 0)
                                        thickness = 1
                                else:
                                    if flow_mag > 4:
                                        color = (0, 0, 255)
                                        thickness = 2
                                    elif flow_mag > 2:
                                        color = (0, 0, 200)
                                        thickness = 1
                                    else:
                                        color = (100, 100, 100)
                                        thickness = 1
                                
                                cv2.line(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                                cv2.circle(annotated_frame, (x1, y1), 2, (255, 0, 0), -1)
                                cv2.circle(annotated_frame, (x2, y2), 2, (0, 255, 255), -1)
                        
                        text = f"Movement: {'Object' if is_object_movement else 'Camera/None'}"
                        cv2.putText(annotated_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(annotated_frame, f"Points: {len(good_new)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(annotated_frame, f"Mean Flow: {mean_flow:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(annotated_frame, f"Confidence: {score}/7", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(annotated_frame, f"Frame: {frame_count}/{max_frames}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(annotated_frame, f"Progress: {int((frame_count/max_frames)*100)}%", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        live_viz_placeholder.image(rgb_frame, caption=f"Frame {frame_count} - Lucas-Kanade Motion Detection", use_container_width=True, width=350)
                        
                    
                    frames.append(frame)
                    p0 = good_new.reshape(-1, 1, 2)
                else:
                    # No good points found, still show frame
                    if enable_live_viz and live_viz_placeholder is not None:
                        annotated_frame = frame.copy()
                        cv2.putText(annotated_frame, "No tracking points found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(annotated_frame, f"Frame: {frame_count}/{max_frames}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(annotated_frame, f"Progress: {int((frame_count/max_frames)*100)}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        live_viz_placeholder.image(rgb_frame, caption=f"Frame {frame_count} - No Points", use_container_width=True, width=350)
                        
                    
                    frames.append(frame)
                    p0 = None
            else:

                if enable_live_viz and live_viz_placeholder is not None:
                    annotated_frame = frame.copy()
                    cv2.putText(annotated_frame, "No optical flow calculated", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(annotated_frame, f"Frame: {frame_count}/{max_frames}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(annotated_frame, f"Progress: {int((frame_count/max_frames)*100)}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    live_viz_placeholder.image(rgb_frame, caption=f"Frame {frame_count} - No Flow", use_container_width=True, width=350)
                    
                
                frames.append(frame)
                p0 = None
            
            prev_gray = curr_gray
            frame_count += 1
            processed_frames += 1
        
        cap.release()
        
        return create_result_dict(object_frames, object_scores, frames, frame_count, result_type="object")


class LucasKanadeDetector:

    def __init__(self, max_corners=100, quality_level=0.3, min_distance=7, block_size=7, win_size=(15, 15), max_level=2):
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.block_size = block_size
        self.win_size = win_size
        self.max_level = max_level
        self.prev_gray = None
        self.prev_points = None
        self.color = np.random.randint(0, 255, (max_corners, 3))
        self.mask = None
    
    def detect_lucas_kanade(self, frame):

        if self.prev_gray is None:
            self.prev_gray = to_grayscale(frame)
            self.prev_points = cv2.goodFeaturesToTrack(
                self.prev_gray, 
                mask=None,
                maxCorners=self.max_corners,
                qualityLevel=self.quality_level,
                minDistance=self.min_distance,
                blockSize=self.block_size
            )
            self.mask = np.zeros_like(frame)
            return False, 0.0, {"error": "First frame"}, frame
        
        gray = to_grayscale(frame)
        
        if self.prev_points is None or len(self.prev_points) == 0:
            self.prev_gray = gray
            self.prev_points = cv2.goodFeaturesToTrack(
                gray, 
                mask=None,
                maxCorners=self.max_corners,
                qualityLevel=self.quality_level,
                minDistance=self.min_distance,
                blockSize=self.block_size
            )
            return False, 0.0, {"error": "No points to track"}, frame
        
        lk_params = dict(
            winSize=self.win_size,
            maxLevel=self.max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_points, None, **lk_params
        )
        
        if new_points is None:
            self.prev_gray = gray
            self.prev_points = cv2.goodFeaturesToTrack(
                gray, 
                mask=None,
                maxCorners=self.max_corners,
                qualityLevel=self.quality_level,
                minDistance=self.min_distance,
                blockSize=self.block_size
            )
            return False, 0.0, {"error": "Flow calculation failed"}, frame
        
        good_new = new_points[status == 1]
        good_old = self.prev_points[status == 1]
        
        if len(good_new) > 0 and len(good_old) > 0:
            distances = np.sqrt(np.sum((good_new - good_old) ** 2, axis=1))
            movement_score = np.mean(distances)
            has_movement = movement_score > 1.0  
        else:
            movement_score = 0.0
            has_movement = False
        
        
        result_frame = frame.copy()
        
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            self.mask = cv2.line(self.mask, (int(a), int(b)), (int(c), int(d)), 
                               self.color[i % len(self.color)].tolist(), 2)
            result_frame = cv2.circle(result_frame, (int(a), int(b)), 5, 
                                    self.color[i % len(self.color)].tolist(), -1)
        
        result_frame = cv2.add(result_frame, self.mask)
        
        self.prev_gray = gray
        self.prev_points = good_new.reshape(-1, 1, 2)
        
        details = {
            "movement_score": movement_score,
            "tracked_points": len(good_new),
            "method": "lucas_kanade",
            "max_displacement": np.max(distances) if len(good_new) > 0 else 0.0,
            "mean_displacement": np.mean(distances) if len(good_new) > 0 else 0.0
        }
        
        return has_movement, movement_score, details, result_frame

    def detect_objects(self, video_path, max_frames=120):
        """Detect object movement in video using Lucas-Kanade optical flow"""
        cap = open_video(video_path)
        if cap is None:
            return {'error': 'Could not read video'}
        
        cap, prev_frame = read_first_frame(cap)
        if prev_frame is None:
            return {'error': 'Could not read video'}
        
        prev_gray = to_grayscale(prev_frame)
        
        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        p0 = cv2.goodFeaturesToTrack(
            prev_gray, 
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=self.block_size
        )
        
        if p0 is None:
            return {'error': 'No corners found in first frame'}
        
        object_frames = []
        object_scores = []
        details_list = []
        annotated_frames = []
        frames = [prev_frame]
        
        frame_idx = 1
        
        while True:
            if frame_idx > max_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break
                
            curr_gray = to_grayscale(frame)
            
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **lk_params)
            
            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]
                
                if len(good_new) > 0 and len(good_old) > 0:
                    flow_vectors = good_new - good_old
                    flow_magnitudes = np.sqrt(flow_vectors[:, 0]**2 + flow_vectors[:, 1]**2)
                    
                    mean_flow = np.mean(flow_magnitudes)
                    max_flow = np.max(flow_magnitudes)
                    motion_pixels = len(good_new)

                    movement_score = mean_flow / 10.0
                    is_movement = movement_score > 0.1  
                    
                    if is_movement:
                        object_frames.append(frame_idx)
                        object_scores.append(movement_score)
                        
                        details = {
                            'num_objects': len(good_new),
                            'motion_pixels': motion_pixels,
                            'flow_magnitude_mean': mean_flow,
                            'flow_magnitude_max': max_flow,
                            'movement_score': movement_score
                        }
                        details_list.append(details)
                    
                    annotated_frame = frame.copy()
                    for i, (new, old) in enumerate(zip(good_new, good_old)):
                        a, b = new.ravel()
                        c, d = old.ravel()
                        annotated_frame = cv2.line(annotated_frame, (int(c), int(d)), (int(a), int(b)), (0, 255, 0), 2)
                        annotated_frame = cv2.circle(annotated_frame, (int(a), int(b)), 3, (0, 0, 255), -1)
                    
                    annotated_frames.append(annotated_frame)
                else:
                    annotated_frames.append(frame)
                
                prev_gray = curr_gray.copy()
                p0 = good_new.reshape(-1, 1, 2)
            else:
                annotated_frames.append(frame)
            
            frames.append(frame)
            frame_idx += 1
        
        cap.release()
        
        return create_result_dict(object_frames, object_scores, frames, frame_idx - 1, details_list, annotated_frames, 'Lucas-Kanade Optical Flow')


class ObjectMovementDetector:
    
    def __init__(self, threshold=0.5, flow_threshold=0.5):
        self.threshold = threshold
        self.flow_threshold = flow_threshold
        self.prev_frame = None
        self.prev_gray = None
    
    def detect_objects(self, frame: np.ndarray) -> tuple[bool, float, dict, np.ndarray]:

        if self.prev_gray is None:
            self.prev_gray = to_grayscale(frame)
            return False, 0.0, {"error": "First frame"}, frame
        
        gray = to_grayscale(frame)
        
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        motion_mask = magnitude > self.flow_threshold
        
        movement_score = np.mean(magnitude)
        has_movement = movement_score > self.threshold
        
        result_frame = frame.copy()
        result_frame[motion_mask] = [0, 0, 255]  

        step = 16
        for y in range(0, frame.shape[0], step):
            for x in range(0, frame.shape[1], step):
                if motion_mask[y, x]:
                    fx, fy = flow[y, x]
                    cv2.arrowedLine(result_frame, (x, y), 
                                   (int(x+fx), int(y+fy)), (0, 255, 0), 1)
        
        motion_mask_uint8 = (motion_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(motion_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(result_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(result_frame, f"Object", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        self.prev_gray = gray
        
        details = {
            "movement_score": movement_score,
            "motion_pixels": np.sum(motion_mask),
            "num_objects": len(contours),
            "method": "farneback_optical_flow",
            "flow_magnitude_mean": np.mean(magnitude),
            "flow_magnitude_max": np.max(magnitude)
        }
        
        return has_movement, movement_score, details, result_frame
    
    def detect_objects_video(self, video_path, max_frames=120):
        cap = open_video(video_path)
        if cap is None:
            return {'error': 'Could not read video'}
        
        cap, prev_frame = read_first_frame(cap)
        if prev_frame is None:
            return {'error': 'Could not read video'}
        
        prev_gray = to_grayscale(prev_frame)
        
        object_frames = []
        object_scores = []
        details_list = []
        annotated_frames = []
        frames = [prev_frame]
        
        frame_idx = 1
        
        while True:
            if frame_idx > max_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break
                
            gray = to_grayscale(frame)
            
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_mask = magnitude > self.flow_threshold
            
            movement_score = np.mean(magnitude)
            has_movement = movement_score > self.threshold
            
            if has_movement:
                object_frames.append(frame_idx)
                object_scores.append(movement_score)
                
                details = {
                    'motion_pixels': np.sum(motion_mask),
                    'flow_magnitude_mean': np.mean(magnitude),
                    'flow_magnitude_max': np.max(magnitude),
                    'movement_score': movement_score
                }
                details_list.append(details)
            
            annotated_frame = frame.copy()
            annotated_frame[motion_mask] = [0, 0, 255]  # Red for motion
            
            step = 16
            for y in range(0, frame.shape[0], step):
                for x in range(0, frame.shape[1], step):
                    if motion_mask[y, x]:
                        fx, fy = flow[y, x]
                        cv2.arrowedLine(annotated_frame, (x, y), 
                                       (int(x+fx), int(y+fy)), (0, 255, 0), 1)
            
            annotated_frames.append(annotated_frame)
            frames.append(frame)
            prev_gray = gray.copy()
            frame_idx += 1
        
        cap.release()
        
        return create_result_dict(object_frames, object_scores, frames, frame_idx - 1, details_list, annotated_frames, 'Farneback Optical Flow')


class FarnebackAnalyzer:
    """Farneback optical flow analyzer for object movement detection"""
    
    def __init__(self, object_threshold=0.1, flow_threshold=0.5):
        self.object_threshold = object_threshold
        self.flow_threshold = flow_threshold
        
    def analyze_video(self, video_path, max_frames=120, frame_skip=1, enable_live_viz=False, live_viz_placeholder=None):
        cap = open_video(video_path)
        if cap is None:
            return None
        
        cap, prev_frame = read_first_frame(cap)
        if prev_frame is None:
            return None
        
        prev_gray = to_grayscale(prev_frame)
        
        object_frames = []
        object_scores = []
        frames = []
        
        frame_count = 0
        processed_frames = 0
        
        while True:
            if frame_count > max_frames:
                break
            
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue
            
            gray = to_grayscale(frame)
            
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mean_magnitude = np.mean(magnitude)
            std_magnitude = np.std(magnitude)
            max_magnitude = np.max(magnitude)
            motion_mask = magnitude > self.flow_threshold * 0.8
            motion_density = np.sum(motion_mask) / motion_mask.size
            motion_variance = std_magnitude / (mean_magnitude + 1e-6)
            
            angle = np.arctan2(flow[..., 1], flow[..., 0])
            angle_hist, _ = np.histogram(angle[motion_mask], bins=8, range=(-np.pi, np.pi))
            angle_consistency = np.max(angle_hist) / (np.sum(angle_hist) + 1e-6)
            
            motion_uniformity = 1.0 - (std_magnitude / (mean_magnitude + 1e-6))
            
            motion_centers = np.where(motion_mask)
            if len(motion_centers[0]) > 0:
                center_y, center_x = np.mean(motion_centers[0]), np.mean(motion_centers[1])
                frame_center_y, frame_center_x = flow.shape[0] // 2, flow.shape[1] // 2
                motion_center_distance = np.sqrt((center_y - frame_center_y)**2 + (center_x - frame_center_x)**2)
                motion_center_ratio = motion_center_distance / np.sqrt(flow.shape[0]**2 + flow.shape[1]**2)
            else:
                motion_center_ratio = 0
            
            is_object_movement = (
                mean_magnitude > self.object_threshold * 0.3 and
                motion_density > 0.003 and motion_density < 0.4 and
                np.sum(motion_mask) > 300 and
                angle_consistency < 0.6 and
                motion_uniformity < 0.7 and
                motion_center_ratio > 0.1
            )
            
            movement_score = mean_magnitude / 2.0
            
            if is_object_movement:
                object_frames.append(frame_count)
                object_scores.append(movement_score)
            
            if enable_live_viz and live_viz_placeholder is not None:
                annotated_frame = frame.copy()
                
                if is_object_movement:
                    annotated_frame[motion_mask] = [0, 255, 0]  # Green for object movement
                else:
                    annotated_frame[motion_mask] = [0, 0, 255]  # Red for camera movement
                
                step = 16
                for y in range(0, frame.shape[0], step):
                    for x in range(0, frame.shape[1], step):
                        if motion_mask[y, x]:
                            fx, fy = flow[y, x]
                            flow_mag = np.sqrt(fx**2 + fy**2)
                            
                            if is_object_movement:
                                if flow_mag > 5:
                                    color = (255, 255, 0)  # Yellow
                                    thickness = 2
                                else:
                                    color = (0, 255, 255)  # Cyan
                                    thickness = 1
                            else:
                                if flow_mag > 3:
                                    color = (255, 0, 255)  # Magenta
                                    thickness = 2
                                else:
                                    color = (128, 128, 128)  # Gray
                                    thickness = 1
                            
                            end_x = int(x + fx * 2)
                            end_y = int(y + fy * 2)
                            
                            h, w = frame.shape[:2]
                            if 0 <= end_x < w and 0 <= end_y < h:
                                cv2.arrowedLine(annotated_frame, (x, y), (end_x, end_y), color, thickness, tipLength=0.3)
                
                text = f"Movement: {'Object' if is_object_movement else 'Camera/None'}"
                cv2.putText(annotated_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f"Motion Pixels: {np.sum(motion_mask)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f"Mean Magnitude: {mean_magnitude:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                live_viz_placeholder.image(rgb_frame, caption=f"Frame {frame_count} - Farneback Motion Detection", use_container_width=True, width=350)
            
            frames.append(frame)
            prev_gray = gray
            frame_count += 1
            processed_frames += 1
        
        cap.release()
        
        return create_result_dict(object_frames, object_scores, frames, frame_count, result_type="object")




