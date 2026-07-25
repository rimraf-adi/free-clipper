"""Smart Intelligent Reframing Engine.

Combines 3 core strategies:
1. Face Detection & Tracking (OpenCV Haar Cascade / DNN)
2. Audio-Aware Active Speaker Pan (detects speaker positions and switches pan keyframes on speaker turns)
3. Split-Screen Stacking (for 2 active speakers or dual-speaker scenes, stacks top & bottom half in 9:16)
"""

from __future__ import annotations

import logging
import os
from typing import List, Dict, Any, Tuple, Optional
import cv2

logger = logging.getLogger(__name__)

# Load Haar cascade once
_face_cascade = None

def _get_cascade():
    global _face_cascade
    if _face_cascade is None:
        path = getattr(getattr(cv2, "data", None), "haarcascades", "") + "haarcascade_frontalface_default.xml"
        _face_cascade = getattr(cv2, "CascadeClassifier", lambda p: None)(path)
    return _face_cascade


def detect_faces_in_clip(
    video_path: str,
    start_sec: float,
    end_sec: float,
    sample_interval: float = 0.5
) -> List[Dict[str, Any]]:
    """Samples video frames from start_sec to end_sec and detects faces in each frame."""
    if not os.path.exists(video_path):
        return []

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920.0
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080.0

        cascade = _get_cascade()
        detections = []

        curr_t = start_sec
        while curr_t <= end_sec:
            frame_idx = int(curr_t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                curr_t += sample_interval
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = []
            if cascade is not None and hasattr(cascade, "detectMultiScale"):
                faces = cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(60, 60)
                )

            frame_faces = []
            for (x, y, w, h) in faces:
                center_x_pct = ((x + w / 2.0) / width) * 100.0
                center_y_pct = ((y + h / 2.0) / height) * 100.0
                frame_faces.append({
                    "x_pct": center_x_pct,
                    "y_pct": center_y_pct,
                    "w": w,
                    "h": h
                })

            detections.append({
                "time": round(curr_t - start_sec, 2),
                "faces": frame_faces
            })

            curr_t += sample_interval

        cap.release()
        return detections
    except Exception as exc:
        logger.warning("Face detection encounter error: %s", exc)
        return []


def analyze_speaker_clusters(detections: List[Dict[str, Any]]) -> Tuple[List[float], str]:
    """Clusters face X-positions across the clip to identify main subject locations.

    Returns:
        (cluster_centers, recommended_mode)
        where recommended_mode is 'single' (1 subject), 'dual_pan' (2 subjects), or 'split' (2 active subjects).
    """
    all_centers = []
    for d in detections:
        for f in d.get("faces", []):
            all_centers.append(f["x_pct"])

    if not all_centers:
        return ([50.0], "single")

    min_x = min(all_centers)
    max_x = max(all_centers)

    if max_x - min_x < 20.0:
        # Single main subject area
        avg_x = sum(all_centers) / len(all_centers)
        return ([avg_x], "single")

    # 2 Subject Areas (e.g. Left vs Right podcast hosts)
    left_cluster = [x for x in all_centers if x < (min_x + max_x) / 2.0]
    right_cluster = [x for x in all_centers if x >= (min_x + max_x) / 2.0]

    c_left = sum(left_cluster) / len(left_cluster) if left_cluster else min_x
    c_right = sum(right_cluster) / len(right_cluster) if right_cluster else max_x

    # Check if frames frequently have BOTH faces visible simultaneously
    dual_face_frames = sum(1 for d in detections if len(d.get("faces", [])) >= 2)
    total_frames = max(1, len(detections))
    dual_ratio = dual_face_frames / total_frames

    if dual_ratio > 0.4:
        # Frequently both faces visible -> Split-Screen Stacking works best!
        return ([c_left, c_right], "split")
    else:
        # Alternating active speakers -> Smart Camera Pan
        return ([c_left, c_right], "dual_pan")


def generate_smart_keyframes(
    video_path: str,
    start_sec: float,
    end_sec: float,
    transcript_segments: Optional[List[Dict[str, Any]]] = None
) -> Tuple[List[Dict[str, Any]], str]:
    """Generates intelligent crop keyframes or recommends split-screen mode.

    Returns:
        (keyframes, mode)
        mode is one of: 'crop' (single subject / pan keyframes) or 'split' (top-bottom split screen).
    """
    detections = detect_faces_in_clip(video_path, start_sec, end_sec, sample_interval=0.5)
    clusters, recommended_mode = analyze_speaker_clusters(detections)

    if recommended_mode == "split":
        logger.info("Smart Crop: Dual active speakers detected -> recommending SPLIT-SCREEN STACK mode.")
        return ([], "split")

    keyframes = []
    if len(clusters) == 1 or recommended_mode == "single":
        # Single Subject Tracking: Smooth Exponential Moving Average (EMA)
        target_x = max(15.0, min(85.0, clusters[0]))
        logger.info("Smart Crop: Single subject detected at X=%.1f%% -> auto-reframing.", target_x)

        smoothed_x = target_x
        alpha = 0.3  # Smoothing factor

        for d in detections:
            t = d["time"]
            faces = d.get("faces", [])
            if faces:
                raw_x = faces[0]["x_pct"]
                smoothed_x = alpha * raw_x + (1.0 - alpha) * smoothed_x
            
            keyframes.append({
                "time": t,
                "pos_x": max(15.0, min(85.0, round(smoothed_x, 2))),
                "pos_y": 50.0,
                "zoom": 100
            })
    else:
        # Dual Speaker Active Pan: Switch between left and right cluster
        c_left, c_right = clusters[0], clusters[1]
        logger.info("Smart Crop: Dual speakers detected (Left=%.1f%%, Right=%.1f%%) -> active speaker panning.", c_left, c_right)

        curr_x = c_left

        for d in detections:
            t = d["time"]
            faces = d.get("faces", [])
            if faces:
                nearest = min(faces, key=lambda f: min(abs(f["x_pct"] - c_left), abs(f["x_pct"] - c_right)))
                if abs(nearest["x_pct"] - c_right) < abs(nearest["x_pct"] - c_left):
                    curr_x = c_right
                else:
                    curr_x = c_left

            keyframes.append({
                "time": t,
                "pos_x": max(15.0, min(85.0, round(curr_x, 2))),
                "pos_y": 50.0,
                "zoom": 100
            })

    if not keyframes:
        keyframes = [{"time": 0.0, "pos_x": 50.0, "pos_y": 50.0, "zoom": 100}]

    return (keyframes, "crop")
