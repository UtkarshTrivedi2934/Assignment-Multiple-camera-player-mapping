import cv2
import os
import numpy as np
import yaml

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def compute_color_histogram(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def extract_features_from_video(video_path, detections_path):
    cap = cv2.VideoCapture(video_path)
    features = {}
    current_frame = -1

    # Group detections by frame
    detections = {}
    with open(detections_path, "r") as f:
        for line in f:
            frame_idx, x1, y1, x2, y2, conf = map(float, line.strip().split(","))
            detections.setdefault(int(frame_idx), []).append((int(x1), int(y1), int(x2), int(y2)))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current_frame += 1

        if current_frame in detections:
            for idx, (x1, y1, x2, y2) in enumerate(detections[current_frame]):
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                hist = compute_color_histogram(crop)
                key = f"{current_frame}_{idx}"
                features[key] = hist

    cap.release()
    return features  # Dict: { "frame_idx_idx": hist_vector }
