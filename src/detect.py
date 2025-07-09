import cv2
import yaml
from ultralytics import YOLO
import os

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def detect_players(video_path, model_path, output_dir, conf_thresh=0.4, device="cpu"):
    model = YOLO(model_path)
    model.to(device)

    cap = cv2.VideoCapture(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_file = os.path.join(output_dir, f"{video_name}_detections.txt")

    frame_idx = 0
    with open(output_file, "w") as f:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, conf=conf_thresh, verbose=False)[0]

            for box in results.boxes:
                cls_id = int(box.cls)
                if cls_id != 0:  # 0 = player, 1 = ball (optional: skip ball)
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                f.write(f"{frame_idx},{x1},{y1},{x2},{y2},{conf}\n")

            frame_idx += 1

    cap.release()
    print(f"[✓] Detections saved to: {output_file}")
