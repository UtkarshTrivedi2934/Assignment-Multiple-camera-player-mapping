#### 🏆 Player Re-Identification in Sports Footage
🔍 Cross-Camera Player Mapping

### 🎯 Objective
Build a system that maps each player across two video feeds (broadcast.mp4 and tacticam.mp4) from different angles of the same match. Each player should retain a consistent ID in both views.

### 🧠 Approach & Architecture

## 1. Player Detection
Used a Pre-trained YOLOv11 model fine-tuned for player and ball detection.

Run on both input videos to extract bounding boxes and confidence scores.

## 2. Feature Extraction
For each detected player, an HSV color histogram is extracted from their cropped region.

Histograms are normalized and flattened to a 512-dimensional vector.

## 3. Re-Identification (Matching)
Cosine similarity matrix is built between all players from both feeds.

Hungarian Algorithm is applied to find the best one-to-one player mapping.

Results are saved to a CSV file for clear review.

### 🧱 Project Structure

```bash
player_reidentification/
├── config.yaml                  # Central config for paths
├── run_reid.py                  # 🚀 Main pipeline runner
├── requirements.txt             # Required Python packages
├── README.md                    # 📘 This file
├── .gitignore                   # 🚫 Ignore list
│
├── models/                      # 🔍 Detection model (add yolov8_player_ball.pt manually)
│   └── model file link    # [ignored in repo]
│
├── videos/                      # 🎥 Input videos (add manually)
│   ├── broadcast.mp4
│   └── tacticam.mp4
│
├── outputs/                     # 📤 Generated detections and final matches
│   ├── broadcast_detections.txt
│   ├── tacticam_detections.txt
│   └── player_matches.csv
│
└── src/                         # 🧠 Source code
    ├── detect.py                # Player detection using YOLO
    ├── extract_features.py      # HSV histogram extractor
    ├── reid_matcher.py          # Re-ID matching via Hungarian
    └── utils.py                 # Helpers & config loader
```

## 📁 Download Required Files

> 📂 After downloading:
> - Place `best.pt` in the `models/` directory

## ⚙️ Setup & Installation

### ✅ Prerequisites

- Python 3.9 or higher
- pip or virtualenv

### 📦 Install Dependencies

``` pip install -r requirements.txt ```

### 🚀 Run the Pipeline

```` python run_reid.py ````

### This script will:

Detect players in both videos

Extract HSV histograms

Match players using cosine similarity

Save results to outputs/player_matches.csv

### 📊 Example Output (player_matches.csv)
Broadcast ID	Tacticam ID	Cosine Distance
24_0	17_2	0.1782
24_1	17_3	0.3025
25_0	18_1	0.2844

### 🧪 Techniques Explored
Tried both Euclidean and cosine similarity (cosine performed better for color features)

Considered using centroids for matching, but discarded due to moving camera

Investigated FastReID and Jersey OCR for future improvements

### 🧗 Challenges Encountered
Missing / blurred detections: Required filtering by confidence

Color similarity limitations: HSV histograms helped but are not enough for visually identical jerseys

Re-ID embeddings: Not used to keep the pipeline lightweight and interpretable
