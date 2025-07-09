import os
import csv
from src.detect import detect_players, load_config
from src.extract_features import extract_features_from_video
from src.reid_matcher import match_players

def main():
    # Load config
    config = load_config()

    # Step 1: Run detections on both videos
    detect_players(
        video_path=config["video_paths"]["broadcast"],
        model_path=config["model_path"],
        output_dir=config["output_dir"],
        conf_thresh=config["confidence_threshold"],
        device=config["device"]
    )

    detect_players(
        video_path=config["video_paths"]["tacticam"],
        model_path=config["model_path"],
        output_dir=config["output_dir"],
        conf_thresh=config["confidence_threshold"],
        device=config["device"]
    )

    # Step 2: Extract features from detection results
    broadcast_txt = os.path.join(config["output_dir"], "broadcast_detections.txt")
    tacticam_txt = os.path.join(config["output_dir"], "tacticam_detections.txt")

    broadcast_feats = extract_features_from_video(config["video_paths"]["broadcast"], broadcast_txt)
    tacticam_feats = extract_features_from_video(config["video_paths"]["tacticam"], tacticam_txt)

    # Step 3: Safety check
    if not broadcast_feats or not tacticam_feats:
        print("❌ No features extracted — check if the videos are valid and contain players.")
        return

    # Step 4: Match players
    matches = match_players(broadcast_feats, tacticam_feats)

    # Step 5: Print match results
    print("\n[✓] Matched Players (Broadcast ↔ Tacticam):\n")
    for b_id, t_id, dist in matches:
        print(f"{b_id} ↔ {t_id} | Distance: {dist:.3f}")

    # Step 6: Save match results to CSV
    match_output_file = os.path.join(config["output_dir"], "player_matches.csv")
    with open(match_output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["broadcast_id", "tacticam_id", "cosine_distance"])

        for b_id, t_id, dist in matches:
            writer.writerow([b_id, t_id, f"{dist:.4f}"])

    print(f"\n[✓] Match results saved to: {match_output_file}")

if __name__ == "__main__":
    main()
