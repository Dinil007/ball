import os
import sys
import csv
import math
import time
import cv2

# Ensure project root is in Python path for standalone script execution
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from camera.webcam import Webcam
from hand.hand_detector import HandDetector
from hand.landmark_smoother import LandmarkSmoother
from hand.hand_orientation import HandOrientation
from ball.ball_detector import BallDetector
from tracking.hand_ball_association import HandBallAssociation
from features.feature_extractor import FeatureExtractor


CSV_COLUMNS = [
    "hand_side",
    "orientation",
    "index_angle",
    "middle_angle",
    "ring_angle",
    "thumb_index_distance",
    "ball_thumb_distance",
    "ball_index_distance",
    "ball_middle_distance",
    "ball_ring_distance",
    "wrist_angle",
    "wrist_rotation_angle",
    "thumb_angle",
    "index_curl",
    "middle_curl",
    "ring_curl",
    "pinky_curl",
    "index_middle_distance",
    "middle_ring_distance",
    "ring_pinky_distance",
    "ball_index_mcp_distance",
    "ball_middle_mcp_distance",
    "ball_ring_mcp_distance",
    "ball_coverage_ratio",
    "label"
]

REQUIRED_FEATURE_KEYS = [col for col in CSV_COLUMNS if col != "label"]


def safe_float(val, default=0.0):
    """
    Safely cast any value to float, handling None, NaN, inf, or formatting errors.
    """
    if val is None:
        return default
    try:
        f = float(val)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except (ValueError, TypeError):
        return default


def count_existing_samples(csv_path, label=None):
    """Count total rows in CSV (and optionally matching label)."""
    if not os.path.exists(csv_path):
        return 0, 0
    total = 0
    label_count = 0
    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if label and row.get("label") == label:
                label_count += 1
    return total, label_count


def validate_features(features):
    """
    Validate that all 24 feature keys are present, not None, and not NaN.
    Returns (is_valid: bool, valid_count: int, missing_keys: list).
    """
    if not features:
        return False, 0, REQUIRED_FEATURE_KEYS

    missing = []
    for k in REQUIRED_FEATURE_KEYS:
        v = features.get(k)
        if v is None:
            missing.append(k)
        elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            missing.append(k)

    valid_count = len(REQUIRED_FEATURE_KEYS) - len(missing)
    return len(missing) == 0, valid_count, missing


def save_sample_to_csv(csv_path, features, label):
    """Append a valid 24-feature dictionary and label to CSV."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0

    row = {col: features.get(col) for col in REQUIRED_FEATURE_KEYS}
    row["label"] = label

    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    print("=" * 60)
    print("    CricketGrip AI - Data Collection (24-Feature Schema)")
    print("=" * 60)

    # Prompt user for grip label
    while True:
        try:
            label = input("\nEnter grip label (e.g. off_spin_grip, knuckle_ball): ").strip()
        except EOFError:
            label = "unlabeled"
        if label:
            break
        print("Label cannot be empty. Please enter a valid grip name.")

    csv_path = os.path.join(PROJECT_ROOT, "data", "features.csv")
    total_samples, label_samples = count_existing_samples(csv_path, label)

    print(f"\n[INFO] Target label: '{label}'")
    print(f"[INFO] Existing samples in dataset: {total_samples} (for '{label}': {label_samples})")
    print("\nControls:")
    print("  [SPACE] -> Capture 1 validated sample")
    print("  [Q]     -> Quit data collection\n")

    # Initialize modules
    webcam            = Webcam(0)
    if not webcam.is_opened():
        print("ERROR: Could not open webcam.")
        return

    ball_detector     = BallDetector()
    hand_detector     = HandDetector()
    smoothers         = {"LEFT": LandmarkSmoother(), "RIGHT": LandmarkSmoother()}
    orientation_det   = HandOrientation()
    association       = HandBallAssociation()
    feature_extractor = FeatureExtractor()

    feedback_msg = ""
    feedback_time = 0.0
    feedback_color = (0, 255, 0)

    while True:
        success, frame = webcam.read()
        if not success:
            print("ERROR: Failed to read frame from webcam.")
            break

        frame_height, frame_width = frame.shape[:2]

        # ── 1. Ball Detection ────────────────────────────────────────────────
        ball = ball_detector.detect(frame)

        # ── 2. Hand Detection ────────────────────────────────────────────────
        hands = hand_detector.detect(frame)

        # ── 3. Landmark Smoothing (One Euro Filter per hand) ─────────────────
        if hands:
            active_sides = set()
            for hand in hands:
                side = hand.get("side", "LEFT")
                active_sides.add(side)
                if side not in smoothers:
                    smoothers[side] = LandmarkSmoother()
                hand["landmarks"] = smoothers[side].smooth(hand["landmarks"])
            for side in list(smoothers.keys()):
                if side not in active_sides:
                    smoothers[side].reset()
        else:
            for s in smoothers.values():
                s.reset()

        # ── 4. Orientation Calculation ────────────────────────────────────────
        orientations = []
        for hand in hands:
            ori_res = orientation_det.calculate(hand)
            orientations.append(ori_res["orientation"])

        # ── 5. Hand-Ball Association ──────────────────────────────────────────
        assoc_result = association.calculate(ball, hands, frame_width, frame_height)
        holder = assoc_result["holder"]

        # ── 6. Feature Extraction ─────────────────────────────────────────────
        features = None
        primary_orientation = "UNCERTAIN"
        active_hand = None

        if hands:
            if holder in ("LEFT", "RIGHT"):
                for idx, h in enumerate(hands):
                    if h.get("side") == holder:
                        active_hand = h
                        primary_orientation = orientations[idx] if idx < len(orientations) else "UNCERTAIN"
                        break
            else:
                active_hand = hands[0]
                primary_orientation = orientations[0] if orientations else "UNCERTAIN"

            if active_hand:
                features = feature_extractor.extract(
                    active_hand,
                    ball,
                    primary_orientation,
                    frame_width,
                    frame_height
                )

        # Validate all 24 features
        is_valid_features, valid_feat_count, missing_feats = validate_features(features)

        # ── Visual Overlays ───────────────────────────────────────────────────
        frame = ball_detector.draw(frame, ball)
        frame = hand_detector.draw_landmarks(frame)

        # Left Info Overlay
        y = 28
        lh = 26

        cv2.putText(frame, f"Label: {label}", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 0), 2)
        y += lh

        cv2.putText(frame, f"Hands: {len(hands)}", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0) if hands else (0, 0, 255), 2)
        y += lh

        ori_color = (0, 255, 255) if primary_orientation == "PALM" else \
                    (0, 165, 255) if primary_orientation == "BACK" else (150, 150, 150)
        cv2.putText(frame, f"Orientation: {primary_orientation}", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, ori_color, 2)
        y += lh

        ball_str = "YES" if ball else "NO"
        ball_col = (0, 255, 0) if ball else (0, 0, 255)
        cv2.putText(frame, f"Ball: {ball_str} | Holder: {holder}", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, ball_col, 2)
        y += lh

        # Feature Validity Indicator
        if is_valid_features:
            cv2.putText(frame, "Features: 24/24", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 0), 2)
        else:
            cv2.putText(frame, f"Features: INVALID ({valid_feat_count}/24)", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 255), 2)
        y += lh

        cv2.putText(frame, f"Samples for '{label}': {label_samples}", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

        # Bottom Instructions & Feedback
        cv2.putText(frame, "[SPACE]: Capture Sample  |  [Q]: Quit",
                    (12, frame_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (220, 220, 220), 1)

        if time.time() - feedback_time < 1.5:
            cv2.putText(frame, feedback_msg, (12, frame_height - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.68, feedback_color, 2)

        # Right Telemetry with Safe Float Formatting
        if features:
            fx = frame_width - 290
            fy = 26
            cv2.putText(frame, "-- KINEMATICS --", (fx, fy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 0), 1)
            fy += 22

            lines = [
                f"Wrist rot : {safe_float(features.get('wrist_rotation_angle')):.1f}",
                f"Thumb curl: {safe_float(features.get('thumb_angle')):.1f}",
                f"Index curl: {safe_float(features.get('index_curl')):.1f}",
                f"Mid curl  : {safe_float(features.get('middle_curl')):.1f}",
                f"Ring curl : {safe_float(features.get('ring_curl')):.1f}",
                f"Pinky curl: {safe_float(features.get('pinky_curl')):.1f}",
                f"Idx-Mid   : {safe_float(features.get('index_middle_distance')):.3f}",
                f"Coverage  : {safe_float(features.get('ball_coverage_ratio')):.3f}",
            ]
            for line in lines:
                cv2.putText(frame, line, (fx, fy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 0), 1)
                fy += 18

        cv2.imshow("CricketGrip AI - Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        # ── Capture Sample [SPACE] ────────────────────────────────────────────
        if key == 32:  # SPACE key
            if not hands:
                feedback_msg = "CANNOT SAVE: No hand detected!"
                feedback_color = (0, 0, 255)
                feedback_time = time.time()
                print(f"[REJECTED] {feedback_msg}")
            elif not ball:
                feedback_msg = "CANNOT SAVE: Ball not detected!"
                feedback_color = (0, 0, 255)
                feedback_time = time.time()
                print(f"[REJECTED] {feedback_msg}")
            elif primary_orientation != "BACK":
                feedback_msg = f"CANNOT SAVE: Orientation must be BACK (Current: {primary_orientation})!"
                feedback_color = (0, 0, 255)
                feedback_time = time.time()
                print(f"[REJECTED] {feedback_msg}")
            elif not is_valid_features:
                feedback_msg = f"CANNOT SAVE: Missing features ({valid_feat_count}/24)!"
                feedback_color = (0, 0, 255)
                feedback_time = time.time()
                print(f"[REJECTED] {feedback_msg} Missing: {missing_feats}")
            else:
                save_sample_to_csv(csv_path, features, label)
                label_samples += 1
                total_samples += 1
                feedback_msg = f"Saved! ({label_samples} samples)"
                feedback_color = (0, 255, 0)
                feedback_time = time.time()
                print(f"[SAVED] Label: '{label}' | Count: {label_samples} | Total: {total_samples}")

        elif key in (ord("q"), ord("Q"), 27):
            print(f"\nExiting data collection. Total samples collected for '{label}': {label_samples}")
            break

    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
