import cv2
from collections import deque, Counter

from camera.webcam import Webcam
from ball.ball_detector import BallDetector
from hand.hand_detector import HandDetector
from hand.landmark_smoother import LandmarkSmoother
from hand.hand_orientation import HandOrientation
from tracking.hand_ball_association import HandBallAssociation
from features.feature_extractor import FeatureExtractor
from classifier.grip_classifier import GripClassifier, evaluate_dataset_samples


def get_smoothed_prediction(history_buffer):
    """
    Perform temporal smoothing over a 10-frame history buffer using majority voting
    and average confidence of the dominant class.
    """
    if not history_buffer:
        return {"grip": "NONE", "raw_grip": "NONE", "confidence": 0.0, "probabilities": {}}

    # Extract all non-NONE valid frame predictions
    valid_predictions = [p for p in history_buffer if p["grip"] != "NONE"]
    if not valid_predictions:
        return {"grip": "NONE", "raw_grip": "NONE", "confidence": 0.0, "probabilities": {}}

    # Majority voting over thresholded predictions
    grip_counts = Counter(p["grip"] for p in valid_predictions)
    most_common_grip, count = grip_counts.most_common(1)[0]

    # Calculate average confidence for the dominant grip
    relevant_confs = [p["confidence"] for p in valid_predictions if p["grip"] == most_common_grip]
    avg_conf = sum(relevant_confs) / len(relevant_confs) if relevant_confs else 0.0

    # Also retrieve most frequent raw prediction
    raw_counts = Counter(p.get("raw_grip", p["grip"]) for p in valid_predictions)
    most_common_raw, _ = raw_counts.most_common(1)[0]

    # Use the latest probability distribution
    latest_probs = valid_predictions[-1].get("probabilities", {})

    return {
        "grip": most_common_grip,
        "raw_grip": most_common_raw,
        "confidence": round(avg_conf, 2),
        "probabilities": latest_probs
    }


def main():
    # ── Run direct dataset sample evaluation first for baseline comparison ───
    evaluate_dataset_samples()

    webcam = Webcam(0)

    if not webcam.is_opened():
        print("ERROR: Could not open webcam.")
        return

    ball_detector     = BallDetector()
    hand_detector     = HandDetector()
    # Independent smoother per hand side to prevent filter-state corruption
    smoothers         = {"LEFT": LandmarkSmoother(), "RIGHT": LandmarkSmoother()}
    orientation_det   = HandOrientation()
    association       = HandBallAssociation()
    feature_extractor = FeatureExtractor()
    grip_classifier   = GripClassifier(confidence_threshold=0.60)

    # 10-frame buffer for temporal prediction stabilization
    prediction_history = deque(maxlen=10)

    print("=" * 60)
    print("    CricketGrip AI - Real-time Multi-Hand Engine")
    print("=" * 60)
    print(f"Model Feature Columns Expected: {len(grip_classifier.feature_columns)}")
    print(f"Confidence Threshold: {grip_classifier.confidence_threshold * 100:.0f}%")
    print("Webcam started. Press Q to quit.\n")

    frame_counter = 0

    while True:

        success, frame = webcam.read()

        if not success:
            print("ERROR: Frame failed.")
            break

        frame_counter += 1
        frame_height, frame_width = frame.shape[:2]

        # ── 1. Detection ──────────────────────────────────────────────────────
        ball  = ball_detector.detect(frame)
        hands = hand_detector.detect(frame)

        # ── 2. Landmark Smoothing (One Euro Filter per hand side) ─────────────
        if hands:
            active_sides = set()
            for hand in hands:
                side = hand.get("side", "LEFT")
                active_sides.add(side)
                if side not in smoothers:
                    smoothers[side] = LandmarkSmoother()
                hand["landmarks"] = smoothers[side].smooth(hand["landmarks"])
            # Reset smoother for any side that disappeared this frame
            for side in list(smoothers.keys()):
                if side not in active_sides:
                    smoothers[side].reset()
        else:
            for s in smoothers.values():
                s.reset()

        # ── 3. Independent Per-Hand Orientation Calculation ───────────────────
        hands_info = []
        if hands:
            for hand in hands:
                ori_res = orientation_det.calculate(hand)
                hands_info.append({
                    "side": hand.get("side", "UNKNOWN"),
                    "orientation": ori_res["orientation"],
                    "confidence": hand.get("confidence", 0.90),
                    "landmarks": hand.get("landmarks", []),
                    "hand": hand
                })

        # ── 4. Hand-ball association (Weighted Fingertip + MCP) ───────────────
        hand_ball_result = association.calculate(
            ball, hands, frame_width, frame_height
        )

        holder = hand_ball_result["holder"]
        holder_conf = hand_ball_result["confidence"]

        # ── 5. Feature Extraction (Active Ball Holder Hand) ────────────────────
        features = None
        active_hand_info = None

        if hands_info:
            # If a specific hand holds the ball, prioritize it; otherwise pick first hand
            if holder in ("LEFT", "RIGHT"):
                for h_info in hands_info:
                    if h_info["side"] == holder:
                        active_hand_info = h_info
                        break
            if active_hand_info is None:
                active_hand_info = hands_info[0]

            features = feature_extractor.extract(
                active_hand_info["hand"],
                ball,
                active_hand_info["orientation"],
                frame_width,
                frame_height
            )

        # ── 6. Grip Classification with 10-Frame Majority Voting ─────────────
        raw_pred = {
            "grip": "NONE",
            "raw_grip": "NONE",
            "confidence": 0.0,
            "probabilities": {},
            "alignment": {}
        }

        # Gating: Evaluates strictly the active hand holding the ball
        gating_passed = (
            active_hand_info is not None and
            active_hand_info["orientation"] == "BACK" and
            ball is not None and
            holder != "NONE" and
            features is not None
        )

        if gating_passed:
            raw_pred = grip_classifier.predict(features)
            prediction_history.append(raw_pred)
        else:
            prediction_history.clear()

        # Apply majority voting over the 10-frame buffer
        smoothed_result = get_smoothed_prediction(prediction_history)

        # ── 7. Detailed Console Telemetry & Probabilities ─────────────────────
        ball_status = "YES" if ball else "NO"

        if hands_info:
            ori_strs = " | ".join([f"[{h['side']}: {h['orientation']}]" for h in hands_info])
            print("-" * 65)
            print(f"[Frame {frame_counter:04d}] Hands: {len(hands_info)} ({ori_strs}) | Ball={ball_status} | Holder={holder} ({holder_conf*100:.0f}%) | Passed={gating_passed}")

            if features and active_hand_info:
                print(f"  Active Hand: {active_hand_info['side']} (Orientation: {active_hand_info['orientation']})")
                print("  Kinematic Features:")
                print(f"    wrist_rotation_angle   : {features.get('wrist_rotation_angle', 0.0):.2f}")
                print(f"    thumb_angle            : {features.get('thumb_angle', 0.0):.2f}")
                print(f"    index_curl             : {features.get('index_curl', 0.0):.2f}")
                print(f"    middle_curl            : {features.get('middle_curl', 0.0):.2f}")
                print(f"    ring_curl              : {features.get('ring_curl', 0.0):.2f}")
                print(f"    pinky_curl             : {features.get('pinky_curl', 0.0):.2f}")
                print(f"    index_middle_distance  : {features.get('index_middle_distance', 0.0):.4f}")
                print(f"    ball_coverage_ratio    : {features.get('ball_coverage_ratio', 0.0)}")

            if gating_passed and raw_pred.get("probabilities"):
                print("  Prediction Probabilities:")
                for cls_name, prob in raw_pred["probabilities"].items():
                    bar = "#" * int(prob * 20)
                    print(f"    - {cls_name:15s}: {prob * 100:5.1f}%  {bar}")

                print(f"  --> Raw Result     : {raw_pred.get('raw_grip', 'NONE').upper()} ({raw_pred.get('confidence', 0.0)*100:.1f}%)")
                print(f"  --> Smoothed (10f) : {smoothed_result['grip'].upper()} ({smoothed_result['confidence']*100:.1f}%)")

        # ── 8. Draw Ball & Landmarks ──────────────────────────────────────────
        frame = ball_detector.draw(frame, ball)
        frame = hand_detector.draw_landmarks(frame)

        # ── 9. On-screen Multi-Hand Debug & Telemetry HUD ─────────────────────
        y = 26
        lh = 22

        # Hands Count
        cv2.putText(frame, f"Hands detected: {len(hands_info)}", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0) if hands_info else (0, 0, 255), 2)
        y += lh

        # Separate Per-Hand Orientation Breakdown
        if hands_info:
            for h_info in hands_info:
                h_side = h_info["side"]
                h_ori = h_info["orientation"]
                h_col = (0, 255, 255) if h_ori == "PALM" else \
                        (0, 165, 255) if h_ori == "BACK" else (180, 180, 180)

                holder_tag = " [HOLDER]" if h_side == holder else ""
                cv2.putText(frame, f"{h_side} HAND: {h_ori}{holder_tag}", (12, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, h_col, 2)
                y += lh
        else:
            cv2.putText(frame, "No hands detected", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 2)
            y += lh

        # Ball Detection & Holder Status
        b_col = (0, 255, 0) if ball else (0, 0, 255)
        cv2.putText(frame, f"Ball: {'YES' if ball else 'NO'}", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, b_col, 2)
        y += lh

        if holder != "NONE":
            cv2.putText(frame, f"Holder: {holder} ({holder_conf * 100:.0f}%)", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Holder: NONE", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (150, 150, 150), 2)
        y += lh + 4

        # Final Smoothed Grip Prediction Box
        grip_name = smoothed_result["grip"].upper()
        conf_val = smoothed_result["confidence"] * 100

        if grip_name == "UNCERTAIN":
            cv2.putText(frame, "Grip: UNCERTAIN (<60%)", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 165, 255), 2)
            y += lh
            cv2.putText(frame, f"Confidence: {conf_val:.0f}%", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 165, 255), 2)
        elif grip_name != "NONE":
            cv2.putText(frame, f"Grip: {grip_name}", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.82, (0, 255, 0), 2)
            y += lh
            cv2.putText(frame, f"Confidence: {conf_val:.0f}%", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2)
        else:
            if not hands_info:
                reason = "No hand"
            elif not ball or holder == "NONE":
                reason = "No ball in hand"
            elif active_hand_info and active_hand_info["orientation"] == "PALM":
                reason = f"{active_hand_info['side']} in Palm view"
            else:
                reason = "Waiting for grip"
            cv2.putText(frame, f"Grip: NONE ({reason})", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 255), 2)
        y += lh + 4

        # Probability Distribution on HUD
        probs = raw_pred.get("probabilities", {})
        if probs:
            cv2.putText(frame, "-- PROBABILITIES --", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 0), 1)
            y += 18
            for cls_name, prob in probs.items():
                p_text = f"{cls_name}: {prob*100:.0f}%"
                cv2.putText(frame, p_text, (12, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1)
                y += 16

        # Right Panel: Kinematic Telemetry
        if features and active_hand_info:
            fx = frame_width - 295
            fy = 24
            cv2.putText(frame, f"=== KINEMATICS ({active_hand_info['side']}) ===", (fx, fy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 0), 1)
            fy += 18

            rot_angle = features.get('wrist_rotation_angle', 0.0)
            rot_tag = "Outward" if rot_angle > -70 else "Inward"

            feature_lines = [
                f"Wrist rot : {rot_angle:.1f} ({rot_tag})",
                f"Thumb curl: {features.get('thumb_angle', 0.0):.1f}",
                f"Index curl: {features.get('index_curl', 0.0):.1f}",
                f"Mid curl  : {features.get('middle_curl', 0.0):.1f}",
                f"Ring curl : {features.get('ring_curl', 0.0):.1f}",
                f"Pinky curl: {features.get('pinky_curl', 0.0):.1f}",
                f"Idx-Mid   : {features.get('index_middle_distance', 0.0):.3f}",
            ]

            if features.get("ball_index_distance") is not None:
                feature_lines += [
                    f"Ball-Idx  : {features.get('ball_index_distance', 0.0):.3f}",
                    f"Ball-Mid  : {features.get('ball_middle_distance', 0.0):.3f}",
                    f"Coverage  : {features.get('ball_coverage_ratio', 0.0):.3f}",
                ]

            for line in feature_lines:
                cv2.putText(frame, line, (fx, fy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 0), 1)
                fy += 17

        cv2.imshow("KhiladiPro - Cricket Grip AI (Multi-Hand)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()