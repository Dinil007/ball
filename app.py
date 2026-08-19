import cv2
import numpy as np
from collections import deque, Counter

from camera.webcam import Webcam
from ball.ball_detector import BallDetector
from hand.hand_detector import HandDetector
from hand.landmark_smoother import LandmarkSmoother
from hand.hand_orientation import HandOrientation
from tracking.hand_ball_association import HandBallAssociation
from features.feature_extractor import FeatureExtractor
from classifier.grip_classifier import GripClassifier


def get_smoothed_prediction(history_buffer):
    """
    Perform temporal smoothing over a 10-frame history buffer using majority voting
    and average confidence of the dominant class.
    """
    if not history_buffer:
        return {"grip": "NONE", "raw_grip": "NONE", "confidence": 0.0}

    # Extract all non-NONE valid frame predictions
    valid_predictions = [p for p in history_buffer if p["grip"] != "NONE"]
    if not valid_predictions:
        return {"grip": "NONE", "raw_grip": "NONE", "confidence": 0.0}

    # Majority voting over thresholded predictions
    grip_counts = Counter(p["grip"] for p in valid_predictions)
    most_common_grip, _ = grip_counts.most_common(1)[0]

    # Calculate average confidence for the dominant grip
    relevant_confs = [p["confidence"] for p in valid_predictions if p["grip"] == most_common_grip]
    avg_conf = sum(relevant_confs) / len(relevant_confs) if relevant_confs else 0.0

    # Also retrieve most frequent raw prediction
    raw_counts = Counter(p.get("raw_grip", p["grip"]) for p in valid_predictions)
    most_common_raw, _ = raw_counts.most_common(1)[0]

    return {
        "grip": most_common_grip,
        "raw_grip": most_common_raw,
        "confidence": round(avg_conf, 2)
    }


def draw_coaching_dashboard(frame, hands_info, ball, holder, holder_conf, smoothed_result, active_hand_info):
    """
    Build a high-definition (1920x1080) fullscreen coaching dashboard.
    - Left side: Camera feed scaled cleanly with aspect-ratio preservation.
    - Right side: Professional multi-hand coaching analysis sidebar.
    """
    canvas_w = 1920
    canvas_h = 1080
    sidebar_w = 520
    cam_area_w = canvas_w - sidebar_w  # 1400px

    # Create full canvas with deep dark background
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:, :] = (18, 15, 12)

    # ── 1. Scale and Position Camera Feed (Left Section) ──────────────────────
    cam_h, cam_w = frame.shape[:2]
    scale = min(cam_area_w / cam_w, canvas_h / cam_h)
    new_w = int(cam_w * scale)
    new_h = int(cam_h * scale)

    resized_cam = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Center camera frame inside the 1400x1080 left section
    offset_x = (cam_area_w - new_w) // 2
    offset_y = (canvas_h - new_h) // 2
    canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized_cam

    # Camera Live Feed Badge
    cv2.rectangle(canvas, (offset_x + 20, offset_y + 20), (offset_x + 175, offset_y + 58), (24, 20, 16), -1)
    cv2.rectangle(canvas, (offset_x + 20, offset_y + 20), (offset_x + 175, offset_y + 58), (55, 48, 40), 1)
    cv2.circle(canvas, (offset_x + 40, offset_y + 39), 7, (0, 0, 255), -1)
    cv2.putText(canvas, "LIVE CAMERA", (offset_x + 56, offset_y + 45),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)

    # ── 2. Right Side Coaching Analysis Panel ─────────────────────────────────
    panel_x = cam_area_w
    cv2.rectangle(canvas, (panel_x, 0), (canvas_w, canvas_h), (26, 22, 18), -1)  # Dark slate panel
    cv2.line(canvas, (panel_x, 0), (panel_x, canvas_h), (48, 42, 35), 2)         # Vertical divider

    # Header Banner Card
    cv2.rectangle(canvas, (panel_x + 25, 30), (canvas_w - 25, 120), (38, 32, 26), -1)
    cv2.rectangle(canvas, (panel_x + 25, 30), (canvas_w - 25, 120), (60, 52, 42), 1)
    cv2.putText(canvas, "CRICKET GRIP ANALYSIS", (panel_x + 45, 72),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 215, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "AI Bowler Real-time Coaching System", (panel_x + 45, 102),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 170, 170), 1, cv2.LINE_AA)

    y_cursor = 145
    card_bg = (34, 29, 24)
    card_border = (55, 48, 40)

    # ── Card 1: Multi-Hand Detection & Tracking ───────────────────────────────
    num_hands = len(hands_info)
    card1_h = 220 if num_hands >= 2 else 185
    cv2.rectangle(canvas, (panel_x + 25, y_cursor), (canvas_w - 25, y_cursor + card1_h), card_bg, -1)
    cv2.rectangle(canvas, (panel_x + 25, y_cursor), (canvas_w - 25, y_cursor + card1_h), card_border, 1)

    cv2.putText(canvas, "HAND DETECTION", (panel_x + 45, y_cursor + 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.60, (0, 215, 255), 1, cv2.LINE_AA)

    # Hands Detected count
    cv2.putText(canvas, "Hands Detected:", (panel_x + 45, y_cursor + 66),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.putText(canvas, str(num_hands), (panel_x + 235, y_cursor + 66),
                cv2.FONT_HERSHEY_DUPLEX, 0.72, (0, 230, 115) if num_hands > 0 else (140, 140, 140), 2, cv2.LINE_AA)

    # Individual Hand entries
    sub_y = y_cursor + 100
    if num_hands == 0:
        cv2.putText(canvas, "No hands visible", (panel_x + 45, sub_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (140, 140, 140), 1, cv2.LINE_AA)
    else:
        for idx, h_info in enumerate(hands_info[:2]):
            h_id = h_info.get("hand_id", idx + 1)
            h_side = h_info["side"]
            h_ori = h_info["orientation"]
            h_col = (0, 230, 115) if h_ori == "BACK" else (0, 215, 255) if h_ori == "PALM" else (160, 160, 160)

            # Highlight tag if this hand holds the ball
            is_holder = (h_side == holder and holder != "NONE")
            tag = "  [HOLDER]" if is_holder else ""

            # Line e.g. "Hand 1: RIGHT  (BACK)"
            cv2.putText(canvas, f"Hand {h_id}:", (panel_x + 45, sub_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{h_side}", (panel_x + 135, sub_y),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65, (240, 240, 240), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{h_ori}{tag}", (panel_x + 245, sub_y),
                        cv2.FONT_HERSHEY_DUPLEX, 0.62, h_col, 2, cv2.LINE_AA)
            sub_y += 34

    # Ball and Holder Status line
    holder_text = holder if holder != "NONE" else "NONE"
    holder_col = (0, 230, 115) if holder != "NONE" else (140, 140, 140)

    status_y = y_cursor + card1_h - 18
    cv2.putText(canvas, f"Ball: {'DETECTED' if ball else 'NONE'}", (panel_x + 45, status_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 230, 115) if ball else (80, 80, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Ball Holder: {holder_text}", (panel_x + 235, status_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, holder_col, 1, cv2.LINE_AA)

    y_cursor += card1_h + 25

    # ── Card 2: Grip Identification (Hero Card) ───────────────────────────────
    card2_h = 220
    cv2.rectangle(canvas, (panel_x + 25, y_cursor), (canvas_w - 25, y_cursor + card2_h), card_bg, -1)
    cv2.rectangle(canvas, (panel_x + 25, y_cursor), (canvas_w - 25, y_cursor + card2_h), card_border, 1)

    cv2.putText(canvas, "IDENTIFIED GRIP", (panel_x + 45, y_cursor + 38),
                cv2.FONT_HERSHEY_DUPLEX, 0.60, (0, 215, 255), 1, cv2.LINE_AA)

    raw_grip = smoothed_result["grip"].upper()
    conf_pct = int(smoothed_result["confidence"] * 100)

    if raw_grip in ("NONE", ""):
        grip_title = "Waiting..."
        grip_col = (160, 160, 160)
        conf_str = "--"
    elif raw_grip == "UNCERTAIN":
        grip_title = "UNCERTAIN"
        grip_col = (0, 215, 255)  # Amber
        conf_str = f"{conf_pct}%"
    else:
        grip_title = raw_grip.replace("_", " ")
        grip_col = (0, 230, 115)  # Crisp coaching green
        conf_str = f"{conf_pct}%"

    # Display Grip Title
    title_scale = 1.05 if len(grip_title) > 12 else 1.25
    cv2.putText(canvas, grip_title, (panel_x + 45, y_cursor + 95),
                cv2.FONT_HERSHEY_DUPLEX, title_scale, grip_col, 2, cv2.LINE_AA)

    # Confidence Row
    cv2.putText(canvas, "Confidence:", (panel_x + 45, y_cursor + 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.putText(canvas, conf_str, (panel_x + 220, y_cursor + 145),
                cv2.FONT_HERSHEY_DUPLEX, 0.82, grip_col, 2, cv2.LINE_AA)

    # Progress bar for confidence
    bar_x = panel_x + 45
    bar_y = y_cursor + 172
    bar_w = sidebar_w - 90
    bar_h = 16
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (48, 42, 36), -1)
    if conf_pct > 0 and raw_grip != "NONE":
        fill_w = int((conf_pct / 100.0) * bar_w)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), grip_col, -1)

    y_cursor += card2_h + 25

    # ── Card 3: Coaching Guidance / Status ────────────────────────────────────
    card3_h = 120
    cv2.rectangle(canvas, (panel_x + 25, y_cursor), (canvas_w - 25, y_cursor + card3_h), card_bg, -1)
    cv2.rectangle(canvas, (panel_x + 25, y_cursor), (canvas_w - 25, y_cursor + card3_h), card_border, 1)

    cv2.putText(canvas, "COACHING GUIDANCE", (panel_x + 45, y_cursor + 35),
                cv2.FONT_HERSHEY_DUPLEX, 0.60, (0, 215, 255), 1, cv2.LINE_AA)

    if not hands_info:
        guide_text = "Position hand in camera"
        guide_col = (160, 160, 160)
    elif not ball:
        guide_text = "Hold cricket ball in hand"
        guide_col = (0, 215, 255)
    elif holder == "NONE":
        guide_text = "Bring ball closer to fingers"
        guide_col = (0, 215, 255)
    elif active_hand_info and active_hand_info["orientation"] == "PALM":
        guide_text = f"Show back of {active_hand_info['side'].lower()} hand"
        guide_col = (0, 215, 255)
    elif raw_grip == "UNCERTAIN":
        guide_text = "Adjust finger alignment on seam"
        guide_col = (0, 215, 255)
    else:
        guide_text = "Grip locked & analyzed"
        guide_col = (0, 230, 115)

    cv2.putText(canvas, guide_text, (panel_x + 45, y_cursor + 80),
                cv2.FONT_HERSHEY_DUPLEX, 0.68, guide_col, 1, cv2.LINE_AA)

    # Footer note
    cv2.putText(canvas, "Press 'Q' or 'ESC' to exit", (panel_x + 160, canvas_h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (130, 130, 130), 1, cv2.LINE_AA)

    return canvas


def main():
    webcam = Webcam(0)

    if not webcam.is_opened():
        print("ERROR: Could not open webcam.")
        return

    ball_detector     = BallDetector()
    hand_detector     = HandDetector()
    smoothers         = {"LEFT": LandmarkSmoother(), "RIGHT": LandmarkSmoother()}
    orientation_det   = HandOrientation()
    association       = HandBallAssociation()
    feature_extractor = FeatureExtractor()
    grip_classifier   = GripClassifier(confidence_threshold=0.60)

    # 10-frame buffer for temporal prediction stabilization
    prediction_history = deque(maxlen=10)

    window_name = "KhiladiPro - Cricket Grip AI Coach"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("=" * 60)
    print("    CricketGrip AI - Multi-Hand Fullscreen Coach")
    print("=" * 60)
    print("Webcam started. Press Q or ESC in the window to quit.\n")

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
            for side in list(smoothers.keys()):
                if side not in active_sides:
                    smoothers[side].reset()
        else:
            for s in smoothers.values():
                s.reset()

        # ── 3. Independent Per-Hand Orientation Calculation ───────────────────
        hands_info = []
        if hands:
            for idx, hand in enumerate(hands):
                ori_res = orientation_det.calculate(hand)
                hands_info.append({
                    "hand_id": idx + 1,
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
            # If 1 hand, use that hand; if 2 hands, prioritize designated ball holder
            if len(hands_info) == 1:
                active_hand_info = hands_info[0]
            elif holder in ("LEFT", "RIGHT"):
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

        # Gating: strictly checks active hand holding the ball
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

        # ── 7. Clean Background Logging (every 30 frames) ─────────────────────
        if hands_info and frame_counter % 30 == 0:
            hand_statuses = [f"Hand {h['hand_id']} ({h['side']}: {h['orientation']})" for h in hands_info]
            h_str = " | ".join(hand_statuses)
            g_str = smoothed_result["grip"].upper()
            c_val = int(smoothed_result["confidence"] * 100)
            print(f"[Frame {frame_counter:04d}] {h_str} | Ball Holder: {holder} | Grip: {g_str} ({c_val}%)")

        # ── 8. Draw Ball & Landmarks on Camera Frame ──────────────────────────
        frame = ball_detector.draw(frame, ball)
        frame = hand_detector.draw_landmarks(frame)

        # ── 9. Render Fullscreen Dashboard ────────────────────────────────────
        dashboard = draw_coaching_dashboard(
            frame,
            hands_info,
            ball,
            holder,
            holder_conf,
            smoothed_result,
            active_hand_info
        )

        cv2.imshow(window_name, dashboard)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()