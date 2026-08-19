import math
import cv2
import mediapipe as mp


class HandDetector:
    """
    Robust MediaPipe Hand Detector with persistent hand tracking across frames
    and per-hand temporal landmark smoothing.
    """

    def __init__(
        self,
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        smoothing_alpha=0.3
    ):
        """
        Parameters:
            static_image_mode (bool): Process video stream (False) or single images (True).
            max_num_hands (int): Max number of hands to detect simultaneously.
            model_complexity (int): Landmark model complexity (1 = full model).
            min_detection_confidence (float): Detection confidence threshold.
            min_tracking_confidence (float): Tracking confidence threshold.
            smoothing_alpha (float): Exponential smoothing weight for current frame (0.3).
        """
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.smoothing_alpha = smoothing_alpha

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.static_image_mode,
            max_num_hands=self.max_num_hands,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.results = None

        # Persistent hand tracking memory: stores tracked hands by id
        # {hand_id: {"side": str, "wrist": (x, y), "landmarks": list}}
        self.tracked_hands = {}
        self.next_hand_id = 0

    def _match_and_smooth(self, detected_raw_hands):
        """
        Match detected hands to previously tracked hands using wrist position distance,
        stabilize hand identity/side, and apply exponential temporal smoothing.
        """
        if not detected_raw_hands:
            # Clear or decay tracking memory when no hands are present
            self.tracked_hands = {}
            return []

        smoothed_output_hands = []
        unmatched_curr = list(range(len(detected_raw_hands)))
        matched_prev_ids = set()

        new_tracked = {}

        # 1. Match current detections to previous tracked hands by minimum wrist distance
        for prev_id, prev_data in self.tracked_hands.items():
            prev_wx, prev_wy = prev_data["wrist"]
            best_idx = None
            best_dist = float("inf")

            for idx in unmatched_curr:
                curr_wrist = detected_raw_hands[idx]["landmarks"][0]
                dist = math.sqrt((curr_wrist["x"] - prev_wx) ** 2 + (curr_wrist["y"] - prev_wy) ** 2)
                if dist < best_dist and dist < 0.25:  # max matching distance in normalized space
                    best_dist = dist
                    best_idx = idx

            if best_idx is not None:
                matched_prev_ids.add(prev_id)
                unmatched_curr.remove(best_idx)

                curr_hand = detected_raw_hands[best_idx]
                prev_landmarks = prev_data["landmarks"]

                # Apply exponential moving average smoothing: 0.7 * previous + 0.3 * current
                alpha = self.smoothing_alpha
                smoothed_lm = []
                for i in range(len(curr_hand["landmarks"])):
                    c = curr_hand["landmarks"][i]
                    p = prev_landmarks[i]
                    sx = (1.0 - alpha) * p["x"] + alpha * c["x"]
                    sy = (1.0 - alpha) * p["y"] + alpha * c["y"]
                    sz = (1.0 - alpha) * p["z"] + alpha * c["z"]
                    smoothed_lm.append({"x": sx, "y": sy, "z": sz})

                # Maintain stable hand side label
                side = curr_hand["side"] if curr_hand["side"] != "UNKNOWN" else prev_data["side"]

                new_tracked[prev_id] = {
                    "side": side,
                    "wrist": (smoothed_lm[0]["x"], smoothed_lm[0]["y"]),
                    "landmarks": smoothed_lm
                }

                smoothed_output_hands.append({
                    "side": side,
                    "confidence": curr_hand["confidence"],
                    "landmarks": smoothed_lm
                })

        # 2. Register any unmatched current detections as new hands
        for idx in unmatched_curr:
            curr_hand = detected_raw_hands[idx]
            hid = self.next_hand_id
            self.next_hand_id += 1

            new_tracked[hid] = {
                "side": curr_hand["side"],
                "wrist": (curr_hand["landmarks"][0]["x"], curr_hand["landmarks"][0]["y"]),
                "landmarks": curr_hand["landmarks"]
            }

            smoothed_output_hands.append(curr_hand)

        self.tracked_hands = new_tracked
        return smoothed_output_hands

    def detect(self, frame):
        """
        Detect hands in the frame, track persistent identity, and return smoothed landmarks.

        Returns:
            list of dict: [{"side": "LEFT"|"RIGHT", "confidence": float, "landmarks": list}, ...]
        """
        if frame is None or frame.size == 0:
            self.tracked_hands = {}
            return []

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb_frame)

        detected_raw = []

        if self.results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(self.results.multi_hand_landmarks):
                landmarks_list = []
                for lm in hand_landmarks.landmark:
                    landmarks_list.append({
                        "x": float(lm.x),
                        "y": float(lm.y),
                        "z": float(lm.z)
                    })

                side = "UNKNOWN"
                confidence = 1.0

                if self.results.multi_handedness and idx < len(self.results.multi_handedness):
                    hand_info = self.results.multi_handedness[idx]
                    raw_label = hand_info.classification[0].label
                    confidence = float(hand_info.classification[0].score)

                    # Correct for standard mirrored camera display
                    if raw_label == "Left":
                        side = "RIGHT"
                    elif raw_label == "Right":
                        side = "LEFT"
                    else:
                        side = raw_label.upper()

                detected_raw.append({
                    "side": side,
                    "confidence": round(confidence, 4),
                    "landmarks": landmarks_list
                })

        return self._match_and_smooth(detected_raw)

    def draw_landmarks(self, frame):
        """
        Draw MediaPipe hand landmarks and connections on the frame.
        """
        if self.results and self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

        return frame