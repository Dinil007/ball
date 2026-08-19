import math
import numpy as np


class FeatureExtractor:
    """
    Convert hand landmarks + ball position + orientation into a
    normalized numerical feature dictionary ready for a classifier.
    """

    # ---------------------------------------------------------------------------
    # Internal geometry helpers
    # ---------------------------------------------------------------------------

    @staticmethod
    def _vec(a, b):
        """2-D vector from landmark dict a → b."""
        return np.array([b["x"] - a["x"], b["y"] - a["y"]])

    @staticmethod
    def _vec3(a, b):
        """3-D vector from landmark dict a → b (uses z coordinate)."""
        return np.array([b["x"] - a["x"], b["y"] - a["y"], b.get("z", 0.0) - a.get("z", 0.0)])

    @staticmethod
    def _angle_between(v1, v2):
        """Angle in degrees between two 2-D vectors."""
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-8 or n2 < 1e-8:
            return 0.0
        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        return math.degrees(math.acos(cos_a))

    @staticmethod
    def _dist2d(a, b):
        """Euclidean distance between two landmark dicts (normalized space)."""
        return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2)

    @staticmethod
    def _hand_scale(landmarks):
        """
        Reference scale = distance from wrist (0) to middle MCP (9).
        Used to normalize all intra-hand distances so they are
        independent of how close the hand is to the camera.
        """
        wrist = landmarks[0]
        mid_mcp = landmarks[9]
        scale = FeatureExtractor._dist2d(wrist, mid_mcp)
        return scale if scale > 1e-6 else 1.0

    # ---------------------------------------------------------------------------
    # Joint angle from a triplet of landmarks  (base → mid → tip)
    # ---------------------------------------------------------------------------

    def _joint_angle(self, landmarks, base_idx, mid_idx, tip_idx):
        """
        Calculate the bending angle at the middle joint (mid_idx)
        formed by the vectors mid→base and mid→tip.
        Returns angle in degrees (0° = fully extended, 180° = fully bent).
        """
        base = landmarks[base_idx]
        mid  = landmarks[mid_idx]
        tip  = landmarks[tip_idx]
        v1 = self._vec(mid, base)
        v2 = self._vec(mid, tip)
        return round(self._angle_between(v1, v2), 2)

    # ---------------------------------------------------------------------------
    # Ball-center in normalized landmark space
    # ---------------------------------------------------------------------------

    @staticmethod
    def _ball_center_norm(ball, frame_width, frame_height):
        """
        Convert ball pixel bbox → normalized (x, y) matching MediaPipe
        landmark coordinate space (0-1).
        """
        cx = (ball["x1"] + ball["x2"]) / 2.0 / frame_width
        cy = (ball["y1"] + ball["y2"]) / 2.0 / frame_height
        return {"x": cx, "y": cy}

    # ---------------------------------------------------------------------------
    # NEW: Biomechanical / Kinematic Feature Helpers
    # ---------------------------------------------------------------------------

    def _wrist_rotation_angle(self, landmarks):
        """
        [Feature 1] Wrist Rotation Angle.

        Measures the rotation of the palm by computing the angle between:
        - index_vector : Wrist(0) → Index MCP(5)
        - pinky_vector : Wrist(0) → Pinky MCP(17)
        and returning the signed angle of their mean direction against horizontal.

        Positive → wrist rotated outward (outswing posture)
        Negative → wrist rotated inward  (inswing posture)
        """
        wrist     = landmarks[0]
        index_mcp = landmarks[5]
        pinky_mcp = landmarks[17]

        # 2D vectors from wrist
        iv = self._vec(wrist, index_mcp)
        pv = self._vec(wrist, pinky_mcp)

        # Mid direction = average of the two vectors
        mid = (iv + pv) / 2.0

        # Signed angle vs positive X axis
        angle = math.degrees(math.atan2(mid[1], mid[0]))
        return round(angle, 2)

    def _finger_curl(self, landmarks, base_idx, mid_idx, tip_idx):
        """
        [Feature 2] Single-finger curl angle using vectors:
            base→mid  and  mid→tip
        Returns 0° (fully extended) → ~180° (fully curled inward).
        """
        return self._joint_angle(landmarks, base_idx, mid_idx, tip_idx)

    def _fingertip_spread(self, landmarks, tip_a_idx, tip_b_idx, scale):
        """
        [Feature 3] Distance between two fingertips, normalized by hand scale.
        """
        d = self._dist2d(landmarks[tip_a_idx], landmarks[tip_b_idx])
        return round(d / scale, 4)

    def _ball_mcp_distance(self, ball_norm, landmarks, mcp_idx, scale):
        """
        [Feature 4] Distance from ball center to an MCP joint, normalized.
        """
        d = self._dist2d(landmarks[mcp_idx], ball_norm)
        return round(d / scale, 4)

    def _ball_coverage_ratio(self, ball_norm, landmarks, scale):
        """
        [Feature 5] Average distance from ball center to index tip (8) and
        middle tip (12), normalized by hand scale.

        Low value → fingers are close to / covering the ball (seam/cutter/knuckle)
        High value → fingers are far from ball
        """
        d8  = self._dist2d(landmarks[8],  ball_norm)
        d12 = self._dist2d(landmarks[12], ball_norm)
        return round(((d8 + d12) / 2.0) / scale, 4)

    # ---------------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------------

    def extract(self, hand, ball, orientation, frame_width, frame_height):
        """
        Extract numerical features from a single hand + ball observation.

        Parameters
        ----------
        hand         : dict  {"side", "confidence", "landmarks": [{x,y,z}×21]}
        ball         : dict  {"x1","y1","x2","y2","confidence"}  or  None
        orientation  : str   "PALM" | "BACK" | "UNCERTAIN"
        frame_width  : int
        frame_height : int

        Returns
        -------
        dict of features, or None if hand data is insufficient.
        """

        if hand is None:
            return None

        landmarks = hand.get("landmarks", [])
        if len(landmarks) < 21:
            return None

        try:
            scale = self._hand_scale(landmarks)

            # ── Existing: Finger joint angles ────────────────────────────────
            index_angle  = self._joint_angle(landmarks, 5, 6, 8)
            middle_angle = self._joint_angle(landmarks, 9, 10, 12)
            ring_angle   = self._joint_angle(landmarks, 13, 14, 16)

            # ── Existing: Thumb-index distance (normalized) ──────────────────
            thumb_index_distance = round(
                self._dist2d(landmarks[4], landmarks[8]) / scale, 4
            )

            # ── Existing: Wrist angle ────────────────────────────────────────
            wrist   = landmarks[0]
            mid_mcp = landmarks[9]
            dx = mid_mcp["x"] - wrist["x"]
            dy = mid_mcp["y"] - wrist["y"]
            wrist_angle = round(
                math.degrees(math.atan2(abs(dy), abs(dx))) if abs(dx) > 1e-8 else 90.0,
                2
            )

            # ── Existing: Ball-relative distances ────────────────────────────
            if ball is not None:
                ball_norm = self._ball_center_norm(ball, frame_width, frame_height)
                ball_thumb_distance  = round(
                    self._dist2d(landmarks[4],  ball_norm) / scale, 4
                )
                ball_index_distance  = round(
                    self._dist2d(landmarks[8],  ball_norm) / scale, 4
                )
                ball_middle_distance = round(
                    self._dist2d(landmarks[12], ball_norm) / scale, 4
                )
                ball_ring_distance   = round(
                    self._dist2d(landmarks[16], ball_norm) / scale, 4
                )
            else:
                ball_norm            = None
                ball_thumb_distance  = None
                ball_index_distance  = None
                ball_middle_distance = None
                ball_ring_distance   = None

            # ── NEW Feature 1: Wrist Rotation Angle ──────────────────────────
            wrist_rotation_angle = self._wrist_rotation_angle(landmarks)

            # ── NEW Feature 2: Per-Finger Curl Angles ────────────────────────
            # Thumb   : joints 2 → 3 → 4
            thumb_angle  = self._finger_curl(landmarks, 2, 3, 4)
            # Index   : joints 5 → 6 → 8
            index_curl   = self._finger_curl(landmarks, 5, 6, 8)
            # Middle  : joints 9 → 10 → 12
            middle_curl  = self._finger_curl(landmarks, 9, 10, 12)
            # Ring    : joints 13 → 14 → 16
            ring_curl    = self._finger_curl(landmarks, 13, 14, 16)
            # Pinky   : joints 17 → 18 → 20
            pinky_curl   = self._finger_curl(landmarks, 17, 18, 20)

            # ── NEW Feature 3: Finger Spread Distances ────────────────────────
            # Index tip(8) ↔ Middle tip(12)
            index_middle_distance = self._fingertip_spread(landmarks, 8, 12, scale)
            # Middle tip(12) ↔ Ring tip(16)
            middle_ring_distance  = self._fingertip_spread(landmarks, 12, 16, scale)
            # Ring tip(16) ↔ Pinky tip(20)
            ring_pinky_distance   = self._fingertip_spread(landmarks, 16, 20, scale)

            # ── NEW Feature 4: Ball–MCP Distances ────────────────────────────
            if ball_norm is not None:
                ball_index_mcp_distance  = self._ball_mcp_distance(ball_norm, landmarks, 5, scale)
                ball_middle_mcp_distance = self._ball_mcp_distance(ball_norm, landmarks, 9, scale)
                ball_ring_mcp_distance   = self._ball_mcp_distance(ball_norm, landmarks, 13, scale)
            else:
                ball_index_mcp_distance  = None
                ball_middle_mcp_distance = None
                ball_ring_mcp_distance   = None

            # ── NEW Feature 5: Ball Coverage Ratio ───────────────────────────
            if ball_norm is not None:
                ball_coverage_ratio = self._ball_coverage_ratio(ball_norm, landmarks, scale)
            else:
                ball_coverage_ratio = None

            return {
                # ── Existing features (unchanged) ──────────────────────────
                "hand_side":                hand.get("side", "UNKNOWN"),
                "orientation":              orientation,
                "index_angle":              index_angle,
                "middle_angle":             middle_angle,
                "ring_angle":               ring_angle,
                "thumb_index_distance":     thumb_index_distance,
                "ball_thumb_distance":      ball_thumb_distance,
                "ball_index_distance":      ball_index_distance,
                "ball_middle_distance":     ball_middle_distance,
                "ball_ring_distance":       ball_ring_distance,
                "wrist_angle":              wrist_angle,
                # ── New kinematic features ──────────────────────────────────
                "wrist_rotation_angle":     wrist_rotation_angle,
                "thumb_angle":              thumb_angle,
                "index_curl":               index_curl,
                "middle_curl":              middle_curl,
                "ring_curl":                ring_curl,
                "pinky_curl":               pinky_curl,
                "index_middle_distance":    index_middle_distance,
                "middle_ring_distance":     middle_ring_distance,
                "ring_pinky_distance":      ring_pinky_distance,
                "ball_index_mcp_distance":  ball_index_mcp_distance,
                "ball_middle_mcp_distance": ball_middle_mcp_distance,
                "ball_ring_mcp_distance":   ball_ring_mcp_distance,
                "ball_coverage_ratio":      ball_coverage_ratio,
            }

        except (IndexError, ZeroDivisionError, ValueError):
            return None
