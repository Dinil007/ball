import numpy as np
from collections import deque, Counter


class HandOrientation:
    """
    Robust Hand Orientation (PALM / BACK / UNCERTAIN) classifier
    using 3D palm plane normals, 2D anatomical chirality, finger direction,
    and 5-frame per-hand temporal majority voting.
    """

    def __init__(self, history_size=5):
        """
        Parameters:
            history_size (int): Number of frames in temporal majority voting buffer (default: 5).
        """
        self.history_size = history_size
        # Separate history buffers for LEFT and RIGHT hands
        self.history = {
            "LEFT": deque(maxlen=history_size),
            "RIGHT": deque(maxlen=history_size),
            "UNKNOWN": deque(maxlen=history_size)
        }

    def _calculate_raw(self, hand):
        """
        Calculate raw single-frame orientation based on palm plane and finger geometry.
        """
        landmarks = hand.get("landmarks", [])
        side = hand.get("side", "UNKNOWN")

        if len(landmarks) < 21:
            return "UNCERTAIN"

        wrist      = landmarks[0]
        thumb_tip  = landmarks[4]
        index_mcp  = landmarks[5]
        index_tip  = landmarks[8]
        middle_mcp = landmarks[9]
        middle_tip = landmarks[12]
        pinky_mcp  = landmarks[17]
        pinky_tip  = landmarks[20]

        # ── Step 1: Palm Plane Normal ─────────────────────────────────────────
        # vector1 = Index MCP (5) - Wrist (0)
        v1 = np.array([
            index_mcp["x"] - wrist["x"],
            index_mcp["y"] - wrist["y"],
            index_mcp["z"] - wrist["z"]
        ])

        # vector2 = Pinky MCP (17) - Wrist (0)
        v2 = np.array([
            pinky_mcp["x"] - wrist["x"],
            pinky_mcp["y"] - wrist["y"],
            pinky_mcp["z"] - wrist["z"]
        ])

        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 < 1e-5 or norm_v2 < 1e-5:
            return "UNCERTAIN"

        normal = np.cross(v1, v2)
        z_normal = normal[2]

        # ── Step 2: Finger & Palm Direction Vectors ───────────────────────────
        # Longitudinal hand axis (Wrist -> Middle MCP)
        fx = middle_mcp["x"] - wrist["x"]
        fy = middle_mcp["y"] - wrist["y"]

        # Finger direction vector (Middle MCP -> Middle Tip)
        finger_x = middle_tip["x"] - middle_mcp["x"]
        finger_y = middle_tip["y"] - middle_mcp["y"]

        # Transverse palm axis (Index MCP -> Pinky MCP)
        px = pinky_mcp["x"] - index_mcp["x"]
        py = pinky_mcp["y"] - index_mcp["y"]

        # 2D cross-product of longitudinal vs transverse (chirality determinant)
        cross_2d = (fx * py) - (fy * px)

        # ── Step 3: Weighted Geometric Score ──────────────────────────────────
        # Normalize score by palm scale
        palm_scale = norm_v1 * norm_v2
        normalized_z = (z_normal / palm_scale) if palm_scale > 1e-6 else z_normal

        # Combined score from 3D normal and 2D chirality
        score = 0.5 * normalized_z + 0.5 * (cross_2d / palm_scale if palm_scale > 1e-6 else cross_2d)

        # ── Step 4: Hand-Specific Classification (Aligned with Camera View) ───
        threshold = 0.05

        if side == "RIGHT":
            # For Right Hand: score < -threshold -> PALM, score > threshold -> BACK
            if score < -threshold:
                return "PALM"
            elif score > threshold:
                return "BACK"
            else:
                return "UNCERTAIN"

        elif side == "LEFT":
            # For Left Hand: score > threshold -> PALM, score < -threshold -> BACK
            if score > threshold:
                return "PALM"
            elif score < -threshold:
                return "BACK"
            else:
                return "UNCERTAIN"

        else:
            # Fallback when side is UNKNOWN
            if abs(score) < threshold:
                return "UNCERTAIN"
            return "PALM" if score < 0 else "BACK"

    def calculate(self, hand):
        """
        Calculate stable palm/back orientation using per-hand temporal majority voting.

        Parameters:
            hand (dict): {"side": "LEFT"|"RIGHT", "confidence": float, "landmarks": list}

        Returns:
            dict: {"orientation": "PALM" | "BACK" | "UNCERTAIN"}
        """
        side = hand.get("side", "UNKNOWN")
        if side not in self.history:
            self.history[side] = deque(maxlen=self.history_size)

        raw_orientation = self._calculate_raw(hand)

        # Append valid classifications to history buffer
        if raw_orientation in ("PALM", "BACK"):
            self.history[side].append(raw_orientation)

        # Apply 5-frame majority voting
        buf = self.history[side]
        if len(buf) >= 3:
            counts = Counter(buf)
            most_common_ori, count = counts.most_common(1)[0]
            if count >= len(buf) * 0.5:
                return {"orientation": most_common_ori}

        return {"orientation": raw_orientation}