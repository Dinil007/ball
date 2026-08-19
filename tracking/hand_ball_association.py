import math


class HandBallAssociation:
    """
    Determines which hand is interacting with/holding the cricket ball using
    a weighted combination of fingertip and MCP joint distances, with distance
    thresholding and confidence scoring.
    """

    FINGERTIP_INDICES = [4, 8, 12, 16, 20]   # Thumb, Index, Middle, Ring, Pinky tips
    MCP_INDICES       = [5, 9, 13, 17]        # Index, Middle, Ring, Pinky MCP joints

    def __init__(self, distance_threshold=160, ambiguity_margin=25):
        """
        Parameters:
            distance_threshold (float): Maximum weighted pixel distance to consider a hand holding the ball.
            ambiguity_margin (float): Minimum score difference between two hands to avoid ambiguous assignment.
        """
        self.distance_threshold = float(distance_threshold)
        self.ambiguity_margin = float(ambiguity_margin)

    def calculate(self, ball, hands, frame_width, frame_height):
        """
        Determine which hand is interacting with the cricket ball.

        Parameters:
            ball         : dict {"x1", "y1", "x2", "y2", "confidence"} or None
            hands        : list of hand dicts [{"side": str, "confidence": float, "landmarks": list}, ...]
            frame_width  : int — width of current frame in pixels
            frame_height : int — height of current frame in pixels

        Returns:
            dict: {"holder": "LEFT" | "RIGHT" | "NONE", "confidence": float}
        """

        if not ball or not hands:
            return {"holder": "NONE", "confidence": 0.0}

        # 1. Calculate ball center in pixel space
        ball_cx = (ball["x1"] + ball["x2"]) / 2.0
        ball_cy = (ball["y1"] + ball["y2"]) / 2.0

        hand_scores = []

        # 2. Compute weighted distance scores for each hand
        for hand in hands:
            landmarks = hand.get("landmarks", [])
            side = hand.get("side", "UNKNOWN")

            if len(landmarks) < 21:
                continue

            # Fingertip distances
            tip_distances = []
            for idx in self.FINGERTIP_INDICES:
                lm = landmarks[idx]
                px = lm["x"] * frame_width
                py = lm["y"] * frame_height
                dist = math.sqrt((px - ball_cx) ** 2 + (py - ball_cy) ** 2)
                tip_distances.append(dist)
            tip_score = sum(tip_distances) / len(tip_distances)

            # MCP joint distances
            mcp_distances = []
            for idx in self.MCP_INDICES:
                lm = landmarks[idx]
                px = lm["x"] * frame_width
                py = lm["y"] * frame_height
                dist = math.sqrt((px - ball_cx) ** 2 + (py - ball_cy) ** 2)
                mcp_distances.append(dist)
            mcp_score = sum(mcp_distances) / len(mcp_distances)

            # Weighted final score (0.7 * tip_score + 0.3 * mcp_score)
            final_score = (0.7 * tip_score) + (0.3 * mcp_score)
            min_tip_dist = min(tip_distances)

            hand_scores.append({
                "side": side,
                "score": final_score,
                "min_tip_dist": min_tip_dist
            })

        if not hand_scores:
            return {"holder": "NONE", "confidence": 0.0}

        # Sort hands by lowest score (closest to ball)
        hand_scores.sort(key=lambda x: x["score"])
        best = hand_scores[0]

        # 3. Distance threshold check: if closest hand is too far, ball is not held
        if best["score"] > self.distance_threshold or best["min_tip_dist"] > self.distance_threshold:
            return {"holder": "NONE", "confidence": 0.0}

        # 4. Multi-hand ambiguity check and confidence calculation
        if len(hand_scores) > 1:
            second_best = hand_scores[1]
            score_diff = second_best["score"] - best["score"]

            # If both hands have almost identical distance scores, avoid ambiguous assignment
            if score_diff < self.ambiguity_margin:
                return {"holder": "NONE", "confidence": 0.0}

            margin_conf = min(1.0, score_diff / 80.0)
            dist_conf = max(0.0, 1.0 - (best["score"] / self.distance_threshold))
            confidence = 0.5 * margin_conf + 0.5 * dist_conf
        else:
            confidence = max(0.0, 1.0 - (best["score"] / self.distance_threshold))

        # Clamp confidence to a realistic positive percentage
        confidence = max(0.50, min(0.99, confidence))

        return {
            "holder": best["side"],
            "confidence": round(confidence, 2)
        }
