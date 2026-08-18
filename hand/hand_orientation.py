import numpy as np


class HandOrientation:

    def calculate(self, landmarks, hand_side):

        wrist = landmarks[0]

        index_mcp = landmarks[5]
        pinky_mcp = landmarks[17]


        v1 = np.array([
            index_mcp.x - wrist.x,
            index_mcp.y - wrist.y,
            index_mcp.z - wrist.z
        ])


        v2 = np.array([
            pinky_mcp.x - wrist.x,
            pinky_mcp.y - wrist.y,
            pinky_mcp.z - wrist.z
        ])


        normal = np.cross(v1, v2)

        z_value = normal[2]


        threshold = 0.015


        # Left hand is mirrored relative to right hand
        if hand_side == "Left":
            z_value = -z_value


        if z_value < -threshold:
            return "PALM"


        elif z_value > threshold:
            return "BACK"


        else:
            return "UNCERTAIN"