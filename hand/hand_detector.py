import mediapipe as mp


class HandDetector:

    def __init__(self, max_hands=2):

        self.max_hands = max_hands

        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.mp_draw = mp.solutions.drawing_utils


    def detect(self, frame):

        rgb_frame = frame[:, :, ::-1]

        results = self.hands.process(rgb_frame)

        return results


    def draw_landmarks(self, frame, results):

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

        return frame


    def get_handedness(self, results):

        hands = []

        if results.multi_hand_landmarks and results.multi_handedness:

            for hand_info in results.multi_handedness:

                label = hand_info.classification[0].label

                # Correct for mirrored webcam display
                if label == "Left":
                    label = "Right"
                else:
                    label = "Left"

                hands.append(label)

        return hands