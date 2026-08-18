import cv2
from ultralytics import YOLO


class BallDetector:

    def __init__(self, model_path="models/best.pt", conf_threshold=0.25):

        self.conf_threshold = conf_threshold
        self.model = YOLO(model_path)


    def detect(self, frame):

        results = self.model(
            frame,
            conf=self.conf_threshold,
            verbose=False
        )

        best_ball = None
        max_conf = 0.0

        for r in results:
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    conf = float(box.conf[0])
                    if conf > max_conf:
                        max_conf = conf
                        coords = box.xyxy[0].cpu().numpy()
                        best_ball = {
                            "x1": int(coords[0]),
                            "y1": int(coords[1]),
                            "x2": int(coords[2]),
                            "y2": int(coords[3]),
                            "confidence": round(conf, 4)
                        }

        return best_ball


    def draw(self, frame, ball):

        if ball is not None:
            x1 = ball["x1"]
            y1 = ball["y1"]
            x2 = ball["x2"]
            y2 = ball["y2"]
            conf = ball["confidence"]

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            label = f"Cricket Ball: {conf:.2f}"
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        return frame