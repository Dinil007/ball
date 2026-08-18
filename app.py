import cv2

from camera.webcam import Webcam
from ball.ball_detector import BallDetector


def main():

    webcam = Webcam(0)

    if not webcam.is_opened():
        print("ERROR: Could not open webcam.")
        return

    ball_detector = BallDetector()

    print("Webcam started.")
    print("Press Q to quit.")

    while True:

        success, frame = webcam.read()

        if not success:
            print("ERROR: Frame failed.")
            break

        ball = ball_detector.detect(frame)

        if ball:
            print(
                f"BALL: x1={ball['x1']} y1={ball['y1']} "
                f"x2={ball['x2']} y2={ball['y2']} "
                f"conf={ball['confidence']:.2f}"
            )

        frame = ball_detector.draw(frame, ball)

        cv2.imshow(
            "KhiladiPro - Cricket Ball Detection",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()