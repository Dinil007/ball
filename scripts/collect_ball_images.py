import cv2
import os


save_path = "data/ball_dataset/images"

os.makedirs(save_path, exist_ok=True)


cap = cv2.VideoCapture(0)

count = 0

print("Press SPACE to capture image")
print("Press Q to quit")


while True:

    ret, frame = cap.read()

    if not ret:
        break


    cv2.imshow(
        "Ball Dataset Collection",
        frame
    )


    key = cv2.waitKey(1) & 0xFF


    if key == ord(" "):

        filename = os.path.join(
            save_path,
            f"ball_{count}.jpg"
        )

        cv2.imwrite(
            filename,
            frame
        )

        print(
            "Saved:",
            filename
        )

        count += 1


    elif key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()