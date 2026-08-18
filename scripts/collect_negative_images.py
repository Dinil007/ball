import cv2
import os
from datetime import datetime


save_folder = "data/negative_images"

os.makedirs(save_folder, exist_ok=True)

camera = cv2.VideoCapture(0)

count = 0

print("Press SPACE to capture image")
print("Press Q to quit")


while True:

    success, frame = camera.read()

    if not success:
        break

    cv2.imshow(
        "Negative Image Collector",
        frame
    )

    key = cv2.waitKey(1) & 0xFF


    # SPACE key
    if key == 32:

        filename = f"negative_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"

        path = os.path.join(
            save_folder,
            filename
        )

        cv2.imwrite(path, frame)

        count += 1

        print(f"Saved: {path}")


    # Q key
    if key == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()

print(f"Total images captured: {count}")