import os

import cv2
import mediapipe

from app.utils import MP_model


def load_files_from_directory(path):
    return os.scandir(path)

if __name__ == "__main__":
    files = load_files_from_directory(".")
    for file in files:
        print(file.name)

    model = MP_model("hand_landmarker.task")

    model.init_livestream()
    
    cap = cv2.VideoCapture(0)
    time = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mediapipe.Image(
            image_format=mediapipe.ImageFormat.SRGB,
            data=frame_rgb
        )

        model.landmarker.detect_async(mp_image, time)
        time += 1

        # Optional: display the frame
        #cv2.imshow("Webcam", frame)
        #if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        #    break

    cap.release()
    cv2.destroyAllWindows()
