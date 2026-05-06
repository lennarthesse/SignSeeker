import cv2

from app.seeker import SignSeeker


if __name__ == "__main__":

    with SignSeeker() as skr:

        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            success, bgr_frame = cap.read()
            if not success:
                print("Ignoring empty camera frame")
                continue

            # actual SignSeeker usage
            label, prob = skr.infer(bgr_frame)
            if label is not None:
                print(f"Prediction: {label} ({prob})")       

            # display frame if available
            if skr.latest_frame_bgr is not None:
                cv2.imshow("Webcam", skr.latest_frame_bgr)
            if cv2.waitKey(100) & 0xFF == 27:  # ESC to quit
                break

        cap.release()
        cv2.destroyAllWindows()
