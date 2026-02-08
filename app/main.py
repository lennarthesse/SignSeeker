import cv2

from app.utils import MP_model
from app.utils import convert_frame_to_mp_image
from app.model.inference import load_model
from app.extraction.video_extractor import build_header_mean_std

if __name__ == "__main__":
    model = MP_model("app/hand_landmarker.task", MP_model.RunningMode.LIVE_STREAM)
    
    cap = cv2.VideoCapture(0)
    time = 0

    while cap.isOpened():
        success, bgr_frame = cap.read()
        if not success:
            print("Ignoring empty camera frame")
            continue

        mp_image_rgb = convert_frame_to_mp_image(bgr_frame)

        model.landmarker.detect_async(mp_image_rgb, time)

        time += 1

        with model.frame_lock:
            if model.latest_frame is not None:
                annotated_image_bgr = cv2.cvtColor(model.latest_frame, cv2.COLOR_RGB2BGR)

                cv2.imshow("Webcam", annotated_image_bgr)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                    break

    cap.release()
    cv2.destroyAllWindows()
