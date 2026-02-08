import cv2
from collections import deque

import numpy as np

from app.utils import MP_model, Video
from app.utils import convert_frame_to_mp_image
from app.model.inference import load_model
from app.extraction.video_extractor import build_row_mean_std

BUFFER_SIZE = 30
MIN_PROBA = 0.8
PREDICT_EVERY_N = 1
N = 0

if __name__ == "__main__":

    buffer = deque(maxlen=BUFFER_SIZE)
    classification_model, label_encoder = load_model("app/model/2k_v1_model.txt", "app/model/2k_v1_labelEncoder.pkl")

    mp_model = MP_model("app/hand_landmarker.task", MP_model.RunningMode.LIVE_STREAM)
    
    cap = cv2.VideoCapture(0)
    time = 0

    while cap.isOpened():
        success, bgr_frame = cap.read()
        if not success:
            print("Ignoring empty camera frame")
            continue

        mp_image_rgb = convert_frame_to_mp_image(bgr_frame)
        mp_model.landmarker.detect_async(mp_image_rgb, time)

        result, frame = None, None

        with mp_model.frame_lock:
            if mp_model.latest_frame is not None:
                result = mp_model.latest_result
                frame = mp_model.latest_frame

        if result is not None and frame is not None:
            buffer.append(result)

            if len(buffer) >= BUFFER_SIZE and N > PREDICT_EVERY_N:
                N = 0
                video = Video("", list(buffer), "")

                row = build_row_mean_std(video)
                
                if classification_model is not None and label_encoder is not None:
                    X = np.array(row[:-1]).reshape(1, -1)
                    y_proba: np.ndarray = classification_model.predict(X) # type: ignore
                    top_y_pred = np.argmax(y_proba, axis=1)[0]

                    top_proba = y_proba[0, top_y_pred]
                    top_label = label_encoder.inverse_transform([top_y_pred])[0]

                    if top_proba > MIN_PROBA:
                        print(f"Prediction: {top_label} ({top_proba})")
            
            annotated_image_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            cv2.imshow("Webcam", annotated_image_bgr)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break
        
        N += 1
        time += 1

    # hopefully prevent segmentation error by destroying the mp_model
    del mp_model
    cap.release()
    cv2.destroyAllWindows()
