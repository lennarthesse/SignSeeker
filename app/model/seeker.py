"""Module providing a high-level interface to the extraction and classification logic."""

from collections import deque
from typing import Union, Tuple

import numpy as np
import cv2
from cv2.typing import MatLike

from app.utils import MP_model, Video
from app.utils import convert_frame_to_mp_image, build_row_mean_std
from app.model.utils import load_model


BUFFER_SIZE = 50
MIN_PROBA = 0.8
PREDICT_EVERY_N = 2
N = 1               # always initialize/reset self.n to 1

class SignSeeker:
    """
    An extraction and classification model providing functionality for obtaining sign predictions from video frames.
    """
    def __init__(self):
        self.mp_model = MP_model("app/hand_landmarker.task", MP_model.RunningMode.LIVE_STREAM)
        self.classification_model, self.label_encoder = load_model("app/model/2k_v1_model.txt", "app/model/2k_v1_labelEncoder.pkl")
        self.buffer = deque(maxlen=BUFFER_SIZE)
        self.time = 0
        self.n = N

        self.latest_frame_bgr = None
    
    def infer(self, bgr_frame: MatLike) -> Union[Tuple[str, str], None]:
        """
        Takes in a 
        """

        mp_image_rgb = convert_frame_to_mp_image(bgr_frame)
        self.mp_model.landmarker.detect_async(mp_image_rgb, self.time)
        self.time += 1

        result, frame = None, None

        with self.mp_model.frame_lock:
            if self.mp_model.latest_frame is not None:
                result = self.mp_model.latest_result
                frame = self.mp_model.latest_frame

        if result is not None and frame is not None:
            self.buffer.append(result)
            self.latest_frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if len(self.buffer) >= BUFFER_SIZE and self.n >= PREDICT_EVERY_N:
                self.n = N

                # create a video object with empty filename and label since they don't exist and build row from video
                video = Video("", list(self.buffer), "") 
                row = build_row_mean_std(video)
                
                X = np.array(row[:-1]).reshape(1, -1) # explain why this is here
                probabilities: np.ndarray = self.classification_model.predict(X) # type: ignore
                top_prediction = np.argmax(probabilities, axis=1)[0]
                top_label = self.label_encoder.inverse_transform([top_prediction])[0]
                top_probability = probabilities[0, top_prediction]

                if top_probability > MIN_PROBA:
                    print(f"Prediction: {top_label} ({top_probability})")
                    return top_label, top_probability
        
        return None
