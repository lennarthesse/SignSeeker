"""Module providing a high-level interface to the live extraction and classification logic."""

from collections import deque
from typing import Union, Tuple, List

import numpy as np
import cv2
from cv2.typing import MatLike

from .utils import MP_model, Video
from .utils import convert_frame_to_mp_image, build_row_mean_std, load_model

# these values seem to be working well but aren't optimzed yet
BUFFER_SIZE = 50
PREDICT_EVERY_N = 10


class SignSeeker:
    """
    An extraction and classification model providing functionality for obtaining sign predictions from video frames.
    """
    def __init__(self, mp_model_path: str, classification_model_path: str, label_encoder_path: str):
        self._mp_model = MP_model(mp_model_path, MP_model.RunningMode.LIVE_STREAM)
        self._classification_model, self._label_encoder = load_model(classification_model_path, label_encoder_path)
        self._buffer = deque(maxlen=BUFFER_SIZE)
        self._time = 0
        self._n = 0

        self.latest_frame_bgr = None

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        # this might need to go deeper. maybe try to shut the actual handlandmarker down in the class
        del self._mp_model
    
    def batch_infer(self, bgr_frames: List[MatLike]) -> Union[Tuple[str, float], Tuple[None, None]]:
        """
        Take a batch of frames and run a prediction on just this data as the buffer.
        
        :param bgr_frames: Batch of frames to run the prediction on
        :type bgr_frames: List[MatLike]
        :return: Predicted label and associated probability or (None, None)
        :rtype: Tuple[str, float] | Tuple[None, None]
        """
        return None, None

    def infer(self, bgr_frame: MatLike) -> Union[Tuple[str, float], Tuple[None, None]]:
        """
        Add a new frame to the models buffer and run a prediction. The model uses the last few frames in its buffer to aggregate the data and run the prediction.
        
        :param bgr_frame: Input frame in BGR color to add to the buffer.
        :type bgr_frame: MatLike
        :return: Predicted label and associated probability or (None, None)
        :rtype: Tuple[str, float] | Tuple[None, None]
        """
        self.preview(bgr_frame, clear_buffer=False)
        self._n += 1

        result = None

        with self._mp_model.frame_lock:
            if self._mp_model.latest_result is not None:
                result = self._mp_model.latest_result

        if result is not None:
            self._buffer.append(result)

            if len(self._buffer) >= BUFFER_SIZE and self._n >= PREDICT_EVERY_N:
                self._n = 0

                # create a video object with empty filename and label since they don't exist and build row from video
                video = Video("", list(self._buffer), "") 
                row = build_row_mean_std(video)
                
                X = np.array(row[:-1]).reshape(1, -1) # explain why this is here
                probabilities: np.ndarray = self._classification_model.predict(X) # type: ignore
                top_prediction = np.argmax(probabilities, axis=1)[0]
                top_label = self._label_encoder.inverse_transform([top_prediction])[0]
                top_probability = probabilities[0, top_prediction]

                return top_label, top_probability
        
        return None, None

    def preview(self, bgr_frame: MatLike, clear_buffer: bool = True) -> MatLike | None:
        if clear_buffer:
            self._buffer.clear()
        
        mp_image_rgb = convert_frame_to_mp_image(bgr_frame)
        self._mp_model.landmarker.detect_async(mp_image_rgb, self._time)
        self._time += 1

        frame = None

        with self._mp_model.frame_lock:
            if self._mp_model.latest_frame is not None:
                frame = self._mp_model.latest_frame

        if frame is not None:
            self.latest_frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return self.latest_frame_bgr

        return None