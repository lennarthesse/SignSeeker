from typing import List, Dict
import shutil
import csv
import os
import unicodedata
import urllib.parse
import pathlib

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
import numpy as np
import cv2

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


WORDS = ["hello", "learn", "sign", "language", "america", "like", "fun", "and", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "what", "who", "where", "how", "good", "bad", "you"]


def filter_video_list(video_list: str, words: List[str]) -> None:
    filtered_list: Dict[str, str] = {}

    with open(video_list, newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)

        for row in reader:
            video = row.get("videos")
            word = row.get("word")

            if not (video and word):
                continue

            if word in words:
                filtered_list[video] = word

    with open("filtered_list.csv", "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["videos", "word"])
        writer.writeheader()
        
        for video, word in filtered_list.items():
            writer.writerow({"videos": video, "word": word})


def copy_videos():
    VIDEO_LIST = "app/input/filtered_labels.csv"
    INPUT_DIR = pathlib.Path("app/input/all_videos")
    OUTPUT_DIR = pathlib.Path("app/output/filtered_videos")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(VIDEO_LIST, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            raw_filename = row.get("videos")

            if not raw_filename:
                continue

            filename = normalize_name(raw_filename)

            shutil.copy2(os.path.join(INPUT_DIR, filename), os.path.join(OUTPUT_DIR, filename))


def normalize_name(name: str) -> str:
    """
    Normalizes a file name by replacing %xx escapes by their single-character
    equivalent and certain special characters and normalizing unicode characters.
    
    :param name: The name to normalize.
    :type name: str
    :return: The normalized name.
    :rtype: str
    """
    name = os.path.basename(name)
    name = urllib.parse.unquote(name)
    name = unicodedata.normalize("NFKC", name)
    name = (
        name.replace("•", "")
            .replace(" ", "_")
            .replace("/", "_")
    )
    return name.lower()


class Video:
    def __init__(self, landmarker_results: List[HandLandmarkerResult]) -> None: # type: ignore
        self.landmarker_results = landmarker_results


class MP_model:
    def __init__(self, path_to_model: str):
        """
        Initializes the path to the MediaPipe model for later use in video/image initializations.
        
        :param path_to_model: Path to the .task-model.
        :type path_to_model: str
        """
        self.model_path = path_to_model
     
    def init_livestream(self):
        """Initializes MediaPipe with VisionRunningMode=LIVE_STREAM."""

        def print_result(result, output_image: mp.Image, timestamp_ms: int):
            print('hand landmarker result: {}'.format(result))
            #annotated_image = draw_landmarks_on_image(output_image.numpy_view(), result)
            #cv2.imshow("imaeg", annotated_image)
            #cv2.waitKey(1)

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=print_result)
        
        self.landmarker = HandLandmarker.create_from_options(options)
        
    def init_video(self):
        """Initializes MediaPipe with VisionRunningMode=VIDEO."""

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2)
        
        self.landmarker = HandLandmarker.create_from_options(options)


def draw_landmarks_on_image(rgb_image, detection_result):
    MARGIN = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([ # type: ignore
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks # type: ignore
        ])
    
        solutions.drawing_utils.draw_landmarks( # type: ignore
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS, # type: ignore
            solutions.drawing_styles.get_default_hand_landmarks_style(), # type: ignore
            solutions.drawing_styles.get_default_hand_connections_style()) # type: ignore

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image
