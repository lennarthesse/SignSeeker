"""
Module for extracting landmarks from videos and saving them as `Video` objects.
"""

import cv2
import os
import csv
import glob
from typing import List
from pathlib import Path

import joblib

from app.utils import MP_model, Video
from app.utils import convert_frame_to_mp_image, draw_landmarks_on_image


LABELS_LOCATION = "app/extraction/input/labels.csv"
INPUT_DIR       = "app/extraction/output/filtered_videos/"
OUTPUT_DIR      = "app/extraction/output/"
MODEL_LOCATION  = "app/hand_landmarker.task"


def _build_video_lookup(csv_path: str) -> dict:
    print("Building lookup...", end="")
    lookup = {}

    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            video = row.get("videos")
            word = row.get("word")

            if not (video and word):
                continue

            if video not in lookup:
                lookup[video] = word

    print("Done")
    return lookup


def _extract_and_convert_videos() -> List[Video]:
    files = os.scandir(INPUT_DIR) 
    file_count = len(glob.glob(INPUT_DIR + "*.mp4"))
    file_idx = 1
    time = 0 # continuously running index to satisfy mediapipes need for a timestamp

    model = MP_model(MODEL_LOCATION, MP_model.RunningMode.VIDEO)

    video_lookup = _build_video_lookup(LABELS_LOCATION)

    videos = []

    for file in files:
        # skip directories and non-video files
        if not (file.is_file() and file.name.endswith("mp4")):
            print(f"{file.name} is not a video, skipping it.")
            continue

        print(f"Processing file {file_idx} of {file_count} ({file.name})...")
        file_idx += 1

        # skip unlabeled videos
        label = video_lookup.get(file.name)
        if label is None:
            print("Couldn't find a label for this video, skipping it.")
            continue

        results = []

        # Loop through each frame in the video
        cap = cv2.VideoCapture(file.path)
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            mp_image = convert_frame_to_mp_image(frame)
            detection_result = model.landmarker.detect_for_video(mp_image, time)
            results.append(detection_result)

            # draw result on the original frame (consider using mp_image.numpy_view() for viewing the image mediapipe actually works with)
            #annotated_image = draw_landmarks_on_image(frame, detection_result)
            #cv2.imshow("Verification", annotated_image)
            #cv2.waitKey(1) # opens the window and displays it for the given number of miliseconds

            time += 1

        cap.release()
        cv2.destroyAllWindows()

        videos.append(Video(file.name, results, label))
    
    return videos


def extract_and_save():
    """
    Extracts landmarks from videos in input directory and saves them as pickled `Video` objects.
    """
    videos = _extract_and_convert_videos()
    dir = Path(OUTPUT_DIR + "extracted_videos/")
    dir.mkdir(parents=True, exist_ok=True)

    n = 1
    total = len(videos)

    for video in videos:
        print(f"Saving file {n} of {total} to disk ({video.filename})...", end="")
        joblib.dump(video, os.path.join(dir, video.filename + ".pkl"))
        print("Done")
        n += 1


if __name__ == "__main__":
    extract_and_save()
