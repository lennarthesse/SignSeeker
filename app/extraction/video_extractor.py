import cv2
import os
import numpy as np
import csv
import glob
from typing import List, Any, Tuple

import joblib

from app.utils import MP_model, Video
from app.utils import convert_frame_to_mp_image

LANDMARK_INDICES = [
     5,  8,
     9, 12,
    13, 16,
    17, 20,
     1,  4
]
HANDS = ["l", "r"]
COORDS = ["x", "y", "z"]
STATS = ["mean", "std"]
LEFT_SLOT = 0
RIGHT_SLOT = 1
FILL_VALUE = -11111111

INPUT_DIR = "app/input/filtered_videos/"
OUTPUT_DIR = "app/output/"
MODEL_LOCATION = "app/hand_landmarker.task"
LABELS_LOCATION = "app/input/labels.csv"


def build_header_mean_std() -> List[Any]:
    header = []

    for hand in HANDS:
        for landmark_idx in range(21):
            if landmark_idx in LANDMARK_INDICES:
                for coord in COORDS:
                    for stat in STATS:
                        header.append(f"{hand}_{coord}_{landmark_idx}_{stat}")
    
    for hand in HANDS:
        for coord in COORDS:
            for stat in STATS:
                header.append(f"{hand}_root_{coord}_{stat}")

    header.append("label")

    return header


def build_row_mean_std(vid: Video) -> List[Any]:
    slots, root_slots = _get_slots(vid)

    row = _aggregate_features(slots, root_slots)
    row.append(vid.label)

    return row


def _get_slots(video: Video) -> Tuple[List[Any], List[Any]]:
    slots = [[], []]
    root_slots = [[], []]

    for landmarker_result in video.landmarker_results:
        # skip empty results
        if len(landmarker_result.hand_world_landmarks) == 0:
            continue

        # assign landmarks in result to either left or right hand
        assignment = [None, None]       # this is a list of List[Landmark]      (list of landmarks of an entire hand)
        root_assignment = [None, None]  # this is a list of NormalizedLandmark  (list of root landmarks)
        for hand_idx in range(len(landmarker_result.hand_world_landmarks)):
            landmarks = landmarker_result.hand_world_landmarks[hand_idx]
            root_landmark = landmarker_result.hand_landmarks[hand_idx][0]
            handedness = landmarker_result.handedness[hand_idx][0].category_name

            # decide preferred slot
            preferred_slot = LEFT_SLOT if handedness.lower() == "left" else RIGHT_SLOT

            if assignment[preferred_slot] is None:
                assignment[preferred_slot] = landmarks
            else:
                other_slot = RIGHT_SLOT if preferred_slot == LEFT_SLOT else LEFT_SLOT
                if assignment[other_slot] is None:
                    assignment[other_slot] = landmarks

            if root_assignment[preferred_slot] is None:
                root_assignment[preferred_slot] = root_landmark
            else:
                other_slot = RIGHT_SLOT if preferred_slot == LEFT_SLOT else LEFT_SLOT
                if root_assignment[other_slot] is None:
                    root_assignment[other_slot] = root_landmark

        # collect results of left hands
        if assignment[LEFT_SLOT] is not None:
            slots[LEFT_SLOT].append(assignment[LEFT_SLOT])
        if root_assignment[LEFT_SLOT] is not None:
            root_slots[LEFT_SLOT].append(root_assignment[LEFT_SLOT])

        # collect results of right hands
        if assignment[RIGHT_SLOT] is not None:
            slots[RIGHT_SLOT].append(assignment[RIGHT_SLOT])
        if root_assignment[RIGHT_SLOT] is not None:
            root_slots[RIGHT_SLOT].append(root_assignment[RIGHT_SLOT])
    
    return slots, root_slots


def _aggregate_features(slots: List[Any], root_slots: List[Any]) -> List[Any]:
    # aggregate mean, min, max for each coordinate
    features = []

    for slot in slots:
        if len(slot) == 0:
            # fill with fill value if no hand was detected in the entire video
            features.extend([FILL_VALUE] * len(LANDMARK_INDICES) * len(COORDS) * len(STATS))
            continue

        buckets = [{"x": [], "y": [], "z": []} for _ in range(21)]

        for landmarks in slot:
            for idx, landmark in enumerate(landmarks):
                if idx in LANDMARK_INDICES:
                    buckets[idx]["x"].append(landmark.x)
                    buckets[idx]["y"].append(landmark.y)
                    buckets[idx]["z"].append(landmark.z)

        for idx, bucket in enumerate(buckets):
            if idx in LANDMARK_INDICES:
                for coord in COORDS:
                    values = np.array(bucket[coord])
                    # replace values.min() and .max() with values.std() for use with standard deviation. also adjust the STATS constant!
                    features.extend([
                        values.mean(),
                        values.std()
                    ])
        
    for root_slot in root_slots:
        if len(root_slot) == 0:
            features.extend([FILL_VALUE] * len(COORDS) * len(STATS))
            continue
        
        bucket = {"x": [], "y": [], "z": []}

        for landmark in root_slot:
            bucket["x"].append(landmark.x)
            bucket["y"].append(landmark.y)
            bucket["z"].append(landmark.z)

        for coord in COORDS:
            values = np.array(bucket[coord])
            features.extend([
                values.mean(),
                values.std()
            ])

    return features


def build_video_lookup(csv_path: str) -> dict:
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


def write_csv_file(rows: List[Any]):
    # Initialize CSV file and writer and write the header row 
    with open(OUTPUT_DIR + "table.csv", "w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(build_header_mean_std())

        for row in rows:
            csv_writer.writerow(row)


def extract_and_convert_videos() -> List[Video]:
    files = os.scandir(INPUT_DIR) 
    file_count = len(glob.glob(INPUT_DIR + "*.mp4"))
    file_idx = 1
    time = 0 # continuously running index to satisfy mediapipes need for a timestamp

    model = MP_model(MODEL_LOCATION, MP_model.RunningMode.VIDEO)

    video_lookup = build_video_lookup(LABELS_LOCATION)

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
    videos = extract_and_convert_videos()
    
    n = 1
    total = len(videos)

    for video in videos:
        print(f"Saving file {n} of {total} to disk ({video.filename})...", end="")
        joblib.dump(video, OUTPUT_DIR + "extracted_videos/" + video.filename + ".pkl")
        print("Done")
        n += 1


def load_and_aggregate():
    rows = []
    n = 1

    files = os.scandir(OUTPUT_DIR + "extracted_videos/")
    for file in files:
        print(f"Processing file {n} ({file.name})...", end="")
        video: Video = joblib.load(OUTPUT_DIR + "extracted_videos/" + file.name)
        rows.append(build_row_mean_std(video))
        print("Done")
        n += 1

    print("Writing CSV file...", end="")
    write_csv_file(rows)
    print("Done")


if __name__ == "__main__":
    #extract_and_save()
    load_and_aggregate()
