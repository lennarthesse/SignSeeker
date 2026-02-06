import cv2
import os
import mediapipe as mp
import numpy as np
import csv
import glob

from utils import MP_model, Video, normalize_name
from utils import draw_landmarks_on_image

NUM_LANDMARKS = 21
STATS = ["mean", "std"]

def convert_frame_to_mp_image(frame) -> mp.Image:
    # Convert frame to BGR for better recognition?
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Convert the frame to a MediaPipe Image object
    return mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_bgr
    )

def build_header_frame() -> list:
    hands = ["l", "r"]
    coords = ["x", "y", "z"]

    header = []

    for hand in hands:
        for landmark_idx in range(NUM_LANDMARKS):
            for coord in coords:
                header.append(f"{hand}_{coord}_{landmark_idx}")

    return header

def build_header_mean_std() -> list:
    hands = ["l", "r"]
    coords = ["x", "y", "z"]

    header = []

    for hand in hands:
        for landmark_idx in range(NUM_LANDMARKS):
            for coord in coords:
                for stat in STATS:
                    header.append(f"{hand}_{coord}_{landmark_idx}_{stat}")

    return header

def build_row_frame(result, frame: int, video_name: str) -> list:
    """
    Builds a CSV row for a single frame in a video containing the extracted landmarks, the frame index in the video, and the name of the video for grouping data by videos (and thus labels).
    
    :param result: The MediaPipe HandLandmarkerResult containing landmark coordinates, handedness and more.
    :type result: HandLandmarkerResult
    :param frame: The index of the frame in the video.
    :type frame: int
    :param video_name: The name of the video the frame is from.
    :type video_name: str
    :return: A list corresponding to a row in the CSV file, containing the extracted landmarks, frame index, and associated video.
    :rtype: list[Any]
    """
    return []

def build_row_mean_std(results: Video, label: str) -> list:
    """
    :param results: The collected results from a detection run on a video.
    :type results: list[HandLandmarkerResult]

    :param label: The corresponding label for the data in this video.
    :type label: str

    :return: A list with aggregated data for each coordinate of each landmark.
    :rtype: list[Any]
    """

    LEFT_SLOT = 0
    RIGHT_SLOT = 1
    FILL_VALUE = -11111111

    slots = [[], []]

    for result in results.landmarker_results:
        # skip empty results
        if len(result.hand_world_landmarks) == 0:
            continue

        # assign landmarks in result to either left or right hand
        assignment = [None, None] 
        for i in range(len(result.hand_world_landmarks)):
            landmarks = result.hand_world_landmarks[i]
            handedness = result.handedness[i][0].category_name

            # decide preferred slot
            preferred_slot = LEFT_SLOT if handedness.lower() == "left" else RIGHT_SLOT

            if assignment[preferred_slot] is None:
                assignment[preferred_slot] = landmarks
            else:
                other_slot = RIGHT_SLOT if preferred_slot == LEFT_SLOT else LEFT_SLOT
                if assignment[other_slot] is None:
                    assignment[other_slot] = landmarks

        # collect results of left hands
        if assignment[LEFT_SLOT] is not None:
            slots[LEFT_SLOT].append(assignment[LEFT_SLOT])

        # collect results of right hands
        if assignment[RIGHT_SLOT] is not None:
            slots[RIGHT_SLOT].append(assignment[RIGHT_SLOT])

    # aggregate mean, min, max for each coordinate
    all_features = []
    for slot in slots:
        if len(slot) == 0:
            # fill with fill value if no hand was detected in the entire video
            all_features.extend([FILL_VALUE] * NUM_LANDMARKS * 3 * len(STATS))
            continue

        buckets = [{"x": [], "y": [], "z": []} for _ in range(NUM_LANDMARKS)]

        for landmarks in slot:
            for idx, landmark in enumerate(landmarks):
                buckets[idx]["x"].append(landmark.x)
                buckets[idx]["y"].append(landmark.y)
                buckets[idx]["z"].append(landmark.z)

        for bucket in buckets:
            for coord in ("x", "y", "z"):
                values = np.array(bucket[coord])
                # replace values.min() and .max() with values.std() for use with standard deviation. also adjust the STATS constant!
                all_features.extend([
                    values.mean(),
                    values.std()
                ])

    all_features.append(label)
    return all_features

def build_video_lookup(csv_path: str) -> dict:
    lookup = {}

    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            video = row.get("videos")
            word = row.get("word")

            if not (video and word):
                continue

            key = normalize_name(video)

            if key not in lookup:
                lookup[key] = word

    return lookup

if __name__ == "__main__":
    INPUT_DIR = "input"

    files = os.scandir(INPUT_DIR) 
    file_count = len(glob.glob(INPUT_DIR + "/*.mp4"))
    file_idx = 1
    time = 0 # continuously running index to satisfy mediapipes need for a timestamp

    model = MP_model("hand_landmarker.task")
    model.init_video()

    # Initialize CSV file and writer and write the header row 
    csv_file = open("output/table.csv", "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_header = build_header_mean_std()
    csv_header.append("sign")
    csv_writer.writerow(csv_header)

    video_lookup = build_video_lookup("input/labels.csv")

    for file in files:
        # skip directories and non-video files
        if not (file.is_file() and file.name.endswith("mp4")):
            print(f"{file.name} is not a video, skipping it.")
            continue

        print(f"Processing file {file_idx} of {file_count} ({file.name})...")
        file_idx += 1

        # skip unlabeled videos
        label = video_lookup.get(normalize_name(file.name))
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
            annotated_image = draw_landmarks_on_image(frame, detection_result)

            cv2.imshow("Verification", annotated_image)
            cv2.waitKey(1) # opens the window and displays it for the given number of miliseconds

            time += 1

        cap.release()
        cv2.destroyAllWindows()

        video = Video(results)

        csv_writer.writerow(build_row_mean_std(video, label.lower()))

    csv_file.close()