"""
Utilities for the AI package containing classes like Video or MP_Model
"""

from typing import List, Tuple, Any
from enum import Enum
import threading

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


# ------- #
# CLASSES #
# ------- #
class Video:
    def __init__(
            self,
            filename: str,
            landmarker_results: List[HandLandmarkerResult], # type: ignore
            label: str
            ) -> None: 
        self.filename = filename
        self.landmarker_results = landmarker_results
        self.label = label


class MP_model:

    class RunningMode(Enum):
        LIVE_STREAM = 0
        VIDEO = 1

    def __init__(self, path_to_model: str, running_mode: RunningMode):
        """
        Initializes the path to the MediaPipe model for later use in video/image initializations.
        
        :param path_to_model: Path to the .task-model.
        :type path_to_model: str
        """
        self.model_path = path_to_model

        if running_mode is self.RunningMode.LIVE_STREAM:
            self.latest_frame = None
            self.latest_result = None
            self.frame_lock = threading.Lock()

            def print_result(result: HandLandmarkerResult, output_image_rgb: mp.Image, timestamp_ms: int): # type: ignore
                #print('hand landmarker result: {}'.format(result))
                annotated_image_rgb = draw_landmarks_on_image(output_image_rgb.numpy_view(), result)
                with self.frame_lock:
                    self.latest_frame = annotated_image_rgb
                    self.latest_result = result

            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self.model_path),
                running_mode=VisionRunningMode.LIVE_STREAM,
                result_callback=print_result,
                num_hands=2)
            
            self.landmarker = HandLandmarker.create_from_options(options)

        elif running_mode is self.RunningMode.VIDEO:
            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self.model_path),
                running_mode=VisionRunningMode.VIDEO,
                num_hands=2)
            
            self.landmarker = HandLandmarker.create_from_options(options)


# ------------------- #
# FEATURE AGGREGATION #
# ------------------- #
LANDMARK_INDICES = [
     5,  8, # index finger knuckle and tip
     9, 12, # middle finger knuckle and tip
    13, 16, # ring finger knuckle and tip
    17, 20, # pinky finger knuckle and tip
     1,  4  # thumb root and tip
]
HANDS = ["l", "r"]
COORDS = ["x", "y", "z"]
STATS = ["mean", "std"]
LEFT_SLOT = 0
RIGHT_SLOT = 1
FILL_VALUE = -11111111


def build_header_mean_std() -> List[Any]:
    """
    Builds a header for the training data that can be written into a CSV file.
    
    :return: List holding the column names.
    :rtype: List[Any]
    """
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


def build_row_mean_std(video: Video) -> List[Any]:
    """
    Builds a row containing the aggregated data of a video that can be written into a CSV file.
    
    :param video: Video object holding landmark and label data.
    :type video: Video
    :return: List holding the feature values of the row.
    :rtype: List[Any]
    """
    slots, root_slots = _get_slots(video)

    row = _aggregate_features(slots, root_slots)
    row.append(video.label)

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


# DELETE THIS IF NO PROBLEMS OCCUR FROM REMOVING
#def normalize_name(name: str) -> str:
#    """
#    Normalizes a file name by replacing %xx escapes by their single-character
#    equivalent and certain special characters and normalizing unicode characters.
#    
#    :param name: The name to normalize.
#    :type name: str
#    :return: The normalized name.
#    :rtype: str
#    """
#    name = os.path.basename(name)
#    name = urllib.parse.unquote(name)
#    name = unicodedata.normalize("NFKC", name)
#    name = (
#        name.replace("•", "")
#            .replace(" ", "_")
#            .replace("/", "_")
#    )
#    return name.lower()



def convert_frame_to_mp_image(bgr_frame: cv2.typing.MatLike) -> mp.Image:
    """
    Convert an OpenCV frame in MatLike format and BGR color to a MediaPipe Image on RGB rolor.
    
    :param bgr_frame: BGR Frame in MatLike format
    :type bgr_rame: MatLike
    :return: RGB Frame in MediaPipe Image format
    :rtype: Image
    """
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

    # Convert the frame to a MediaPipe Image object
    return mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


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
