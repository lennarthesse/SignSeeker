"""
Utilities for preparing the input videos for extraction
"""

from typing import List, Dict
from pathlib import Path
import shutil
import csv
import os


WORDS           = ["hello", "learn", "sign", "language", "america", "like", "fun", "and", "what", "who", "where", "how", "good", "bad", "you"]
FULL_CSV        = Path("app/extraction/input/labels.csv")
FILTERED_CSV    = Path("app/extraction/input/filtered_labels.csv")
INPUT_DIR       = Path("/path/to/all_videos")
OUTPUT_DIR      = Path("app/extraction/output/filtered_videos")


def filter_video_list(video_list: Path, words: List[str]) -> None:
    """
    Filters the provided words from the original labels CSV and creates a filtered CSV.
    
    :param video_list: Path to the full labels CSV.
    :type video_list: str
    :param words: Words to filter.
    :type words: List[str]
    """
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

    with open(FILTERED_CSV, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["videos", "word"])
        writer.writeheader()
        
        for video, word in filtered_list.items():
            writer.writerow({"videos": video, "word": word})


def copy_videos(csv_path: Path, input_dir: Path, output_dir: Path) -> None:
    """
    Copies videos listed in labels CSV file from the original dataset to a specified directory.
    
    :param csv_path: Path to the CSV file containing video filenames and labels.
    :type csv_path: Path
    :param input_dir: Path to directory containing all video files.
    :type input_dir: Path
    :param output_dir: Where to copy the filtered videos to."
    :type output_dir: Path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    not_found = 0

    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            filename = row.get("videos")

            if not filename:
                continue
            try:
                print(f"Copying file {filename}...", end="")
                shutil.copy2(os.path.join(input_dir, filename), os.path.join(output_dir, filename))
                print("Done")
            except FileNotFoundError:
                print(f"[WARNING] Couldn't find file with name {filename}")
                not_found += 1
    
    print(f"Done! Skipped {not_found} files because they were not found")


if __name__ == "__main__":
    filter_video_list(FULL_CSV, WORDS)
    copy_videos(FILTERED_CSV, INPUT_DIR, OUTPUT_DIR)
