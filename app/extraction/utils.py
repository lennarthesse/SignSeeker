"""
Utilities for preparing the input videos for extraction
"""

from typing import List, Dict
import shutil
import pathlib
import csv
import os


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
    INPUT_DIR = pathlib.Path("/media/lennart/Data/Files/Documents/Studium/HS Harz/5. Semester/Jahresprojekt/datasets/all_videos")
    OUTPUT_DIR = pathlib.Path("app/output/filtered_videos")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    not_found = 0

    with open(VIDEO_LIST, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            filename = row.get("videos")

            if not filename:
                continue

            try:
                shutil.copy2(os.path.join(INPUT_DIR, filename), os.path.join(OUTPUT_DIR, filename))
            except FileNotFoundError:
                print(f"[WARNING] Couldn't find file with name {filename}")
                not_found += 1
    
    print(f"Done! Skipped {not_found} files because they were not found")