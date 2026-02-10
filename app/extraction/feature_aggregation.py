"""
Module for aggregating video data into a CSV rows.
"""

import os
import csv
from typing import List, Any

import joblib

from app.utils import Video
from app.utils import build_header_mean_std, build_row_mean_std


OUTPUT_DIR = "app/extraction/output/"


def _write_csv_file(rows: List[Any]):
    """
    Writes a number of rows into a CSV file.
    
    :param rows: Rows to be written into the CSV file.
    :type rows: List[Any]
    """
    with open(OUTPUT_DIR + "table.csv", "w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(build_header_mean_std())

        for row in rows:
            csv_writer.writerow(row)


def load_and_aggregate():
    """
    Loads `Video` objects and aggregates their data into a CSV file for model training.
    """
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
    _write_csv_file(rows)
    print("Done")


if __name__ == "__main__":
    load_and_aggregate()
