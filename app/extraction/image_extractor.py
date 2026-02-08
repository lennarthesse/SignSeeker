import os
import sys
import csv

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

from app.utils import draw_landmarks_on_image


if __name__ == "__main__":
    # specify the name of the input directory
    input_name = input("What is the exact name of your input directory? ")
    
    if not os.path.isdir(input_name):
        print("Could not find specified input directory. Is your data in the same folder you are running this script from?")
        sys.exit()
    
    # set name of output directory and dataset
    dataset_name = input("Please give the dataset you are creating a name (no special characters): ").lower().replace(" ", "_")

    if os.path.isdir(dataset_name):
        if dataset_name == input_name:
            print("Can't use input directory as output!")
            print("Aborting...")
            sys.exit()
            
        overwrite = input("Found an existing output directory, do you really want to use it? (y/n) ")
        if not overwrite.lower() == "y":
            print("Aborting...")
            sys.exit()
    else:
        os.mkdir(dataset_name)
    
    # create csv file, keeping original newline characters and using utf-8
    csv_file = open(os.path.join(dataset_name, dataset_name + ".csv"), "w", newline="", encoding="utf-8")
    
    csv_writer = csv.writer(csv_file)
    
    # set up csv header row, adjust for landmark setup (hands, hands+face, holistic)
    csv_header = [f"{i}_{axis}" for i in range(21) for axis in ("x", "y", "z")]
    csv_header.extend(["sign", "hash"])
    
    csv_writer.writerow(csv_header)
    
    model_path = "hand_landmarker.task"
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2)
    
    # files[] = load files from "input" folder
    with os.scandir(input_name) as directory, vision.HandLandmarker.create_from_options(options) as landmarker:
        # for each file in files[]:
        for f in directory:
            print("Processing " + f.name + "...")

            # extract sign label from filename and skip if the labeling is incorrect
            sign = f.name[0].lower() 
            if not sign.isalpha():
                print("Invalid file labeling, skipping this file. Does the filename start with the corresponding sign letter?")
                continue

            # detection_result = run handlandmarker(file)
            image = mp.Image.create_from_file(os.path.join(input_name, f.name))
            detection_result = landmarker.detect(image)
            
            # draw landmarks on image and save as control #
            # annotated_image = draw_landmarks_on_image(file, detection_result)
            annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
            
            # cv2.imwrite(os.path.join(dataset_name, sign + "_" +  str(hash(f.name)) + ".jpg"), annotated_image)
            cv2.imwrite(os.path.join(dataset_name, f.name), annotated_image)
            #cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            
            #for hand in handedness list:
            for hand in detection_result.hand_landmarks:
                row = []
                for landmark in hand:
                    #print(str(landmark.x) + " " + str(landmark.y) + " " + str(landmark.z))                
                    row.extend([landmark.x, landmark.y, landmark.z])

                # Write to CSV
                # wenn zwei hände in einem bild vorhanden sind, bekommen sie jeweils denselben hash damit tabpfn weiß, dass sie zusammen gehören #
                # wenn es immer nur eine hand gäbe, würde tabpfn den hash effekiv ignorieren, da jede zeile einen einzigartigen hash hat und kein clustering entsteht #
                row.extend([sign, hash(f.name)])
                csv_writer.writerow(row)
            
    csv_file.close
    print("Done!")
