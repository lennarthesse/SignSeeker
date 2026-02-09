# How to Extract Landmark Data from Videos

*Package: app.extraction*

The extraction package is used to prepare the training data for the model. It takes example videos of people signing and detects up to two hands per frame with MediaPipe. The landmarker results are collected into a video object along with the associated label and filename and saved to the disk. In the next step the video objects are loaded and processed further by aggregating select landmarks over the length of the video. The resulting data of each video is treated as a single row in a CSV file that is later used to train a classification model (see [classification.md](./classification.md)).

## Filter Videos

We used a dataset from Zahid Mittha with this project which can be found [here](https://huggingface.co/datasets/ZahidYasinMittha/American-Sign-Language-Dataset). It contains 100k+ labeled videos of people signing a single sign in American Sign Language. There are more than 2k different words with a minimum of 30 videos per word. The labels are provided for each file name in a CSV file. Since classification with 2k classes is a hard task you should filter the dataset to only contain the words you actually need.

### app.extraction.utils

This module contains functions to filter and copy videos from the original dataset. There are five setup variables:

1. `WORDS`: A list of words to filter from the full dataset.

2. `FULL_CSV`: Location of the original label CSV file.

3. `FILTERED_CSV`: Path to the filtered CSV file.

4. `INPUT_DIR`: Path to the directory containing all the video files.

5. `OUTPUT_DIR`: Where to copy the filtered videos to.

You will have to set `INPUT_DIR` to point to the location where you store your video files and define the words you want to filter in `WORDS`.

Now you can run the script:

```bash
$ python -m app.extraction.utils
```

This will filter the words specified in `WORDS` from the original CSV file and create a new one. Then the videos contained in this file are copied into the output folder.

## Extract Landmarks from Videos

To extract landmarks from the videos and save them as video objects, use the module `app.extraction.video_extractor`. There are four setup variables:

1. `LABELS_LOCATION`: Location of the CSV file storing the labels for each video. For faster execution use your filtered CSV file.

2. `INPUT_DIR`: Location of the videos you want to extract landmarks from.

3. `OUTPUT_DIR`: Where to save the extracted video objects to.

4. `MODEL_LOCATION`: Location of the MediaPipe landmarker model.

Check that `INPUT_DIR` points to the directory that contains all the videos you want to extract data from. If you used `app.extraction.utils` to copy a filtered list of videos those videos will be in the output directory.

When everything is set up, you can run the script:

```bash
$ python -m app.extraction.video_extractor
```

This will extract the landmarks from all videos, create a video object for each one and save them to the disk.

## Build CSV with Aggregated Features

