"""
Module for loading and evaluation a LightGBM classification model.
"""

import numpy as np

from lightgbm import Booster
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from app.model.utils import load_X_y, load_model


def print_predictions(
        labelEncoder: LabelEncoder,
        y_pred,
        y_proba,
        y_test,
        top_k: int):
    """
    Prints out the predictions on the test data with the corresponding probability values.
    
    :param labelEncoder: LabelEncoder holding the label <-> int pairs.
    :type labelEncoder: LabelEncoder
    :param y_pred: The model's prediction on the test data.
    :param y_proba: The probability values of the prediction.
    :param y_test: The original testdata's target column as string labels.
    :param top_k: Number of predictions to display for each prediction run.
    :type top_k: int
    """
    # decode labels
    y_pred_labels = labelEncoder.inverse_transform(y_pred)
    y_test_encoded = labelEncoder.transform(y_test)

    n = top_k
    for i in range(len(y_pred)):
        print(
            f"Prediction: {y_pred[i]} ({y_pred_labels[i]}), " # type: ignore
            f"Actual: {y_test_encoded[i]} ({y_test[i]})" # type: ignore
        )

        probs = y_proba[i] # type: ignore
        top_n_idx = np.argsort(probs)[::-1][:n] # type: ignore

        for idx in top_n_idx:
            cls_label = labelEncoder.classes_[idx]
            print(f"  Class {idx} ({cls_label}): {probs[idx]:.4f}") # type: ignore
        print()

    # print final overall accuracy
    print("Accuracy: " + str(accuracy_score(y_test_encoded, y_pred))) # type: ignore


def evaluate_model(X_test, y_actual, booster: Booster, labelEncoder: LabelEncoder) -> bool:
    """
    Runs predictions on test data X_test and compares the output to the expected target values y_actual.
    
    :param X_test: Input features X.
    :param y_actual: Target column with expected classes.
    :param booster: LightGBM Booster containing the model.
    :type booster: Booster
    :param labelEncoder: Label encoder containing the encodings for the model's class labels. 
    :type labelEncoder: LabelEncoder
    :return: True, if the prediction run was successful, False else.
    :rtype: bool
    """
    try:
        print("Running prediction...")

        # run predictions
        y_proba = booster.predict(X_test)  # (n_samples, n_classes)
        # get predicted class indices
        y_pred = np.argmax(y_proba, axis=1) # type: ignore
        
        print_predictions(
            labelEncoder,
            y_pred,
            y_proba,
            y_actual,
            top_k=3
        )

        print("Done")
        return True
    except:
        print("\n[ERROR] Failed to run the predictions")
        return False


if __name__ == "__main__":
    booster, labelEncoder = load_model("vid_small_model.txt", "vid_small_labelEncoder.pkl")

    if not (booster is None or labelEncoder is None):
        print(f"Number of classes: {len(labelEncoder.classes_)}")
        
        X, y = load_X_y("test.csv")
        evaluate_model(X, y, booster, labelEncoder)
