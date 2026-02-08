"""
This module loads the entire asl dataset (100.000+ samples), trains a model and evaluates it.
"""
from sklearn.model_selection import train_test_split

import inference
from utils import load_X_y
from training import train_model


def train_and_evaluate():
    X_train, y_train = load_X_y("table_knuckles+tips.csv")
    X_test, y_test = load_X_y("table_knuckles+tips_test.csv")
    
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    MODEL_NAME = "2k_v1"

    if train_model(X_train, y_train, MODEL_NAME):
        booster, labelEncoder = inference.load_model(f"{MODEL_NAME}_model.txt", f"{MODEL_NAME}_labelEncoder.pkl")

        if not (booster is None or labelEncoder is None):
            print(f"Number of classes: {len(labelEncoder.classes_)}")
            
            inference.evaluate_model(X_test, y_test, booster, labelEncoder)


if __name__ == "__main__":
    pass
