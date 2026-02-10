"""
Utility module for the AI pipeline.
"""

import joblib
import pandas as pd
from lightgbm import Booster
from sklearn.preprocessing import LabelEncoder


def load_model(model_file: str, encoder_file: str):
    """
    Docstring for load_model
    
    :param model_file: Path to a saved model file (.txt).
    :type model_file: str
    :param encoder_file: Path to a saved label encoder file (.pkl).
    :type encoder_file: str
    :return: A LightGBM Booster and a LabelEncoder if successfull, tuple[None, None] else.
    :rtype: tuple[Booster, LabelEncoder]
    """
    print("Loading model and label encoder...", end=" ")
    booster: Booster = Booster(model_file=model_file)
    labelEncoder: LabelEncoder = joblib.load(encoder_file)
    print("Done")
    return booster, labelEncoder


def load_X_y(csv_file: str, print_X_y=False):
    """
    Reads data from a CSV file into a DataFrame and returns the input features as X and target column as y.
    
    :param csv_file: Path to the CSV file.
    :type csv_file: str
    :param print_X_y: Print the input features and target column.
    :type print_X_y: bool
    :return: A tuple containing the input features X and target column y or None if no file was found.
    :rtype: tuple[DataFrame, Series[Any]] | tuple[None, None]
    """
    try:
        df = pd.read_csv(csv_file, delimiter=",")
        
        X: pd.DataFrame = df.drop("label", axis=1)
        y: pd.Series = df["label"]
        
        if print_X_y:
            print("INPUT")
            print(X)
            print()
            print("TARGET")
            print(y)
            print()

        return X, y
    except FileNotFoundError:
        print(f"[ERROR] Could not find a file with the name {csv_file}")
        return None, None
