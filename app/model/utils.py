"""
Utility module for the AI pipeline.
"""
import pandas as pd


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
