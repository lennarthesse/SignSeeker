"""
Module for training and saving the classification model.
"""
import joblib
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

from app.model.utils import load_X_y


TRAIN_DATA = "app/extraction/output/table.csv"


def train_model(X_train, y_train, model_name: str ="") -> bool:
    """
    Trains a LightGBM classification model on the given data X and y and saves it under the provided model name.
    
    :param X_train: Input features X.
    :param y_train: Target column y.
    :param model_name: The name under which to save the model to the disk.
    :type model_name: str
    :return: True, if the model was trained and saved successfully, False else.
    :rtype: bool
    """
    labelEncoder = LabelEncoder()

    # encode the label strings as numbers
    y_train = labelEncoder.fit_transform(y_train)

    print(f"Number of classes: {len(labelEncoder.classes_)}")

    # initialize model
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(labelEncoder.classes_),

        n_estimators=1000,          # more data -> more trees help
        learning_rate=0.03,

        num_leaves=16,              # allow richer interactions
        max_depth=5,

        min_data_in_leaf=30,        # matches the per-class minimum
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,

        min_split_gain=0.1,
        n_jobs=-1,                  # use all available cores
        #verbosity=-1,               # suppress warnings and infos (consider deactivating this)
        random_state=42
        )

    # train model
    try:
        print("Training the model...", end=" ")

        model.fit(X_train, y_train)
        
        print("Done")
    except Exception as e:
        print(f"\n[ERROR] An error occurred during model training: {e}")
        return False

    # save model
    try:
        print("Saving the model...", end=" ")
        model.booster_.save_model(f"{model_name}_model.txt")
        joblib.dump(labelEncoder, f"{model_name}_labelEncoder.pkl")
        print("Done")
    except:
        print("\n[ERROR] An error occurred while saving the model")
        return False

    print("Successfully trained the model")
    return True


if __name__ == "__main__":
    X, y = load_X_y(TRAIN_DATA)
    train_model(X, y, "unit1")
