# How to Train and Run the Classification Model

*Package: app.model*

The model package handles the training and evaluation of the LightGBM classification model and includes a pre-trained and ready-to-use model. 

## Model Training

*app.model.training*

Once you have obtained the training data from the `extraction` package as a CSV file, the training and saving is very simple. The `training` module contains one setup variable `TRAIN_DATA`, which needs to point to your training dataset in the form of a CSV file. Then simply run the module:

```bash
$ python -m app.model.training
```

This will load your training data from the CSV file, train the model, and save it in the `app.model` package. For optimal results you might need to adjust the hyperparameters in the code.

> **Note:** The utility function `load_X_y` expects the training data to contain a column named `label` that contains the label for each sample. You dataset **must** contain this column for the training to work!

## Model Evaluation

*app.model.evaluation*

TBD