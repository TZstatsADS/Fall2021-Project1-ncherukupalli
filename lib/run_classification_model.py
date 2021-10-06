"""
Filename: run_classification_model.py
Author: Nikhil Cherukupalli
"""

import ast
import itertools

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.svm import SVC

from lib.build_features import get_clean_sentence, get_vectorized_sentences


def run_classification_model():
    """
        Service function. Reads/preprocesses data, vectorizes lemmatized
        sentences and builds SVM model with vectors as independent variables
        and authors as target variables. The model is trained on 80% of the
        data and tested on the remainder. Finally, the train/test splits as
        well as the model are stored to disk under '../output/'.
    """
    # Read data, build features
    path_to_data = "../data/raw_data.csv"
    df = pd.read_csv(path_to_data)
    df["lemmatized_str"] = df["tokenized_txt"].apply(
        lambda row: get_clean_sentence(ast.literal_eval(row))
    )

    # Using ordinal encoding to transform authors to numeric categories
    encoder = OrdinalEncoder()
    authors = df.loc[:, ["author"]].values
    target = encoder.fit_transform(authors)
    target = np.array(list((itertools.chain(*target))))

    # Find vectorized sentences, split data
    vectorized_sentences = get_vectorized_sentences(df["lemmatized_str"].values)
    x_train, x_test, y_train, y_test = train_test_split(vectorized_sentences,
                                                        target,
                                                        test_size=0.2,
                                                        random_state=42
                                                        )

    # Build SVM model w/ linear kernel
    clf = SVC(gamma=0.001, C=100., kernel="linear", random_state=0)
    clf.fit(x_train, y_train)

    # Store data and model to disk
    model_data_dict = {"x_train": x_train,
                       "x_test": x_test,
                       "y_train": y_train,
                       "y_test": y_test}
    joblib.dump(model_data_dict, "../output/train_test_data.dat")
    joblib.dump(clf, "../output/svm_model.sav")


def save_predicted_y_vals():
    """
        Service function. Reads test data/model from disk and
        predicts categories. Writes the predictions to disk under '../output/'.
    """
    # Read test data, trained SVM model
    train_test_dict = joblib.load("../output/train_test_data.dat")
    x_test = train_test_dict["x_test"]
    svm = joblib.load("../output/svm_model.sav")

    # Predict authors for test data, write to disk
    y_pred = svm.predict(x_test)
    joblib.dump({"y_pred": y_pred}, "../output/y_predicted.dat")


if __name__ == "__main__":
    run_classification_model()
    save_predicted_y_vals()
