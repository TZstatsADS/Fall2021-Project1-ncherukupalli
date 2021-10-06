"""
Filename: run_classificaiton_model.py
Author: Nikhil Cherukupalli
"""

import ast
import pandas as pd
import joblib
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
import itertools

from lib.build_features import get_clean_sentence, get_vectorized_sentences


def run_classification_model():
    print("Reading data...")
    # Read data, build features
    PATH_TO_DATA = "../data/raw_data.csv"
    df = pd.read_csv(PATH_TO_DATA)
    df["lemmatized_str"] = df["tokenized_txt"].apply(
        lambda row: get_clean_sentence(ast.literal_eval(row))
    )

    print("Encoding y values...")
    # Using ordinal encoding to transform authors --> numeric categories
    encoder = OrdinalEncoder()
    target = df.loc[:, ["author"]].values
    y = encoder.fit_transform(target)
    y = np.array(list((itertools.chain(*y))))

    print("Vectorizing sentences...")
    # Find vectorized sentences, split data
    vectorized_sentences = get_vectorized_sentences(df["lemmatized_str"].values)
    x_train, x_test, y_train, y_test = train_test_split(
        vectorized_sentences,
        y,
        test_size=0.2,
        random_state=42
    )

    print("Training model...")
    # Build SVM model
    clf = SVC(gamma=0.001, C=100., kernel="linear", random_state=0)
    clf.fit(x_train, y_train)

    print("Storing results...")
    # Store data and model to disk
    model_data_dict = {"x_train": x_train,
                       "x_test": x_test,
                       "y_train": y_train,
                       "y_test": y_test}
    joblib.dump(model_data_dict, "../output/train_test_data.dat")
    joblib.dump(clf, "../output/svm_model.sav")


if __name__ == "__main__":
    run_classification_model()
