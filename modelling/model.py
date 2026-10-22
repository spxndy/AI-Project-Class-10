import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import numpy as np
from pathlib import Path


# Creates a training and testing dataset, selects features based on correlation with the target variable, and returns the split datasets.
def clean_and_split(df: pd.DataFrame):
    features = df.drop("stress_level", axis=1)  # features
    target = df["stress_level"]  # target

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    threshold = 0.5
    corr = X_train.corrwith(y_train)

    selected_features = corr[corr.abs() > threshold].index

    X_train, X_test = (
        X_train[selected_features.to_list()],
        X_test[selected_features.to_list()],
    )

    return X_train, X_test, y_train, y_test


# Generates a Decision Tree Classifier model, fits it to the training data, and returns the model along with the test datasets.
def generate_model(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = clean_and_split(df)

    model = DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=42)

    model.fit(X_train, y_train)

    return model, X_test, y_test


# Evaluates the model's performance on the test dataset and returns a DataFrame containing precision, recall, and F1-score for each class.
def test_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    df = pd.DataFrame(
        classification_report(y_test, predictions, output_dict=True)
    ).transpose()
    df.drop(["support"], axis=1, inplace=True)
    df.drop(["accuracy", "macro avg", "weighted avg"], inplace=True)
    df.index = ['Class "Low"', 'Class "Medium"', 'Class "High"']
    return df
