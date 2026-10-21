import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from pathlib import Path


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


def generate_model(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = clean_and_split(df)

    model = DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=42)

    model.fit(X_train, y_train)

    return model, X_test, y_test


"""def test_model(model, X_test, y_test, save=True):
    SAVE_PATH = Path("results", "evaluation")
    y_pred_raw = model.predict(X_test)
    y_pred = np.clip(np.round(y_pred_raw), 0, 2).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    accuracy = np.diag(cm).sum() / cm.sum()
    precision = []
    recall = []
    f1 = []
    for i in range(len(cm[0])):
        column = cm[:, i]
        row = cm[i]

        tp = column[i]
        fp = column.sum() - tp
        fn = row.sum() - tp

        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0

        precision.append(p)
        recall.append(r)
        f1.append(f)

    percentage_format = lambda x: f"{np.round(x * 100, 2)}%"
    results = pd.DataFrame(
        {
            "Precision": map(percentage_format, precision),
            "Recall": map(percentage_format, recall),
            "F1 Score": map(percentage_format, f1),
        },
        index=[0, 1, 2],
    )

    macro = pd.DataFrame(
        {
            "Mean Accuracy": percentage_format(accuracy),
            "Macro Precision": percentage_format(np.mean(precision)),
            "Macro Recall": percentage_format(np.mean(recall)),
            "Macro F1 Score": percentage_format(np.mean(f1)),
        },
        index=[0],
    )

    if save:
        results.to_excel(Path(SAVE_PATH, "classiwise_results.xlsx"))
        macro.to_excel(Path(SAVE_PATH, "average_results.xlsx"))
    else:
        return results, macro"""


def test_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    df = pd.DataFrame(
        classification_report(y_test, predictions, output_dict=True)
    ).transpose()
    df.drop(["support"], axis=1, inplace=True)
    df.drop(["accuracy", "macro avg", "weighted avg"], inplace=True)
    df.index = ['Class "Low"', 'Class "Medium"', 'Class "High"']
    return df
