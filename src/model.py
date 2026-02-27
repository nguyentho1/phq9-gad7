"""
model.py
--------
Trains Decision Tree classifiers for GAD-7 and PHQ-9 severity
classification using CART with hyperparameters matching the paper
(minbucket = 500 → min_samples_leaf, minsplit = 1000 → min_samples_split).
"""

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd


# Shared hyperparameters replicating the paper's CART configuration
CART_PARAMS = dict(
    criterion="gini",
    min_samples_leaf=500,   # minbucket=500 in original paper
    min_samples_split=1000,
    random_state=42,
)


def prepare_features(df: pd.DataFrame, drop_cols: list[str]) -> tuple:
    """
    Split a DataFrame into feature matrix X and target vector y.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset including features and target.
    drop_cols : list[str]
        Columns to exclude from the feature matrix (e.g. ['Total', 'Severity']).

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    X = df.drop(columns=drop_cols)
    y = df["Severity"]
    return X, y


def train_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.3,
) -> tuple:
    """
    Split data and train a CART Decision Tree classifier.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target severity labels.
    test_size : float
        Proportion of data reserved for testing (default: 0.30).

    Returns
    -------
    clf : DecisionTreeClassifier
        Fitted classifier.
    X_train, X_test, y_train, y_test : splits
    y_pred : predicted labels on the test set
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    clf = DecisionTreeClassifier(**CART_PARAMS)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return clf, X_train, X_test, y_train, y_test, y_pred
