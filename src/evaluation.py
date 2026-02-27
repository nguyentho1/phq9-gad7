"""
evaluation.py
-------------
Performance metrics and visualisations for the GAD-7 / PHQ-9
CART severity classifiers.

Includes:
  - Decision tree structure plot
  - Confusion matrix heatmap
  - Feature importance bar chart
  - One-vs-Rest ROC curves with micro-average AUC
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import LabelBinarizer


# ---------------------------------------------------------------------------
# Decision-tree visualisation
# ---------------------------------------------------------------------------

def plot_decision_tree(
    clf: DecisionTreeClassifier,
    feature_names,
    title: str = "CART Decision Tree",
    save_path: str = None,
) -> None:
    """Plot the fitted decision tree structure."""
    plt.figure(figsize=(20, 10))
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=clf.classes_,
        filled=True,
        rounded=True,
    )
    plt.title(title)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def print_classification_report(y_true, y_pred, label: str = "") -> None:
    """Print sklearn classification report."""
    header = f"Classification Report — {label}" if label else "Classification Report"
    print(f"\n{'='*60}\n{header}\n{'='*60}")
    print(classification_report(y_true, y_pred))


def plot_confusion_matrix(
    y_true,
    y_pred,
    classes,
    title: str = "Confusion Matrix",
    save_path: str = None,
) -> None:
    """Plot a labelled confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=classes,
        yticklabels=classes,
        cmap="Blues",
    )
    plt.xlabel("Predicted Severity")
    plt.ylabel("Actual Severity")
    plt.title(title)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def plot_feature_importances(
    clf: DecisionTreeClassifier,
    feature_names,
    title: str = "Feature Importances",
    save_path: str = None,
) -> None:
    """Bar chart of Gini-based feature importances."""
    importances = pd.Series(clf.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False)

    print("\nFeature Importances:\n", importances.to_string())

    plt.figure(figsize=(10, 5))
    importances.plot(kind="bar")
    plt.title(title)
    plt.ylabel("Importance (Gini)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# ROC curves (One-vs-Rest)
# ---------------------------------------------------------------------------

def plot_ovr_roc_curves(
    clf: DecisionTreeClassifier,
    X_test,
    y_test,
    title: str = "ROC Curves (One-vs-Rest)",
    save_path: str = None,
) -> None:
    """
    Compute and plot One-vs-Rest ROC curves for each severity class
    plus a micro-average ROC curve.
    """
    y_score = clf.predict_proba(X_test)

    lb = LabelBinarizer()
    y_test_bin = lb.fit_transform(y_test)
    classes = lb.classes_

    plt.figure(figsize=(10, 8))

    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {cls} (AUC = {roc_auc:.2f})")

    # Micro-average
    fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    plt.plot(
        fpr_micro,
        tpr_micro,
        label=f"Micro-average (AUC = {auc_micro:.2f})",
        linestyle=":",
        linewidth=4,
    )

    plt.plot([0, 1], [0, 1], "k--", linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
