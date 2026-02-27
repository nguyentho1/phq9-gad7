"""
main.py
-------
End-to-end pipeline:
  1. Generate synthetic GAD-7 and PHQ-9 data
  2. Train Decision Tree classifiers
  3. Evaluate and visualise results

Usage
-----
    python main.py
"""

from src.data_generator import build_datasets
from src.model import prepare_features, train_classifier
from src.evaluation import (
    plot_decision_tree,
    print_classification_report,
    plot_confusion_matrix,
    plot_feature_importances,
    plot_ovr_roc_curves,
)


def run_pipeline(df, feature_prefix: str, label: str):
    """Run the full train-evaluate pipeline for one instrument."""
    print(f"\n{'#'*60}")
    print(f"  {label} Pipeline")
    print(f"{'#'*60}")

    # Prepare features
    drop_cols = ["Total", "Severity"]
    X, y = prepare_features(df, drop_cols)

    # Train
    clf, X_train, X_test, y_train, y_test, y_pred = train_classifier(X, y)

    # Decision tree structure
    plot_decision_tree(clf, X.columns, title=f"CART Rules — {label}")

    # Classification report
    print_classification_report(y_test, y_pred, label=label)

    # Confusion matrix
    plot_confusion_matrix(
        y_test, y_pred, clf.classes_, title=f"Confusion Matrix — {label}"
    )

    # Feature importances
    plot_feature_importances(clf, X.columns, title=f"Feature Importances — {label}")

    # ROC curves
    plot_ovr_roc_curves(clf, X_test, y_test, title=f"ROC Curves (OvR) — {label}")

    return clf, X_test, y_test


def main():
    # --- 1. Generate data ---
    print("Generating synthetic data …")
    df_gad, df_phq = build_datasets(num_observations=20585, random_seed=42)
    print(f"  GAD-7 shape: {df_gad.shape}")
    print(f"  PHQ-9 shape: {df_phq.shape}")

    # --- 2. GAD-7 pipeline ---
    run_pipeline(df_gad, feature_prefix="GAD", label="GAD-7")

    # --- 3. PHQ-9 pipeline ---
    run_pipeline(df_phq, feature_prefix="PHQ", label="PHQ-9")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
