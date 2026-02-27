A Python project that replicates CART (Classification and Regression Tree) decision rules for classifying depression (PHQ-9) and anxiety (GAD-7) severity using the `scikit-learn` `DecisionTreeClassifier`.

## Overview

This project generates synthetic survey response data for the **PHQ-9** (Patient Health Questionnaire-9) and **GAD-7** (Generalized Anxiety Disorder-7) instruments, trains separate Decision Tree classifiers to predict severity (Min/Mild, Moderate, Severe), and evaluates them with standard metrics and visualisations.

The CART hyperparameters replicate those described in the original paper:
- `min_samples_leaf = 500` (minbucket)
- `min_samples_split = 1000` (minsplit)
- `criterion = 'gini'`

 ## Severity Cutoffs

| Instrument | Min/Mild | Moderate | Severe |
|------------|----------|----------|--------|
| GAD-7      | 0 – 9    | 10 – 14  | 15 – 21 |
| PHQ-9      | 0 – 9    | 10 – 14  | 15 – 27 |

## Outputs

| Output | Description |
|--------|-------------|
| Decision tree plot | Visual CART rules for each instrument |
| Classification report | Precision / Recall / F1 per severity class |
| Confusion matrix | Heatmap of true vs. predicted severity |
| Feature importances | Gini-based ranking of survey items |
| ROC curves | One-vs-Rest AUC for each class + micro-average |

---

## Dependencies

- `pandas` — data manipulation  
- `numpy` — numerical operations  
- `scikit-learn` — machine learning  
- `matplotlib` — plotting  
- `seaborn` — statistical visualisations  
