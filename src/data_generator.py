"""
data_generator.py
-----------------
Generates synthetic GAD-7 and PHQ-9 survey response data
with a simulated latent distress trait.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Severity cutoff constants (based on clinical guidelines)
# ---------------------------------------------------------------------------

# GAD-7: 0–9 Min/Mild | 10–14 Moderate | 15–21 Severe
GAD_CUTOFFS = [-1, 9, 14, 21]
GAD_LABELS = ["Min/Mild", "Moderate", "Severe"]

# PHQ-9: 0–9 Min/Mild | 10–14 Moderate | 15–27 Severe
PHQ_CUTOFFS = [-1, 9, 14, 27]
PHQ_LABELS = ["Min/Mild", "Moderate", "Severe"]


def generate_responses(n_items: int, n_obs: int) -> pd.DataFrame:
    """
    Generate synthetic Likert-scale survey responses (0–3).

    Uses a Gamma-distributed latent trait to simulate realistic
    inter-item correlations (i.e., people with one symptom tend to
    have others).

    Parameters
    ----------
    n_items : int
        Number of survey items (7 for GAD-7, 9 for PHQ-9).
    n_obs : int
        Number of simulated respondents.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns Item_1 … Item_n and integer scores 0–3.
    """
    # Latent distress trait drawn from a Gamma distribution
    latent_trait = np.random.gamma(shape=2, scale=2, size=n_obs)
    data = {}
    for i in range(1, n_items + 1):
        # Scale trait to 0-3 range with Gaussian noise
        raw = latent_trait * 0.5 + np.random.normal(0, 0.5, n_obs)
        data[f"Item_{i}"] = np.clip(np.round(raw), 0, 3).astype(int)
    return pd.DataFrame(data)


def build_datasets(num_observations: int = 20585, random_seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the GAD-7 and PHQ-9 DataFrames with severity labels.

    Parameters
    ----------
    num_observations : int
        Number of synthetic respondents (default: 20 585).
    random_seed : int
        Seed for NumPy RNG (default: 42).

    Returns
    -------
    df_gad : pd.DataFrame
        GAD-7 responses with Total score and Severity label.
    df_phq : pd.DataFrame
        PHQ-9 responses with Total score and Severity label.
    """
    np.random.seed(random_seed)

    # --- GAD-7 ---
    df_gad = generate_responses(7, num_observations).rename(
        columns={f"Item_{i}": f"GAD{i}" for i in range(1, 8)}
    )
    df_gad["Total"] = df_gad.sum(axis=1)
    df_gad["Severity"] = pd.cut(df_gad["Total"], bins=GAD_CUTOFFS, labels=GAD_LABELS)

    # --- PHQ-9 ---
    df_phq = generate_responses(9, num_observations).rename(
        columns={f"Item_{i}": f"PHQ{i}" for i in range(1, 10)}
    )
    df_phq["Total"] = df_phq.sum(axis=1)
    df_phq["Severity"] = pd.cut(df_phq["Total"], bins=PHQ_CUTOFFS, labels=PHQ_LABELS)

    return df_gad, df_phq
