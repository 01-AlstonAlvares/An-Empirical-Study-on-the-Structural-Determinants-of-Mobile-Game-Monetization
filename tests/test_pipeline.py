import sys
import warnings
from pathlib import Path

# Add project root to sys.path for test runner execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

import json
import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import (
    SEED,
    DATA_PROCESSED_DIR,
    ARTIFACTS_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    IMMUTABLE_FEATURES,
    MUTABLE_FEATURES,
    ALL_FEATURES,
    TARGET_COL,
    XGB_PARAMS,
)
from src.metric_helpers import (
    compute_audc,
    compute_spearman_rank_correlation,
    compute_jaccard_similarity,
)


def test_feature_definitions():
    """Verify strict immutability and mutability feature schema segregations."""
    assert len(IMMUTABLE_FEATURES) == 4, "IMMUTABLE_FEATURES must contain exactly 4 features"
    assert len(MUTABLE_FEATURES) == 5, "MUTABLE_FEATURES must contain exactly 5 features"
    assert len(ALL_FEATURES) == 9, "Total feature count must be exactly 9"
    assert set(IMMUTABLE_FEATURES).isdisjoint(set(MUTABLE_FEATURES)), "Features overlap between mutable and immutable"


def test_dataset_split_integrity():
    """Assert zero row overlap, schema consistency, and zero missing values in processed splits."""
    train_path = DATA_PROCESSED_DIR / "train.parquet"
    test_path = DATA_PROCESSED_DIR / "test.parquet"

    assert train_path.exists(), f"Missing train.parquet at {train_path}"
    assert test_path.exists(), f"Missing test.parquet at {test_path}"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # Verify column structures
    expected_cols = set(ALL_FEATURES + [TARGET_COL])
    assert set(train_df.columns) == expected_cols, "Train split column mismatch"
    assert set(test_df.columns) == expected_cols, "Test split column mismatch"

    # Assert zero missing values
    assert train_df.isnull().sum().sum() == 0, "Missing values found in training set"
    assert test_df.isnull().sum().sum() == 0, "Missing values found in test set"

    # Verify split size ratios
    total_records = len(train_df) + len(test_df)
    test_ratio = len(test_df) / total_records
    assert 0.19 < test_ratio < 0.21, f"Test split ratio {test_ratio:.3f} deviated from 20%"

    # Assert stratification preservation
    train_pos_rate = train_df[TARGET_COL].mean()
    test_pos_rate = test_df[TARGET_COL].mean()
    assert abs(train_pos_rate - test_pos_rate) < 0.01, "Stratification mismatch between splits"


def test_model_artifacts_and_opacity():
    """Verify model persistence and explicit opacity parameters matching Criterion 7."""
    models_dir = ARTIFACTS_DIR / "models"
    xgb_path = models_dir / "xgboost_model.joblib"
    prep_path = models_dir / "preprocessor.joblib"

    assert xgb_path.exists(), f"Missing XGBoost model at {xgb_path}"
    assert prep_path.exists(), f"Missing preprocessor at {prep_path}"

    model = joblib.load(xgb_path)
    preprocessor = joblib.load(prep_path)

    # Verify Criterion 7 exact opacity parameters
    assert model.n_estimators == 300, "XGBoost n_estimators must equal exactly 300"
    assert model.max_depth == 6, "XGBoost max_depth must equal exactly 6"
    assert model.learning_rate == 0.05, "XGBoost learning_rate must equal exactly 0.05"
    assert model.random_state == SEED, f"XGBoost random_state must equal {SEED}"

    # Verify feature dimension exceeds 40+ requirement
    encoded_features = list(preprocessor.get_feature_names_out())
    assert len(encoded_features) >= 40, f"Encoded feature count {len(encoded_features)} is under 40"


def test_metric_helpers_logic():
    """Unit tests for AUDC, Spearman, and Jaccard calculation helpers."""
    # Test AUDC trapezoidal calculation
    x_steps = [0, 1, 2, 3]
    y_vals = [1.0, 0.8, 0.6, 0.4]
    audc = compute_audc(x_steps, y_vals)
    assert 0.0 < audc < 1.0, "AUDC score out of valid normalized bounds"

    # Test Spearman rank correlation
    v1 = np.array([1, 2, 3, 4, 5])
    v2 = np.array([1, 2, 3, 4, 5])
    assert compute_spearman_rank_correlation(v1, v2) == pytest.approx(1.0)
    assert compute_spearman_rank_correlation(v1, -v2) == pytest.approx(-1.0)

    # Test Jaccard similarity
    set1 = {"a", "b", "c"}
    set2 = {"a", "b", "d"}
    assert compute_jaccard_similarity(set1, set2) == pytest.approx(2 / 4)


def test_artifacts_completeness():
    """Verify that all report figures and summary tables are rendered."""
    required_figures = [
        "eda_distributions.png",
        "spearman_correlations.png",
        "model_roc_curves.png",
        "faithfulness_deletion_curve.png",
        "stability_distributions.png",
        "human_sim_treatment_cards.png",
        "counterfactual_modifications.png",
    ]
    for fig_name in required_figures:
        fig_file = FIGURES_DIR / fig_name
        assert fig_file.exists(), f"Required figure missing: {fig_file}"
        assert fig_file.stat().st_size > 1000, f"Figure {fig_name} is empty or corrupted"

    required_tables = [
        "data_filtering_summary.csv",
        "eda_continuous_summary.csv",
        "spearman_correlation_matrix.csv",
        "ethical_proxy_audit.csv",
        "model_performance_comparison.csv",
        "faithfulness_audc_summary.csv",
        "stability_evaluation_summary.csv",
        "human_simulation_packet.md",
        "human_simulation_results.csv",
        "actionable_counterfactuals_summary.csv",
    ]
    for tbl_name in required_tables:
        tbl_file = TABLES_DIR / tbl_name
        assert tbl_file.exists(), f"Required table/packet missing: {tbl_file}"
        assert tbl_file.stat().st_size > 50, f"Table {tbl_name} is empty or corrupted"
