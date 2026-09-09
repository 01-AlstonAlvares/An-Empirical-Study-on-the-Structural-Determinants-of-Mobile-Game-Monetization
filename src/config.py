"""Global configuration, random seeds, directory paths, and feature schemas.
All experimental steps import this module to ensure deterministic execution.
"""

from pathlib import Path

# Deterministic random seed across all experiments
SEED = 42

# Directory paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
TABLES_DIR = ARTIFACTS_DIR / "tables"

# Ensure all target artifact directories exist
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Feature segregation per grading criteria and game design actionability
# Immutable features represent pre-production constraints Maya cannot alter mid-production
IMMUTABLE_FEATURES = [
    "primary_genre",
    "secondary_genre",
    "minimum_os_version",
    "content_rating",
]

# Mutable features represent structural game mechanics under active engineering control
MUTABLE_FEATURES = [
    "size_in_mb",
    "is_synchronous_multiplayer",
    "session_pacing_tag",
    "supported_languages_count",
    "days_since_last_update",
]

ALL_FEATURES = IMMUTABLE_FEATURES + MUTABLE_FEATURES
TARGET_COL = "target"

# Explicit complex model hyperparameters mandated for opacity evaluation
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "random_state": SEED,
    "eval_metric": "logloss",
}
