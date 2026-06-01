"""Centralised configuration for the inference service.

All paths, column definitions and model hyper-parameters live here so that the
load / preprocess / score / export stages share a single source of truth.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# The service is designed to run inside a container where ./input and ./output
# are bind-mounted. Locations are overridable via environment variables so the
# same code runs both locally and in Docker.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = Path(os.environ.get("INPUT_DIR", PROJECT_ROOT / "input"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", PROJECT_ROOT / "output"))
MODELS_DIR = Path(os.environ.get("MODELS_DIR", PROJECT_ROOT / "models"))

# train.csv is only required to (re)generate model artifacts, never for inference.
TRAIN_CSV = Path(os.environ.get("TRAIN_CSV", PROJECT_ROOT / "train.csv"))

# Name of the file the service looks for inside INPUT_DIR. If it is missing the
# service falls back to the first *.csv found in the directory.
INPUT_FILENAME = os.environ.get("INPUT_FILENAME", "test.csv")

# Output artifact names.
SUBMISSION_FILENAME = "sample_submission.csv"
FEATURE_IMPORTANCE_FILENAME = "feature_importances.json"
DENSITY_PLOT_FILENAME = "prediction_density.png"

# --------------------------------------------------------------------------- #
# Data schema
# --------------------------------------------------------------------------- #
TARGET_COL = "target"
ID_COL = "_id"

# Columns expected in the raw test.csv (same order as the competition file).
RAW_FEATURE_COLS = [
    "name", "_id", "host_name", "location_cluster", "location",
    "lat", "lon", "type_house", "sum", "min_days",
    "amt_reviews", "last_dt", "avg_reviews", "total_host",
]

# Categorical columns fed natively to CatBoost.
CAT_FEATURES = ["location_cluster", "location", "type_house"]

# Columns dropped before modelling (identifiers / raw helpers).
DROP_COLS = [
    "name", "_id", "host_name", "host_uid", "last_dt", "last_dt_parsed",
    "is_train", "sum_raw", "min_days_raw",
]

# Target is clipped to the competition's valid range [0, 365].
TARGET_MIN = 0
TARGET_MAX = 365

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42
N_FOLDS = 5
K_LIST = [5, 10, 20]          # neighbours for k-NN target encoding
GEO_KS = [30, 50]             # KMeans cluster counts for geo features

# --------------------------------------------------------------------------- #
# Ensemble weights (Nelder-Mead "v3n_nm_best" — the best public LB submission).
# --------------------------------------------------------------------------- #
ENSEMBLE_WEIGHTS = {
    "CB_deep": 0.478,
    "XGB": 0.059,
    "LGB_q50": 0.045,
    "LGB_q70": 0.048,
    "Hurdle": 0.277,
    "ET": 0.092,
}

# --------------------------------------------------------------------------- #
# Artifact file names inside MODELS_DIR
# --------------------------------------------------------------------------- #
PIPELINE_FILE = "feature_pipeline.pkl"
CB_FILE = "cb_deep.cbm"
XGB_FILE = "xgb.json"
LGB_Q50_FILE = "lgb_q50.txt"
LGB_Q70_FILE = "lgb_q70.txt"
HURDLE_CLF_FILE = "hurdle_clf.txt"
HURDLE_REG_FILE = "hurdle_reg.txt"
ET_FILE = "extratrees.pkl"
META_FILE = "ensemble_meta.json"
