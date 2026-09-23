"""
Shared, cached model-loading helpers for the Streamlit app.

Caching rule (spec section 32): st.cache_resource is used ONLY for the model
object itself (the same trained weights serve every user/session -- that is
correct and required for performance). It is NEVER used to cache an inference
RESULT, because a cached result could leak from one uploaded image to another.
Every call to analyze() in app pages passes the freshly uploaded image bytes
through the pipeline fresh, keyed in st.session_state by a hash of those bytes
so that switching images never shows a stale result.

Model-state handling (engineering spec Phase 6/8): availability is reported
with an explicit reason string so pages can render a proper
"pending / not trained yet" state instead of a generic error or -- worse -- a
fabricated result.
"""
import hashlib
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
CLASSIFIER_PATH = REPO_ROOT / "models" / "classifier_v1.pt"
DETECTOR_PATH = REPO_ROOT / "models" / "detector_v1.pt"


def classifier_status() -> tuple:
    """Returns (available: bool, message: str)."""
    if CLASSIFIER_PATH.exists():
        return True, "Classifier loaded."
    return False, ("Classifier weights are not available yet "
                   f"(`{CLASSIFIER_PATH.name}` not found in `models/`). "
                   "Train it with `python scripts/train_classifier.py`.")


def detector_status() -> tuple:
    """Returns (available: bool, message: str)."""
    if DETECTOR_PATH.exists():
        return True, "Detector loaded."
    return False, ("Detector weights are not available yet "
                   f"(`{DETECTOR_PATH.name}` not found in `models/`). "
                   "Train it with `python scripts/train_detector.py`.")


@st.cache_resource(show_spinner="Loading classifier model...")
def get_single_cell_analyzer():
    from src.inference.single_cell import SingleCellAnalyzer
    return SingleCellAnalyzer(CLASSIFIER_PATH)


@st.cache_resource(show_spinner="Loading detector and classifier models...")
def get_whole_smear_analyzer():
    from src.inference.whole_smear import WholeSmearAnalyzer
    # Reuse the already-cached classifier instance instead of loading the
    # same weights into memory a second time (spec section 32).
    shared_classifier = get_single_cell_analyzer()
    return WholeSmearAnalyzer(DETECTOR_PATH, single_cell_analyzer=shared_classifier)


def classifier_available() -> bool:
    return CLASSIFIER_PATH.exists()


def detector_available() -> bool:
    return DETECTOR_PATH.exists()


def hash_image_bytes(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()
