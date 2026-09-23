"""Tests for src/inference/single_cell.py: image quality checks and the
required fail-fast behavior when no trained model exists yet (spec section
35 -- never fabricate a prediction from an untrained model)."""
from pathlib import Path

import numpy as np
import pytest

from src.errors import ModelNotAvailableError
from src.inference.single_cell import check_image_quality, SingleCellAnalyzer


def test_quality_check_rejects_tiny_image():
    tiny = np.zeros((10, 10, 3), dtype=np.uint8)
    result = check_image_quality(tiny)
    assert not result.passed
    assert any("Resolution" in r for r in result.reasons)


def test_quality_check_rejects_uniform_blurry_image():
    flat = np.full((200, 200, 3), 128, dtype=np.uint8)  # perfectly flat -> zero sharpness
    result = check_image_quality(flat)
    assert not result.passed
    assert any("blurry" in r for r in result.reasons)


def test_quality_check_rejects_overexposed_image():
    bright = np.full((200, 200, 3), 255, dtype=np.uint8)
    result = check_image_quality(bright)
    assert not result.passed
    assert any("bright" in r for r in result.reasons)


def test_quality_check_passes_reasonable_image():
    rng = np.random.default_rng(0)
    varied = rng.integers(60, 200, size=(200, 200, 3), dtype=np.uint8)
    result = check_image_quality(varied)
    assert result.passed
    assert result.reasons == []


def test_analyzer_raises_clear_error_when_model_missing():
    missing_path = Path("nonexistent_model_path_for_test.pt")
    with pytest.raises(ModelNotAvailableError, match="does not fabricate predictions"):
        SingleCellAnalyzer(missing_path)


def test_missing_model_error_is_typed_as_model_state():
    """Phase 6 error taxonomy: missing weights are MODEL STATE, not a generic
    inference failure -- the UI must show a pending/trained-first state."""
    missing_path = Path("nonexistent_model_path_for_test.pt")
    with pytest.raises(ModelNotAvailableError) as exc_info:
        SingleCellAnalyzer(missing_path)
    assert exc_info.value.category.value == "MODEL_STATE"
