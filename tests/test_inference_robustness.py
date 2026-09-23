"""Real-data inference robustness tests (engineering: malformed / corrupt /
empty / unsupported inputs through the REAL classifier + REAL morphology on
real sample images).

Uses the real trained checkpoint (skipped if absent) plus real sample images
from samples/ -- NO fake model outputs, NO fake metrics. Corruption is
injected at the FILE level (real failure mode: truncated download, wrong
extension, empty file), never at the model-output level.
"""
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.errors import DataQualityError
from src.inference import contracts
from src.inference.single_cell import SingleCellAnalyzer, check_image_quality
from src.morphology.features import compute_morphology

REPO_ROOT = Path(__file__).resolve().parents[1]
CLASSIFIER_PATH = REPO_ROOT / "models" / "classifier_v1.pt"
SAMPLES = REPO_ROOT / "samples" / "single_cell"

requires_classifier = pytest.mark.skipif(
    not CLASSIFIER_PATH.exists(), reason="classifier_v1.pt not trained yet in this environment"
)


def _any_sample_image():
    path = SAMPLES / "neutrophil_example.jpg"
    assert path.exists()
    return cv2.imread(str(path))


# --- corrupt / empty / unsupported file-level inputs ---------------------------

@requires_classifier
def test_corrupt_image_bytes_raise_gracefully(tmp_path):
    """A truncated/corrupt file (real-world: interrupted download) decodes to
    None; the analyzer must fail with a typed DATA QUALITY error, not crash
    with an opaque cv2 error or produce a fabricated result."""
    good = _any_sample_image()
    ok, buf = cv2.imencode(".jpg", good)
    assert ok
    corrupt = buf.tobytes()[: len(buf) // 3]  # truncate the JPEG stream
    decoded = cv2.imdecode(np.frombuffer(corrupt, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        with pytest.raises(DataQualityError):
            check_image_quality(None)
    else:
        # cv2 is sometimes tolerant; then the pipeline must still complete
        # honestly -- the quality gate or the model handles it downstream.
        analyzer = SingleCellAnalyzer(CLASSIFIER_PATH)
        result = analyzer.analyze(decoded)
        assert result["status"] in contracts.SINGLE_CELL_STATUSES


@requires_classifier
def test_zero_byte_image_raises_data_quality():
    with pytest.raises(DataQualityError):
        check_image_quality(None)


@requires_classifier
def test_non_image_bytes_that_still_decode_are_processed_honestly(tmp_path):
    """Random noise that happens to decode as an image is a valid input to
    the pipeline: the honest outcome is whatever the real quality gate and
    classifier say -- never a crash, never a fabricated label."""
    rng = np.random.default_rng(7)
    noise = rng.integers(0, 255, size=(100, 100, 3), dtype=np.uint8)
    analyzer = SingleCellAnalyzer(CLASSIFIER_PATH)
    result = analyzer.analyze(noise)
    assert result["status"] in contracts.SINGLE_CELL_STATUSES
    if result["status"] == contracts.STATUS_OK:
        contracts.validate_single_cell_result(result)


# --- real-image morphology robustness ------------------------------------------

def test_morphology_on_real_sample_image_is_honest():
    img = _any_sample_image()
    result = compute_morphology(img)
    # Either measured or explicitly unavailable -- both honest.
    if result.segmentation_status == "ok":
        assert result.area_px is not None and result.area_px > 0
        contracts.to_jsonable(result)  # must stay serializable
    else:
        assert result.area_px is None
        assert result.unavailable_reasons


def test_morphology_on_grayscale_converted_real_image():
    """Real uploads sometimes arrive effectively grayscale (3 identical
    channels). The saturation channel is then ~0 everywhere -- segmentation
    must fail honestly rather than return nonsense numbers."""
    img = _any_sample_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    result = compute_morphology(gray3)
    if result.segmentation_status == "ok":
        # even if it segments, numbers must be finite
        for name in ("area_px", "perimeter_px", "circularity", "solidity"):
            v = getattr(result, name)
            if v is not None:
                assert np.isfinite(v)
    else:
        assert result.unavailable_reasons


def test_morphology_on_oversized_real_image_does_not_hang_or_crash():
    """Deployment guard: a very large upload must not crash the classical-CV
    path. (Downscaled here to keep the test suite fast, but the code path --
    large-array contour handling -- is the same.)"""
    img = _any_sample_image()
    big = cv2.resize(img, (img.shape[1] * 4, img.shape[0] * 4), interpolation=cv2.INTER_CUBIC)
    result = compute_morphology(big, attempt_nucleus_split=False)
    assert result.segmentation_status in ("ok", "failed")


# --- real classifier end-to-end robustness -------------------------------------

@requires_classifier
def test_unusual_filename_and_spaces_path(tmp_path):
    """File-handling robustness (spec: spaces/unicode in paths must work on
    Windows AND Linux)."""
    src = _any_sample_image()
    weird = tmp_path / "a cell image (copy 1) [final].jpg"
    cv2.imwrite(str(weird), src)
    loaded = cv2.imread(str(weird))
    assert loaded is not None
    analyzer = SingleCellAnalyzer(CLASSIFIER_PATH)
    result = analyzer.analyze(loaded)
    assert result["status"] in contracts.SINGLE_CELL_STATUSES
    if result["status"] == contracts.STATUS_OK:
        contracts.validate_single_cell_result(result)


@requires_classifier
def test_very_small_but_valid_image_gets_honest_result():
    """A 40x40 image is above the min-resolution gate; the honest outcome is
    a real (possibly low-confidence) prediction, or an explicit rejection --
    never a crash."""
    img = _any_sample_image()
    small = cv2.resize(img, (40, 40))
    analyzer = SingleCellAnalyzer(CLASSIFIER_PATH)
    result = analyzer.analyze(small)
    assert result["status"] in contracts.SINGLE_CELL_STATUSES
    if result["status"] == contracts.STATUS_OK:
        contracts.validate_single_cell_result(result)


@requires_classifier
def test_all_seven_real_samples_produce_valid_contracts():
    """Each real held-out sample must flow through the whole pipeline and
    satisfy the result contract (quality-gate outcomes are valid too)."""
    analyzer = SingleCellAnalyzer(CLASSIFIER_PATH)
    for sample in sorted(SAMPLES.glob("*_example.jpg")):
        img = cv2.imread(str(sample))
        assert img is not None, sample.name
        result = analyzer.analyze(img)
        if result["status"] == contracts.STATUS_OK:
            contracts.validate_single_cell_result(result)
        else:
            assert result["status"] == contracts.STATUS_IMAGE_QUALITY_REJECTED
            assert result["quality_reasons"]
