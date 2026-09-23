"""Tests for src/morphology/features.py -- uses synthetic images so these
tests do not depend on any downloaded dataset or trained model being present."""
import numpy as np
import cv2
import pytest

from src.morphology.features import compute_morphology


def _synthetic_cell_image(radius=40, size=120, saturated=True):
    """A colored circle on a low-saturation background -- mimics a stained
    cell against a lighter background well enough for the Otsu-on-saturation
    segmentation to succeed."""
    img = np.full((size, size, 3), 220, dtype=np.uint8)  # light, low-saturation background
    center = (size // 2, size // 2)
    color = (180, 60, 160) if saturated else (215, 210, 218)  # BGR, saturated purple-ish vs near-background
    cv2.circle(img, center, radius, color, thickness=-1)
    return img


def test_compute_morphology_on_clean_circle_succeeds():
    img = _synthetic_cell_image(radius=40)
    result = compute_morphology(img, attempt_nucleus_split=False)
    assert result.segmentation_status == "ok"
    assert result.area_px is not None and result.area_px > 0
    # A circle should have circularity close to 1.0
    assert result.circularity is not None
    assert 0.7 < result.circularity <= 1.05
    assert result.aspect_ratio is not None
    assert result.aspect_ratio < 1.3  # near-circular bounding box


def test_compute_morphology_reports_failure_not_fabricated_values():
    blank = np.full((100, 100, 3), 128, dtype=np.uint8)  # no cell at all, flat color
    result = compute_morphology(blank, attempt_nucleus_split=False)
    assert result.segmentation_status == "failed"
    assert result.area_px is None
    assert result.circularity is None
    assert len(result.unavailable_reasons) > 0


def test_compute_morphology_never_returns_negative_area():
    img = _synthetic_cell_image(radius=25)
    result = compute_morphology(img)
    if result.area_px is not None:
        assert result.area_px >= 0
    if result.perimeter_px is not None:
        assert result.perimeter_px >= 0


def test_nucleus_split_skips_gracefully_on_tiny_region():
    tiny = _synthetic_cell_image(radius=3, size=20)  # too small for a stable 2-cluster fit
    result = compute_morphology(tiny, attempt_nucleus_split=True)
    # Either segmentation itself fails (region too small/degenerate) or the
    # nucleus split specifically reports why it was skipped -- never a fabricated ratio.
    if result.segmentation_status == "ok":
        assert result.nucleus_to_cell_ratio is None
        assert any("nucleus" in r for r in result.unavailable_reasons)
