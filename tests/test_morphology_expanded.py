"""Expanded morphology edge-case tests (engineering spec Phase 2 item 2):
tiny regions, impossible/negative measurements, missing nucleus, partial
measurements, and the guarantee that failures report None + reason rather
than fabricated values."""
import cv2
import numpy as np

from src.morphology.features import (
    MorphologyResult, _segment_cell_mask, compute_morphology,
)


def _cell_image(radius=40, size=120, color=(180, 60, 160)):
    img = np.full((size, size, 3), 220, dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), radius, color, thickness=-1)
    return img


def test_segmentation_returns_none_for_blank_image():
    assert _segment_cell_mask(np.full((100, 100, 3), 128, dtype=np.uint8)) is None


def test_segmentation_rejects_near_whole_image_foreground():
    """A mask covering essentially the entire image is a failed segmentation
    (background misidentified), not a giant cell."""
    img = np.full((100, 100, 3), 30, dtype=np.uint8)  # saturated dark everywhere
    cv2.rectangle(img, (0, 0), (99, 99), (200, 50, 200), thickness=-1)
    assert _segment_cell_mask(img) is None


def test_tiny_region_reports_failure_or_unavailable_nucleus_not_fake_values():
    tiny = _cell_image(radius=2, size=20)
    result = compute_morphology(tiny, attempt_nucleus_split=True)
    if result.segmentation_status == "ok":
        # nucleus split must be skipped with a reason, never a fabricated ratio
        assert result.nucleus_to_cell_ratio is None
        assert any("nucleus" in r for r in result.unavailable_reasons)
    else:
        assert result.area_px is None


def test_measurements_are_never_negative_where_impossible():
    img = _cell_image(radius=30)
    result = compute_morphology(img)
    for name in ("area_px", "perimeter_px", "equivalent_diameter_px", "circularity",
                 "aspect_ratio", "solidity", "extent", "eccentricity",
                 "intensity_std", "edge_density", "nucleus_to_cell_ratio"):
        value = getattr(result, name)
        if value is not None:
            assert value >= 0, f"{name} must never be negative, got {value}"


def test_ratios_within_mathematically_valid_ranges():
    img = _cell_image(radius=35)
    result = compute_morphology(img)
    if result.circularity is not None:
        assert 0.0 < result.circularity <= 1.05  # circle = 1, allow tiny numeric slack
    if result.solidity is not None:
        assert 0.0 <= result.solidity <= 1.0
    if result.extent is not None:
        assert 0.0 <= result.extent <= 1.0
    if result.eccentricity is not None:
        assert 0.0 <= result.eccentricity < 1.0
    if result.nucleus_to_cell_ratio is not None:
        assert 0.0 <= result.nucleus_to_cell_ratio <= 1.0


def test_partial_measurements_carry_reasons():
    """When some features are unavailable, unavailable_reasons must explain
    which and why -- silent partial results are not acceptable."""
    result = compute_morphology(_cell_image(radius=3, size=20), attempt_nucleus_split=True)
    if result.segmentation_status == "ok":
        missing = [f for f in ("circularity", "eccentricity", "nucleus_to_cell_ratio")
                   if getattr(result, f) is None]
        if missing:
            assert result.unavailable_reasons


def test_default_result_is_not_attempted_with_no_measurements():
    result = MorphologyResult()
    assert result.segmentation_status == "not_attempted"
    assert result.area_px is None
    assert result.unavailable_reasons == []


def test_to_dict_preserves_none_measurements():
    result = MorphologyResult(segmentation_status="failed")
    data = result.to_dict()
    assert data["area_px"] is None
    assert data["segmentation_status"] == "failed"


def test_nucleus_split_flag_respected():
    img = _cell_image(radius=40)
    result = compute_morphology(img, attempt_nucleus_split=False)
    assert result.nucleus_area_px is None
    assert result.nucleus_to_cell_ratio is None


def test_color_stats_present_only_on_successful_segmentation():
    ok = compute_morphology(_cell_image(radius=40), attempt_nucleus_split=False)
    assert ok.mean_rgb is not None and len(ok.mean_rgb) == 3
    failed = compute_morphology(np.full((100, 100, 3), 128, dtype=np.uint8))
    assert failed.mean_rgb is None
