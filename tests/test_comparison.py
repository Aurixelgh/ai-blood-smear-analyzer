"""Tests for src/morphology/comparison.py -- pure data-manipulation logic,
no model or dataset dependency."""
from src.morphology.comparison import compare_measured, get_general_knowledge
from src.morphology.features import MorphologyResult


def test_compare_measured_computes_real_differences():
    result_a = MorphologyResult(area_px=100.0, circularity=0.9, segmentation_status="ok")
    result_b = MorphologyResult(area_px=60.0, circularity=0.7, segmentation_status="ok")
    comparison = compare_measured(result_a, "monocyte", result_b, "lymphocyte")
    assert comparison["features"]["area_px"]["available"] is True
    assert comparison["features"]["area_px"]["difference"] == 40.0
    assert abs(comparison["features"]["circularity"]["difference"] - 0.2) < 1e-6


def test_compare_measured_never_fabricates_missing_values():
    result_a = MorphologyResult(area_px=None, segmentation_status="failed")
    result_b = MorphologyResult(area_px=100.0, segmentation_status="ok")
    comparison = compare_measured(result_a, "a", result_b, "b")
    assert comparison["features"]["area_px"]["available"] is False
    assert comparison["features"]["area_px"]["difference"] is None
    assert comparison["features"]["area_px"]["value_a"] is None


def test_get_general_knowledge_known_pair_both_orders():
    forward = get_general_knowledge("monocyte", "lymphocyte")
    backward = get_general_knowledge("lymphocyte", "monocyte")
    assert forward is not None
    assert forward == backward


def test_get_general_knowledge_unknown_pair_returns_none_not_fabricated():
    result = get_general_knowledge("basophil", "platelet")
    assert result is None
