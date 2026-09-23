"""Expanded uncertainty tests (engineering spec Phase 2 item 4): high/low
confidence, temperature scaling behavior, and malformed probability/logit
inputs. These exercise the calibration math directly -- no model needed."""
import numpy as np
import pytest
import torch

from src.uncertainty.calibration import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    TemperatureScaler,
    classify_with_confidence_gate,
)


def test_uniform_logits_give_low_confidence():
    result = classify_with_confidence_gate(torch.zeros(7), list("abcdefg"), threshold=0.55)
    assert result.status == "LOW_CONFIDENCE"
    assert result.predicted_class is None
    assert abs(result.top1_probability - 1 / 7) < 1e-5


def test_extreme_logits_give_near_one_confidence():
    logits = torch.tensor([50.0, 0.0, 0.0])
    result = classify_with_confidence_gate(logits, ["a", "b", "c"], threshold=0.55)
    assert result.status == "CONFIDENT"
    assert result.top1_probability > 0.999


def test_probabilities_sum_to_one_regardless_of_temperature():
    logits = torch.tensor([2.0, -1.0, 0.5, 3.0])
    for temp in (0.5, 1.0, 2.0, 10.0):
        result = classify_with_confidence_gate(logits, list("abcd"), temperature=temp)
        total = sum(result.calibrated_probabilities.values())
        assert abs(total - 1.0) < 1e-5, f"temperature={temp} broke normalization"


def test_temperature_must_be_positive():
    """A zero/negative temperature would divide logits by 0 or flip their
    sign -- both silently corrupt every downstream confidence number. The
    gate now guards this explicitly (engineering fix from this test run)."""
    logits = torch.tensor([1.0, 2.0])
    with pytest.raises(ValueError, match="temperature"):
        classify_with_confidence_gate(logits, ["a", "b"], temperature=0.0)
    with pytest.raises(ValueError, match="temperature"):
        classify_with_confidence_gate(logits, ["a", "b"], temperature=-1.0)


def test_threshold_boundary_is_inclusive():
    """top1 exactly at threshold counts as confident (>= semantics)."""
    # craft logits whose softmax top1 is exactly 0.55 for 2 classes:
    # p = e^x / (e^x + e^0) = 0.55 -> x = ln(0.55/0.45)
    x = float(np.log(0.55 / 0.45))
    logits = torch.tensor([x, 0.0])
    result = classify_with_confidence_gate(logits, ["a", "b"], threshold=0.55)
    assert result.status == "CONFIDENT"


def test_class_count_mismatch_raises_clearly():
    """If logits and class_names disagree, the gate must fail loudly rather
    than index out of bounds or silently drop classes."""
    logits = torch.tensor([1.0, 2.0, 3.0])
    with pytest.raises((IndexError, KeyError, RuntimeError)):
        classify_with_confidence_gate(logits, ["only_one_class"])


def test_temperature_scaler_temperature_property_tracks_parameter():
    scaler = TemperatureScaler()
    assert abs(scaler.temperature - 1.0) < 1e-6
    with torch.no_grad():
        scaler.log_temperature.fill_(float(np.log(2.0)))
    assert abs(scaler.temperature - 2.0) < 1e-6


def test_temperature_scaler_forward_divides_logits():
    scaler = TemperatureScaler()
    with torch.no_grad():
        scaler.log_temperature.fill_(float(np.log(4.0)))
    out = scaler(torch.tensor([8.0, 4.0]))
    assert torch.allclose(out, torch.tensor([2.0, 1.0]))


def test_temperature_scaler_fit_softens_wrong_but_confident_predictions():
    """When the model is confidently WRONG on the fitting data, NLL
    minimization must raise T > 1 (soften). Note: perfectly-classified
    overconfident logits legitimately keep T=1 -- NLL is already minimal
    there, which is why the fit data must contain errors for T to move."""
    scaler = TemperatureScaler()
    logits = torch.tensor([[20.0, -20.0], [-20.0, 20.0]])
    wrong_labels = torch.tensor([1, 0])
    temp = scaler.fit(logits, wrong_labels)
    assert temp > 1.0


def test_temperature_scaler_fit_keeps_temperature_near_one_when_well_calibrated():
    scaler = TemperatureScaler()
    labels = torch.tensor([0, 1, 0, 1])
    overconfident = torch.tensor([[20.0, -20.0], [-20.0, 20.0], [20.0, -20.0], [-20.0, 20.0]])
    temp = scaler.fit(overconfident, labels)
    assert abs(temp - 1.0) < 0.5  # no error signal -> no meaningful softening


def test_default_threshold_is_documented_constant():
    assert 0.0 < DEFAULT_CONFIDENCE_THRESHOLD < 1.0
