"""Tests for src/uncertainty/calibration.py."""
import torch

from src.uncertainty.calibration import classify_with_confidence_gate, TemperatureScaler


def test_confident_prediction_above_threshold():
    class_names = ["a", "b", "c"]
    logits = torch.tensor([5.0, 0.1, 0.1])  # strongly peaked -> high softmax confidence
    result = classify_with_confidence_gate(logits, class_names, temperature=1.0, threshold=0.55)
    assert result.status == "CONFIDENT"
    assert result.predicted_class == "a"
    assert result.top1_probability > 0.55


def test_low_confidence_prediction_below_threshold():
    class_names = ["a", "b", "c"]
    logits = torch.tensor([0.1, 0.05, 0.0])  # near-uniform -> low softmax confidence
    result = classify_with_confidence_gate(logits, class_names, temperature=1.0, threshold=0.55)
    assert result.status == "LOW_CONFIDENCE"
    assert result.predicted_class is None
    assert result.message is not None
    # probabilities must still be reported for all classes even when gated
    assert set(result.calibrated_probabilities.keys()) == set(class_names)


def test_higher_temperature_reduces_confidence():
    class_names = ["a", "b", "c"]
    logits = torch.tensor([5.0, 0.1, 0.1])
    low_temp = classify_with_confidence_gate(logits, class_names, temperature=1.0, threshold=0.0)
    high_temp = classify_with_confidence_gate(logits, class_names, temperature=5.0, threshold=0.0)
    assert high_temp.top1_probability < low_temp.top1_probability


def test_temperature_scaler_fits_without_error():
    scaler = TemperatureScaler()
    val_logits = torch.randn(50, 4)
    val_labels = torch.randint(0, 4, (50,))
    temperature = scaler.fit(val_logits, val_labels)
    assert temperature > 0
