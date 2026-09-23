"""
Tests for src/explainability/gradcam.py using a freshly-initialized (untrained)
model -- this exercises the real forward/backward hook mechanics and output
shape/range contract without needing the trained checkpoint, so it can run
before training completes. A brief one-shot forward+backward pass on a single
small tensor, not sustained computation.
"""
import torch

from scripts.train_classifier import build_model
from src.explainability.gradcam import GradCAM, get_default_target_layer


def test_gradcam_mobilenet_produces_valid_heatmap():
    model = build_model(num_classes=7, backbone="mobilenet_v3_small")
    target_layer = get_default_target_layer(model, "mobilenet_v3_small")
    cam = GradCAM(model, target_layer)

    input_tensor = torch.randn(1, 3, 128, 128)
    result = cam(input_tensor)

    assert result.heatmap.shape == (128, 128)
    assert result.heatmap.min() >= 0.0
    assert result.heatmap.max() <= 1.0 + 1e-5
    assert 0 <= result.predicted_class_index < 7
    cam.close()


def test_gradcam_resnet18_produces_valid_heatmap():
    model = build_model(num_classes=7, backbone="resnet18")
    target_layer = get_default_target_layer(model, "resnet18")
    cam = GradCAM(model, target_layer)

    input_tensor = torch.randn(1, 3, 128, 128)
    result = cam(input_tensor)

    assert result.heatmap.shape == (128, 128)
    assert result.heatmap.min() >= 0.0
    assert result.heatmap.max() <= 1.0 + 1e-5
    cam.close()


def test_gradcam_respects_explicit_class_index():
    model = build_model(num_classes=7, backbone="mobilenet_v3_small")
    target_layer = get_default_target_layer(model, "mobilenet_v3_small")
    cam = GradCAM(model, target_layer)

    input_tensor = torch.randn(1, 3, 128, 128)
    result = cam(input_tensor, class_index=3)
    assert result.predicted_class_index == 3
    cam.close()


def test_get_default_target_layer_rejects_unknown_backbone():
    model = build_model(num_classes=7, backbone="resnet18")
    try:
        get_default_target_layer(model, "not_a_real_backbone")
        assert False, "expected ValueError for unsupported backbone"
    except ValueError:
        pass
