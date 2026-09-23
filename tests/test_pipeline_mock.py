"""End-to-end pipeline smoke tests using a stub classifier (engineering spec
Phase 2 item 9).

These exercise the full Mode-A flow -- quality gate -> preprocessing ->
'inference' (stub) -> confidence gate -> Grad-CAM (stub) -> morphology ->
contract validation -- to prove the ORCHESTRATION wiring is correct.

The stub clearly simulates the classifier/Grad-CAM modules only; no result
from these tests is ever presented as a real model prediction, and
real-model e2e coverage lives in test_end_to_end.py (skipped until training
completes).
"""
import cv2
import numpy as np
import pytest
import torch

from src.inference import contracts
from src.inference.contracts import validate_single_cell_result
from src.inference.single_cell import SingleCellAnalyzer, check_image_quality
from src.morphology.features import compute_morphology


class _StubModel(torch.nn.Module):
    """Deterministic logits: strong 'neutrophil' signal. A wiring fixture."""

    def __init__(self, num_classes=7):
        super().__init__()
        self.num_classes = num_classes
        self.linear = torch.nn.Identity()

    def forward(self, x):
        logits = torch.zeros(self.num_classes)
        logits[1] = 6.0  # 'neutrophil' wins decisively
        return logits.unsqueeze(0).expand(x.shape[0], -1)


class _StubGradCAM:
    def __init__(self, *a, **k):
        pass

    def __call__(self, input_tensor, class_index=None):
        from src.explainability.gradcam import GradCAMResult
        h = input_tensor.shape[2]
        w = input_tensor.shape[3]
        return GradCAMResult(heatmap=np.zeros((h, w), dtype=np.float32),
                             predicted_class_index=class_index or 0,
                             target_layer_name="StubLayer")


CLASS_NAMES = ["rbc", "neutrophil", "lymphocyte", "monocyte",
               "eosinophil", "basophil", "platelet"]


@pytest.fixture
def stub_analyzer(tmp_path):
    analyzer = SingleCellAnalyzer.__new__(SingleCellAnalyzer)
    analyzer.model = _StubModel()
    analyzer.class_names = CLASS_NAMES
    analyzer.backbone = "stub"
    analyzer.img_size = 128
    analyzer.confidence_threshold = 0.55
    analyzer.model_path = tmp_path / "unused.pt"
    analyzer.gradcam = _StubGradCAM()
    from torchvision import transforms
    from scripts.train_classifier import IMAGENET_MEAN, IMAGENET_STD
    analyzer.transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return analyzer


def _sharp_cell_image(size=160):
    img = np.full((size, size, 3), 220, dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), 45, (180, 60, 160), thickness=-1)
    rng = np.random.default_rng(1)
    noise = rng.integers(-8, 8, size=img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def test_full_pipeline_produces_confident_result_passing_contract(stub_analyzer):
    result = stub_analyzer.analyze(_sharp_cell_image())
    assert result["status"] == contracts.STATUS_OK
    assert result["prediction"]["status"] == contracts.PREDICTION_CONFIDENT
    assert result["prediction"]["predicted_class"] == "neutrophil"
    assert result["model_evidence"] is not None
    assert result["model_evidence"]["gradcam_heatmap"] is not None
    validate_single_cell_result(result)


def test_full_pipeline_quality_rejection_short_circuits(stub_analyzer):
    result = stub_analyzer.analyze(np.zeros((10, 10, 3), dtype=np.uint8))
    assert result["status"] == contracts.STATUS_IMAGE_QUALITY_REJECTED
    assert result["quality_reasons"]
    with pytest.raises(KeyError):
        result["prediction"]  # no prediction is fabricated for rejected input


def test_full_pipeline_morphology_measured_or_honestly_unavailable(stub_analyzer):
    result = stub_analyzer.analyze(_sharp_cell_image())
    morph = result["measured_morphology"]
    assert morph["segmentation_status"] == "ok"
    assert morph["area_px"] > 0
    validate_single_cell_result(result)


def test_low_confidence_result_has_no_gradcam_and_no_class(stub_analyzer, monkeypatch):
    """When the gate says LOW_CONFIDENCE, no model evidence is computed and
    no class label is asserted -- the honest 'say I don't know' path."""
    analyzer = stub_analyzer

    class _AmbiguousModel(_StubModel):
        def forward(self, x):
            return torch.zeros(7).unsqueeze(0)  # uniform -> low confidence

    analyzer.model = _AmbiguousModel()
    result = analyzer.analyze(_sharp_cell_image())
    assert result["status"] == contracts.STATUS_OK
    assert result["prediction"]["status"] == contracts.PREDICTION_LOW_CONFIDENCE
    assert result["prediction"]["predicted_class"] is None
    assert result["model_evidence"] is None
    validate_single_cell_result(result)


def test_gradcam_failure_degrades_gracefully_without_losing_prediction(stub_analyzer, monkeypatch):
    """Engineering Phase 4: explainability is an addition, never a reason to
    lose a valid prediction -- and never silently fabricated either."""

    class _BrokenGradCAM:
        def __call__(self, input_tensor, class_index=None):
            raise RuntimeError("simulated gradcam failure")

    stub_analyzer.gradcam = _BrokenGradCAM()
    result = stub_analyzer.analyze(_sharp_cell_image())
    assert result["prediction"]["status"] == contracts.PREDICTION_CONFIDENT
    assert result["prediction"]["predicted_class"] == "neutrophil"  # prediction survives
    assert result["model_evidence"]["gradcam_heatmap"] is None
    assert "unavailable" in result["model_evidence"]["note"].lower()


def test_morphology_failure_still_yields_prediction(stub_analyzer):
    """A uniform image passes the quality gate (thresholds permitting) but
    may fail segmentation -- prediction stands, morphology says unavailable."""
    result = stub_analyzer.analyze(_sharp_cell_image(size=64))
    morph = result["measured_morphology"]
    if morph["segmentation_status"] != "ok":
        assert morph["area_px"] is None
        assert morph["unavailable_reasons"]
    validate_single_cell_result(result)
