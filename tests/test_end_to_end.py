"""
End-to-end tests against the REAL trained models and REAL sample images
(copied from held-out test splits into samples/, see scripts that produced
them). These are skipped automatically if a model has not been trained yet
in this environment -- they never fabricate a pass by mocking the model.

Covers the full flow required by spec section 41:
  upload -> quality check -> (detect ->) crop -> classify -> confidence ->
  morphology -> explanation -> counts/annotation
"""
from pathlib import Path

import cv2
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CLASSIFIER_PATH = REPO_ROOT / "models" / "classifier_v1.pt"
DETECTOR_PATH = REPO_ROOT / "models" / "detector_v1.pt"
SAMPLES_SINGLE_CELL = REPO_ROOT / "samples" / "single_cell"
SAMPLES_WHOLE_SMEAR = REPO_ROOT / "samples" / "whole_smear"

requires_classifier = pytest.mark.skipif(
    not CLASSIFIER_PATH.exists(), reason="classifier_v1.pt not trained yet in this environment"
)
requires_detector = pytest.mark.skipif(
    not DETECTOR_PATH.exists(), reason="detector_v1.pt not trained yet in this environment"
)


@requires_classifier
@pytest.mark.parametrize("expected_label", [
    "neutrophil", "lymphocyte", "monocyte", "eosinophil", "basophil", "platelet", "rbc",
])
def test_single_cell_pipeline_end_to_end_on_real_holdout_image(expected_label):
    """Runs the full Mode-A pipeline on a REAL held-out test-set image for
    each of the 7 classes and asserts the pipeline completes and returns a
    self-consistent, non-fabricated result. Does NOT assert the prediction
    equals expected_label -- that would conflate a smoke test with a proper
    accuracy claim, which belongs in evaluation/classifier_v1_evaluation.json
    (real measured test-set metrics), not in this pass/fail unit test.

    Contract note: `result["status"]` is the PIPELINE status (OK /
    IMAGE_QUALITY_REJECTED / ...); the classifier's own CONFIDENT /
    LOW_CONFIDENCE verdict is nested at `result["prediction"]["status"]` --
    see src/inference/contracts.py, which formalized this split. Validated
    against that module's own schema checker, not just ad-hoc assertions."""
    from src.inference import contracts
    from src.inference.single_cell import SingleCellAnalyzer

    image_path = SAMPLES_SINGLE_CELL / f"{expected_label}_example.jpg"
    assert image_path.exists(), f"Sample image missing for {expected_label} -- run the sample-copy step first."

    analyzer = SingleCellAnalyzer(CLASSIFIER_PATH)
    bgr_image = cv2.imread(str(image_path))
    assert bgr_image is not None

    result = analyzer.analyze(bgr_image)
    contracts.validate_single_cell_result(result)

    assert result["status"] == contracts.STATUS_OK
    assert "measured_morphology" in result
    prediction = result["prediction"]
    assert prediction["status"] in contracts.PREDICTION_STATUSES
    probs = prediction["calibrated_probabilities"]
    assert set(probs.keys()) == set(analyzer.class_names)
    assert abs(sum(probs.values()) - 1.0) < 1e-3  # a real softmax distribution, not fabricated numbers

    if prediction["status"] == contracts.PREDICTION_CONFIDENT:
        assert prediction["predicted_class"] in analyzer.class_names
        assert result["model_evidence"] is not None
        assert result["model_evidence"]["gradcam_heatmap"] is not None


def _box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@requires_classifier
@requires_detector
@pytest.mark.parametrize("sample_name", [
    "example_smear.png", "example_smear_sparse.png",
    "example_smear_medium.png", "example_smear_dense.png",
])
def test_whole_smear_pipeline_end_to_end_on_real_image(sample_name):
    """Full Mode-B pipeline (detect -> crop -> classify -> count -> annotate)
    on real TXL-PBC test-split whole-smear images spanning sparse (5 GT
    cells), medium (14), and dense (28) real fields -- not just one sample,
    so the pipeline is exercised across the real density range this project
    actually has evidence for."""
    from src.inference import contracts
    from src.inference.whole_smear import WholeSmearAnalyzer, draw_annotations

    image_path = SAMPLES_WHOLE_SMEAR / sample_name
    assert image_path.exists()

    analyzer = WholeSmearAnalyzer(DETECTOR_PATH, CLASSIFIER_PATH)
    bgr_image = cv2.imread(str(image_path))
    assert bgr_image is not None

    result = analyzer.analyze(bgr_image)
    contracts.validate_whole_smear_result(result)
    assert result.status in contracts.WHOLE_SMEAR_STATUSES

    if result.status == contracts.STATUS_OK:
        assert len(result.detections) == sum(result.counts.values())
        annotated = draw_annotations(bgr_image, result)
        assert annotated.shape == bgr_image.shape

        # No duplicated/double-counted detections: YOLO's own NMS should
        # already prevent two boxes from covering the same physical cell,
        # but verify it directly on the actual output rather than trusting
        # that internal behavior blindly -- two detections with very high
        # mutual IoU would mean the same cell got counted twice.
        boxes = [d.box_xyxy for d in result.detections]
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                assert _box_iou(boxes[i], boxes[j]) < 0.7, \
                    f"detections {i} and {j} overlap too heavily ({_box_iou(boxes[i], boxes[j]):.2f} IoU) -- possible duplicate/double-counted cell"

        for det in result.detections:
            assert 0.0 <= det.final_confidence <= 1.0
            # Regression check for the fixed bug where RBC/Platelet
            # low-confidence flagging compared a detection's confidence back
            # against the same hard filter threshold it had already passed,
            # making the flag permanently unreachable. Every detection here
            # already cleared analyzer.detector_confidence_threshold (the
            # hard filter), so a flagged RBC/Platelet detection's confidence
            # must sit strictly below the separate, higher display threshold.
            if det.detector_class in ("RBC", "Platelet"):
                assert det.detector_confidence >= analyzer.detector_confidence_threshold
                if det.is_low_confidence:
                    assert det.detector_confidence < analyzer.low_confidence_display_threshold
