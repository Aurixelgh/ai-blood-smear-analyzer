"""Whole-smear aggregation and robustness tests (engineering spec Phase 2
items 5/6, Phase 5).

These tests use STUB detectors/classifiers (clearly marked `_Stub*` classes)
to exercise the *software contracts* of WholeSmearAnalyzer -- aggregation,
box clamping, degenerate inputs, malformed detector output. Stubs are test
fixtures for this module's logic only; they never stand in for real model
predictions in any user-facing path. Real-model behavior is covered
separately in test_end_to_end.py (skipped until training completes).
"""
import numpy as np
import pytest

from src.errors import ModelNotAvailableError
from src.inference import contracts
from src.inference.whole_smear import (
    CellDetection,
    WholeSmearAnalyzer,
    WholeSmearResult,
    _clamp_box,
    draw_annotations,
)


class _StubBox:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = [np.array(xyxy, dtype=np.float64)]
        self.conf = [np.array(conf, dtype=np.float64)]
        self.cls = [np.array(cls, dtype=np.float64)]


class _StubYoloResult:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


class _StubDetector:
    """Stands in for ultralytics YOLO.predict() output shape. Returns
    pre-scripted boxes -- a contract fixture, NOT a model producing
    predictions."""

    def __init__(self, boxes, names={0: "WBC", 1: "RBC", 2: "Platelet"}):
        self._boxes = boxes
        self.names = names

    def predict(self, image, conf=None, verbose=False):
        return [_StubYoloResult(self._boxes, self.names)]


def _confident_prediction(cls="neutrophil", prob=0.9):
    return {
        "status": contracts.STATUS_OK,
        "prediction": {
            "status": contracts.PREDICTION_CONFIDENT,
            "predicted_class": cls,
            "top1_probability": prob,
            "calibrated_probabilities": {cls: prob},
        },
    }


class _StubSingleCellAnalyzer:
    """Stands in for SingleCellAnalyzer.analyze() with scripted results in
    the exact dict schema the real method emits."""

    def __init__(self, result=None):
        self.result = result or _confident_prediction()
        self.received_crops = []

    def analyze(self, crop):
        self.received_crops.append(crop)
        return self.result


def _make_analyzer(stub_detector, stub_classifier=None,
                   det_thr=0.25, display_thr=0.5):
    obj = WholeSmearAnalyzer.__new__(WholeSmearAnalyzer)
    obj.detector = stub_detector
    obj.detector_confidence_threshold = det_thr
    obj.low_confidence_display_threshold = display_thr
    obj.single_cell_analyzer = stub_classifier or _StubSingleCellAnalyzer()
    return obj


def _image(size=200):
    """Sharp, mid-brightness noise: passes the quality gate so tests reach
    the detection/aggregation logic under test."""
    rng = np.random.default_rng(0)
    return rng.integers(60, 200, size=(size, size, 3), dtype=np.uint8)


# --- _clamp_box ---------------------------------------------------------------

def test_clamp_box_passthrough_for_in_bounds_box():
    assert _clamp_box((10, 10, 50, 50), 200, 200) == (10, 10, 50, 50)


def test_clamp_box_clips_overshooting_coordinates():
    assert _clamp_box((150, 150, 260, 260), 200, 200) == (150, 150, 200, 200)


def test_clamp_box_rejects_fully_out_of_bounds_box():
    assert _clamp_box((300, 300, 400, 400), 200, 200) is None


def test_clamp_box_rejects_zero_area_box():
    assert _clamp_box((50, 50, 50, 100), 200, 200) is None
    assert _clamp_box((50, 50, 100, 50), 200, 200) is None


def test_clamp_box_rejects_negative_box():
    assert _clamp_box((-10, -10, -1, -1), 200, 200) is None


# --- construction / model state ------------------------------------------------

def test_missing_detector_file_raises_model_not_available():
    with pytest.raises(ModelNotAvailableError, match="does not fabricate detections"):
        WholeSmearAnalyzer("definitely_not_a_real_detector_file.pt", classifier_path=None)


def test_analyzer_requires_classifier_source(monkeypatch):
    """Neither a shared analyzer nor a classifier_path: must fail loudly, not
    later with a confusing AttributeError."""
    import src.inference.whole_smear as ws_module
    monkeypatch.setattr(ws_module, "_load_detector", lambda path: _StubDetector([]))
    with pytest.raises(ValueError, match="single_cell_analyzer or classifier_path"):
        WholeSmearAnalyzer("any_detector.pt", classifier_path=None, single_cell_analyzer=None)


# --- analyze() aggregation -------------------------------------------------------

def test_zero_detections_returns_no_cells_status():
    analyzer = _make_analyzer(_StubDetector([]))
    result = analyzer.analyze(_image())
    assert result.status == contracts.STATUS_NO_CELLS_DETECTED
    assert result.detections == []
    assert result.counts == {}
    assert result.message


def test_single_rbc_detection_counts_and_label():
    boxes = [_StubBox([10, 10, 50, 50], 0.9, 1)]  # class 1 = RBC
    analyzer = _make_analyzer(_StubDetector(boxes))
    result = analyzer.analyze(_image())
    assert result.status == contracts.STATUS_OK
    assert result.counts == {"rbc": 1}
    assert result.detections[0].final_label == "rbc"
    assert result.detections[0].is_low_confidence is False


def test_platelet_detection_maps_to_platelet_label():
    boxes = [_StubBox([10, 10, 30, 30], 0.9, 2)]  # class 2 = Platelet
    analyzer = _make_analyzer(_StubDetector(boxes))
    result = analyzer.analyze(_image())
    assert result.counts == {"platelet": 1}


def test_low_confidence_rbc_flagged_below_display_threshold():
    boxes = [_StubBox([10, 10, 50, 50], 0.3, 1)]  # passes hard filter 0.25, below display 0.5
    analyzer = _make_analyzer(_StubDetector(boxes))
    result = analyzer.analyze(_image())
    assert result.detections[0].is_low_confidence is True
    assert result.detections[0].final_confidence == 0.3


def test_wbc_resolved_via_classifier_stub():
    boxes = [_StubBox([10, 10, 60, 60], 0.9, 0)]  # class 0 = WBC
    analyzer = _make_analyzer(_StubDetector(boxes), _StubSingleCellAnalyzer(_confident_prediction("lymphocyte", 0.8)))
    result = analyzer.analyze(_image())
    assert result.counts == {"lymphocyte": 1}
    assert result.detections[0].final_label == "lymphocyte"
    assert result.detections[0].final_confidence == 0.8
    assert len(analyzer.single_cell_analyzer.received_crops) == 1


def test_wbc_low_confidence_prediction_becomes_unresolved():
    boxes = [_StubBox([10, 10, 60, 60], 0.9, 0)]
    low_conf = {
        "status": contracts.STATUS_OK,
        "prediction": {
            "status": contracts.PREDICTION_LOW_CONFIDENCE,
            "predicted_class": None,
            "top1_probability": 0.3,
            "calibrated_probabilities": {"a": 0.3, "b": 0.3, "c": 0.4},
        },
    }
    analyzer = _make_analyzer(_StubDetector(boxes), _StubSingleCellAnalyzer(low_conf))
    result = analyzer.analyze(_image())
    assert result.counts == {contracts.LABEL_WBC_UNRESOLVED: 1}
    assert result.detections[0].is_low_confidence is True


def test_wbc_quality_rejected_crop_becomes_unresolved():
    boxes = [_StubBox([10, 10, 60, 60], 0.9, 0)]
    rejected = {"status": contracts.STATUS_IMAGE_QUALITY_REJECTED, "quality_reasons": ["blurry"]}
    analyzer = _make_analyzer(_StubDetector(boxes), _StubSingleCellAnalyzer(rejected))
    result = analyzer.analyze(_image())
    assert result.counts == {contracts.LABEL_WBC_UNRESOLVED: 1}


def test_wbc_classifier_exception_degrades_to_unresolved_not_crash():
    """A per-cell classifier failure must not abort the whole smear."""
    boxes = [_StubBox([10, 10, 60, 60], 0.9, 0)]

    class _ExplodingAnalyzer:
        def analyze(self, crop):
            raise RuntimeError("simulated per-cell failure")

    analyzer = _make_analyzer(_StubDetector(boxes), _ExplodingAnalyzer())
    result = analyzer.analyze(_image())
    assert result.status == contracts.STATUS_OK
    assert result.counts == {contracts.LABEL_WBC_UNRESOLVED: 1}


def test_mixed_classes_counted_correctly():
    boxes = [
        _StubBox([10, 10, 50, 50], 0.9, 1),   # RBC
        _StubBox([60, 60, 100, 100], 0.9, 1), # RBC
        _StubBox([110, 10, 150, 50], 0.9, 2), # Platelet
        _StubBox([10, 110, 60, 160], 0.9, 0), # WBC -> neutrophil
    ]
    analyzer = _make_analyzer(_StubDetector(boxes), _StubSingleCellAnalyzer(_confident_prediction("neutrophil", 0.95)))
    result = analyzer.analyze(_image())
    assert result.counts == {"rbc": 2, "platelet": 1, "neutrophil": 1}
    assert sum(result.counts.values()) == len(result.detections) == 4


def test_out_of_bounds_boxes_skipped_not_fabricated():
    boxes = [
        _StubBox([10, 10, 50, 50], 0.9, 1),       # valid
        _StubBox([300, 300, 400, 400], 0.9, 1),   # fully outside a 200x200 image
        _StubBox([60, 60, 60, 90], 0.9, 1),       # zero width
    ]
    analyzer = _make_analyzer(_StubDetector(boxes))
    result = analyzer.analyze(_image())
    assert len(result.detections) == 1
    assert result.counts == {"rbc": 1}


def test_all_boxes_degenerate_returns_no_cells():
    boxes = [_StubBox([300, 300, 400, 400], 0.9, 1)]
    analyzer = _make_analyzer(_StubDetector(boxes))
    result = analyzer.analyze(_image())
    assert result.status == contracts.STATUS_NO_CELLS_DETECTED


def test_crop_index_preserves_original_detection_order():
    boxes = [
        _StubBox([10, 10, 50, 50], 0.9, 1),
        _StubBox([300, 300, 400, 400], 0.9, 1),  # skipped
        _StubBox([60, 60, 100, 100], 0.9, 1),
    ]
    analyzer = _make_analyzer(_StubDetector(boxes))
    result = analyzer.analyze(_image())
    assert [d.crop_index for d in result.detections] == [0, 2]


def test_quality_rejected_short_circuits_before_detection():
    flat = np.full((200, 200, 3), 128, dtype=np.uint8)  # blurry -> rejected
    analyzer = _make_analyzer(_StubDetector([_StubBox([10, 10, 50, 50], 0.9, 1)]))
    result = analyzer.analyze(flat)
    assert result.status == contracts.STATUS_IMAGE_QUALITY_REJECTED
    assert result.quality_reasons


def test_results_satisfy_contract_validator():
    boxes = [_StubBox([10, 10, 50, 50], 0.9, 1), _StubBox([60, 60, 100, 100], 0.9, 2)]
    analyzer = _make_analyzer(_StubDetector(boxes))
    result = analyzer.analyze(_image())
    contracts.validate_whole_smear_result(result)


def test_detector_predict_failure_raises_typed_inference_error():
    """A detector that crashes mid-predict must surface as a typed
    INFERENCE FAILURE with the real cause preserved -- never a silent
    empty-result success."""
    from src.errors import InferenceError

    class _ExplodingDetector:
        names = {0: "WBC", 1: "RBC", 2: "Platelet"}

        def predict(self, image, conf=None, verbose=False):
            raise RuntimeError("simulated CUDA/OOM-style detector failure")

    analyzer = _make_analyzer(_ExplodingDetector())
    with pytest.raises(InferenceError) as exc_info:
        analyzer.analyze(_image())
    assert exc_info.value.category.value == "INFERENCE_FAILURE"
    assert "simulated" in (exc_info.value.details or "")


def test_detector_returning_unknown_class_index_raises_clearly():
    """A detector emitting a class index outside its own names map is a
    contract violation -- must fail loudly, not KeyError deep in the loop."""
    boxes = [_StubBox([10, 10, 50, 50], 0.9, 99)]  # class 99 not in names
    analyzer = _make_analyzer(_StubDetector(boxes))
    with pytest.raises((KeyError, IndexError)):
        analyzer.analyze(_image())


# --- draw_annotations ------------------------------------------------------------

def test_draw_annotations_preserves_image_shape():
    boxes = [_StubBox([10, 10, 50, 50], 0.9, 1)]
    analyzer = _make_analyzer(_StubDetector(boxes))
    result = analyzer.analyze(_image())
    annotated = draw_annotations(_image(), result)
    assert annotated.shape == _image().shape


def test_draw_annotations_does_not_mutate_input():
    boxes = [_StubBox([10, 10, 50, 50], 0.9, 1)]
    analyzer = _make_analyzer(_StubDetector(boxes))
    image = _image()
    original = image.copy()
    result = analyzer.analyze(image)
    draw_annotations(image, result)
    assert np.array_equal(image, original)
