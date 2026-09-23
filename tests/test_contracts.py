"""Tests for src/inference/contracts.py -- status constants, JSON-safe
serialization, and schema validation. Pure software-contract tests; no model
required. Mock-free: hand-built results are the *inputs under test*, never
presented as real model output."""
import numpy as np
import pytest

from src.inference import contracts
from src.inference.contracts import (
    ContractViolationError, to_jsonable, validate_single_cell_result,
    validate_whole_smear_result,
)
from src.inference.whole_smear import CellDetection, WholeSmearResult
from src.morphology.features import MorphologyResult


# --- to_jsonable -------------------------------------------------------------

def test_to_jsonable_converts_dataclass_numpy_path_tuple():
    morph = MorphologyResult(area_px=12.0, bounding_box_wh_px=(3, 4),
                             segmentation_status="ok")
    data = to_jsonable(morph)
    assert isinstance(data, dict)
    assert data["area_px"] == 12.0
    assert data["bounding_box_wh_px"] == [3, 4]  # tuple -> list
    out = contracts.to_jsonable({
        "arr": np.array([1, 2], dtype=np.int64),
        "f": np.float32(2.5),
        "b": np.bool_(True),
    })
    assert out == {"arr": [1, 2], "f": 2.5, "b": True}


def test_to_jsonable_preserves_none_never_substitutes_placeholder():
    """None is a meaningful 'measurement unavailable' value in this codebase --
    serialization must never coerce it to 0 or ''."""
    morph = MorphologyResult(area_px=None, circularity=None)
    data = to_jsonable(morph)
    assert data["area_px"] is None
    assert data["circularity"] is None


# --- validate_single_cell_result ---------------------------------------------

def _valid_single_cell_result():
    return {
        "status": contracts.STATUS_OK,
        "prediction": {
            "status": contracts.PREDICTION_CONFIDENT,
            "predicted_class": "neutrophil",
            "top1_probability": 0.9,
            "calibrated_probabilities": {"neutrophil": 0.9, "rbc": 0.1},
        },
        "model_evidence": {"gradcam_heatmap": None, "note": "x"},
        "measured_morphology": MorphologyResult(segmentation_status="ok", area_px=5.0).to_dict(),
        "quality_reasons": [],
    }


def test_validator_accepts_wellformed_single_cell_result():
    validate_single_cell_result(_valid_single_cell_result())


def test_validator_rejects_unknown_status():
    result = _valid_single_cell_result()
    result["status"] = "SOMETHING_ELSE"
    with pytest.raises(ContractViolationError):
        validate_single_cell_result(result)


def test_validator_rejects_probabilities_not_summing_to_one():
    result = _valid_single_cell_result()
    result["prediction"]["calibrated_probabilities"] = {"a": 0.5, "b": 0.2}
    with pytest.raises(ContractViolationError, match="sum"):
        validate_single_cell_result(result)


def test_validator_rejects_confident_prediction_without_class():
    result = _valid_single_cell_result()
    result["prediction"]["predicted_class"] = None
    with pytest.raises(ContractViolationError):
        validate_single_cell_result(result)


def test_validator_rejects_low_confidence_with_class():
    result = _valid_single_cell_result()
    result["prediction"]["status"] = contracts.PREDICTION_LOW_CONFIDENCE
    with pytest.raises(ContractViolationError, match="LOW_CONFIDENCE"):
        validate_single_cell_result(result)


def test_validator_rejects_failed_segmentation_with_fake_area():
    """A failed segmentation carrying a non-None area would violate the
    no-fabrication rule -- the validator must catch it."""
    result = _valid_single_cell_result()
    result["measured_morphology"] = MorphologyResult(
        segmentation_status="failed", area_px=123.0).to_dict()
    with pytest.raises(ContractViolationError, match="None"):
        validate_single_cell_result(result)


def test_validator_accepts_quality_rejection_with_reasons():
    result = {
        "status": contracts.STATUS_IMAGE_QUALITY_REJECTED,
        "quality_reasons": ["Image appears too blurry (sharpness score 0.0)."],
        "message": "No clear blood cell could be analyzed.",
    }
    validate_single_cell_result(result)


def test_validator_rejects_quality_rejection_without_reasons():
    result = {"status": contracts.STATUS_IMAGE_QUALITY_REJECTED,
              "quality_reasons": [], "message": "x"}
    with pytest.raises(ContractViolationError):
        validate_single_cell_result(result)


# --- validate_whole_smear_result ---------------------------------------------

def _valid_whole_smear_result():
    dets = [
        CellDetection(box_xyxy=(0, 0, 10, 10), detector_class="RBC",
                      detector_confidence=0.9, final_label="rbc",
                      final_confidence=0.9, is_low_confidence=False, crop_index=0),
        CellDetection(box_xyxy=(5, 5, 20, 20), detector_class="Platelet",
                      detector_confidence=0.4, final_label="platelet",
                      final_confidence=0.4, is_low_confidence=True, crop_index=1),
    ]
    return WholeSmearResult(status=contracts.STATUS_OK, detections=dets,
                            counts={"rbc": 1, "platelet": 1})


def test_validator_accepts_wellformed_whole_smear_result():
    validate_whole_smear_result(_valid_whole_smear_result())


def test_validator_rejects_counts_not_matching_detections():
    result = _valid_whole_smear_result()
    result.counts = {"rbc": 5}
    with pytest.raises(ContractViolationError, match="counts"):
        validate_whole_smear_result(result)


def test_validator_rejects_degenerate_box():
    result = _valid_whole_smear_result()
    result.detections[0].box_xyxy = (5, 5, 5, 10)  # zero width
    with pytest.raises(ContractViolationError, match="positive area"):
        validate_whole_smear_result(result)


def test_validator_rejects_confidence_out_of_range():
    result = _valid_whole_smear_result()
    result.detections[0].final_confidence = 1.5
    with pytest.raises(ContractViolationError, match="final_confidence"):
        validate_whole_smear_result(result)


def test_validator_rejects_duplicate_crop_indices():
    result = _valid_whole_smear_result()
    result.detections[1].crop_index = 0
    with pytest.raises(ContractViolationError, match="crop_index"):
        validate_whole_smear_result(result)


def test_validator_accepts_no_cells_detected_with_message():
    result = WholeSmearResult(status=contracts.STATUS_NO_CELLS_DETECTED,
                              message="No clear blood cells were detected.")
    validate_whole_smear_result(result)


def test_validator_rejects_non_ok_status_without_message():
    result = WholeSmearResult(status=contracts.STATUS_NO_CELLS_DETECTED, message="")
    with pytest.raises(ContractViolationError, match="message"):
        validate_whole_smear_result(result)


def test_whole_smear_result_serialization_round_trip():
    result = _valid_whole_smear_result()
    data = result.to_dict()
    assert isinstance(data["detections"][0]["box_xyxy"], list)
    validate_whole_smear_result(data)  # serialized form still satisfies contract
