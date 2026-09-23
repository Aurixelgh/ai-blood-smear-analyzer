"""
Inference data contracts (engineering spec Phase 3).

Single source of truth for:
  1. Status strings shared by every pipeline stage (quality check, single-cell
     analysis, whole-smear analysis) -- previously scattered as string
     literals across src/ and app/pages/, which made typos silent and
     downstream consumers fragile.
  2. JSON-safe serialization of the dataclass results
     (MorphologyResult, CellDetection, WholeSmearResult, ...) so the Streamlit
     layer, exports, and tests all consume one predictable schema.
  3. Schema validators that assert an analysis result actually matches its
     contract -- exercised by the test suite to catch pipeline regressions.

Contract invariants (the no-fabrication rule, spec sections 8/35):
  - A `None` measurement stays `None` in serialized output -- it is never
    replaced by 0, "", or a placeholder.
  - Statuses are exact strings from this module; anything else is a bug.
"""
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# --- Pipeline stage statuses -------------------------------------------------

STATUS_OK = "OK"
STATUS_IMAGE_QUALITY_REJECTED = "IMAGE_QUALITY_REJECTED"
STATUS_NO_CELLS_DETECTED = "NO_CELLS_DETECTED"
STATUS_MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"

SINGLE_CELL_STATUSES = (STATUS_OK, STATUS_IMAGE_QUALITY_REJECTED, STATUS_MODEL_UNAVAILABLE)
WHOLE_SMEAR_STATUSES = (STATUS_OK, STATUS_IMAGE_QUALITY_REJECTED, STATUS_NO_CELLS_DETECTED, STATUS_MODEL_UNAVAILABLE)

# --- Classifier prediction statuses (src/uncertainty/calibration.py) ---------

PREDICTION_CONFIDENT = "CONFIDENT"
PREDICTION_LOW_CONFIDENCE = "LOW_CONFIDENCE"
PREDICTION_STATUSES = (PREDICTION_CONFIDENT, PREDICTION_LOW_CONFIDENCE)

# --- Segmentation statuses (src/morphology/features.py) ----------------------

SEGMENTATION_OK = "ok"
SEGMENTATION_FAILED = "failed"
SEGMENTATION_NOT_ATTEMPTED = "not_attempted"
SEGMENTATION_STATUSES = (SEGMENTATION_OK, SEGMENTATION_FAILED, SEGMENTATION_NOT_ATTEMPTED)

# --- Label used when a WBC subtype could not be resolved ---------------------

LABEL_WBC_UNRESOLVED = "wbc_unresolved"


class ContractViolationError(ValueError):
    """Raised when an analysis result does not match its declared schema."""


def to_jsonable(obj):
    """Recursively convert dataclasses / numpy / Path / tuples into plain
    JSON-serializable Python. None stays None -- never substituted."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- Schema validators --------------------------------------------------------

def _require(condition: bool, message: str):
    if not condition:
        raise ContractViolationError(message)


def validate_single_cell_result(result: dict) -> None:
    """Assert the dict returned by SingleCellAnalyzer.analyze matches its
    contract. Raises ContractViolationError otherwise."""
    _require(isinstance(result, dict), "single-cell result must be a dict")
    status = result.get("status")
    _require(status in SINGLE_CELL_STATUSES, f"unknown single-cell status: {status!r}")

    if status == STATUS_IMAGE_QUALITY_REJECTED:
        _require(isinstance(result.get("quality_reasons"), list) and result["quality_reasons"],
                 "rejected result must carry non-empty quality_reasons")
        _require(isinstance(result.get("message"), str) and result["message"],
                 "rejected result must carry a user-facing message")
        return

    pred = result.get("prediction")
    _require(isinstance(pred, dict), "non-rejected result must carry a prediction dict")
    _require(pred.get("status") in PREDICTION_STATUSES, f"unknown prediction status: {pred.get('status')!r}")
    probs = pred.get("calibrated_probabilities")
    _require(isinstance(probs, dict) and probs, "prediction must carry non-empty calibrated_probabilities")
    _require(all(isinstance(p, (int, float)) and 0.0 <= p <= 1.0 for p in probs.values()),
             "probabilities must be numbers in [0, 1]")
    _require(abs(sum(probs.values()) - 1.0) < 1e-3, "probabilities must sum to ~1.0")
    if pred["status"] == PREDICTION_CONFIDENT:
        _require(pred.get("predicted_class") in probs,
                 "confident prediction's class must be one of the probability keys")
    else:
        _require(pred.get("predicted_class") is None,
                 "LOW_CONFIDENCE prediction must not carry a predicted_class")

    morph = result.get("measured_morphology")
    _require(isinstance(morph, dict), "result must carry measured_morphology")
    _require(morph.get("segmentation_status") in SEGMENTATION_STATUSES,
             f"unknown segmentation_status: {morph.get('segmentation_status')!r}")
    if morph["segmentation_status"] != SEGMENTATION_OK:
        _require(morph.get("area_px") is None,
                 "failed segmentation must report None measurements, never placeholders")


def validate_whole_smear_result(result) -> None:
    """Assert a WholeSmearResult matches its contract (works on the dataclass
    or its serialized dict form)."""
    data = to_jsonable(result) if not isinstance(result, dict) else result
    _require(isinstance(data, dict), "whole-smear result must be a dict or dataclass")
    status = data.get("status")
    _require(status in WHOLE_SMEAR_STATUSES, f"unknown whole-smear status: {status!r}")

    if status != STATUS_OK:
        _require(isinstance(data.get("message"), str) and data["message"],
                 "non-OK whole-smear result must carry a user-facing message")
        return

    detections = data.get("detections")
    _require(isinstance(detections, list), "OK result must carry a detections list")
    counts = data.get("counts", {})
    _require(isinstance(counts, dict), "counts must be a dict")
    _require(sum(counts.values()) == len(detections),
             "counts must be derived from actual detections (sum(counts) == len(detections))")

    seen_crop_indices = set()
    for det in detections:
        _require(isinstance(det, dict), "each detection must serialize to a dict")
        box = det.get("box_xyxy")
        _require(isinstance(box, list) and len(box) == 4, "detection box_xyxy must have 4 coordinates")
        x1, y1, x2, y2 = box
        _require(x2 > x1 and y2 > y1, f"detection box must have positive area, got {box}")
        _require(isinstance(det.get("final_confidence"), (int, float))
                 and 0.0 <= det["final_confidence"] <= 1.0,
                 "final_confidence must be a number in [0, 1]")
        _require(isinstance(det.get("detector_confidence"), (int, float))
                 and 0.0 <= det["detector_confidence"] <= 1.0,
                 "detector_confidence must be a number in [0, 1]")
        _require(isinstance(det.get("is_low_confidence"), bool), "is_low_confidence must be a bool")
        idx = det.get("crop_index")
        _require(isinstance(idx, int) and idx not in seen_crop_indices,
                 "crop_index must be present and unique per detection")
        seen_crop_indices.add(idx)
