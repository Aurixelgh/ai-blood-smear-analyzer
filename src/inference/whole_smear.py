"""
Whole-smear inference orchestration (Mode B -- spec section 4/20).

Pipeline actually implemented:
  1. image quality check
  2. YOLO-nano detector (models/detector_v1.pt, trained by scripts/train_detector.py
     on TXL-PBC) finds cell-sized boxes and classifies each into one of 3
     coarse classes: WBC (unclassified), RBC, Platelet.
  3. RBC and Platelet detections are reported directly using the detector's
     own label -- the detector was trained specifically to distinguish these,
     and there is no WBC-subtype ambiguity to resolve for them.
  4. WBC detections are cropped and passed through the SAME 7-class classifier
     used in Mode A (src/inference/single_cell.py) to get the specific
     subtype (neutrophil/lymphocyte/monocyte/eosinophil/basophil) plus a
     calibrated confidence -- exactly as in single-cell mode -- this reuses
     one model for one job, rather than duplicating classification logic.
  5. Per-detection confidence and low-confidence flags are preserved through
     to the final per-cell record so the UI can mark ambiguous detections
     (spec section 4 point 10).

Every count in the output comes from an actual detection + (for WBCs) an
actual classifier forward pass on that specific crop -- spec section 4's
"All counts must come from actual model inference" requirement. Empty crops,
degenerate boxes, and per-cell classifier failures are skipped or flagged
explicitly -- never substituted with fabricated detections.
"""
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.errors import InferenceError, ModelArtifactError, ModelNotAvailableError
from src.inference import contracts
from src.inference.single_cell import SingleCellAnalyzer, check_image_quality


@dataclass
class CellDetection:
    box_xyxy: tuple                 # pixel coordinates in the original image
    detector_class: str             # "WBC" | "RBC" | "Platelet"
    detector_confidence: float
    final_label: str                # resolved 7-class label (or detector_class for RBC/Platelet)
    final_confidence: float
    is_low_confidence: bool
    crop_index: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class WholeSmearResult:
    status: str
    detections: list = field(default_factory=list)
    counts: dict = field(default_factory=dict)
    quality_reasons: list = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict:
        return contracts.to_jsonable(self)


def _clamp_box(box_xyxy, image_width: int, image_height: int):
    """Clamps a detector box to the image bounds and guarantees a positive
    area. YOLO boxes can slightly overshoot image borders; unclamped boxes
    would produce empty/degenerate crops downstream. Returns None for boxes
    that are fully outside or zero-area after clamping (never fabricated)."""
    x1, y1, x2, y2 = (int(round(float(v))) for v in box_xyxy)
    # Reject boxes that do not intersect the image at all BEFORE clamping --
    # otherwise a far-outside box collapses to a 1px corner sliver.
    if x2 <= 0 or y2 <= 0 or x1 >= image_width or y1 >= image_height:
        return None
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(x2, image_width), min(y2, image_height)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _box_iou(box_a, box_b) -> float:
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


def _suppress_duplicate_boxes(boxes_info: list, iou_threshold: float = 0.65) -> list:
    """Application-layer safety net on top of the detector's own internal
    per-class NMS (Ultralytics default iou=0.7).

    Real finding from this project's own end-to-end testing (see
    reports/methodology_report.md): on a real, dense held-out test image,
    two same-class (both RBC) boxes with raw-coordinate mutual IoU
    measuring right at Ultralytics' 0.7 NMS cutoff (0.700, and 0.706 after
    the pixel-rounding _clamp_box applies downstream) both survived
    internal NMS -- a coordinate-space rounding effect between the model's
    internal 416x416 inference resolution and the original image's pixel
    coordinates. Left alone, this double-counts one physical cell. The
    threshold here is set to 0.65, with margin below that observed
    boundary case, so this safety net reliably catches it (and anything
    closer to a true duplicate) even after downstream integer rounding
    shifts the measured IoU by a few hundredths.

    This pass explicitly keeps only the highest-confidence box among any
    same-class group whose mutual IoU exceeds `iou_threshold`. Two
    genuinely distinct touching/overlapping cells almost always have much
    lower mutual box IoU than this (their boxes only share the touching
    edge, not near-total overlap), so this does not suppress real adjacent
    detections -- it only catches near-identical duplicates of the same
    class. Never invents or drops a class count silently: every remaining
    box is still a real detection at its own real confidence."""
    boxes_info = sorted(boxes_info, key=lambda b: b["conf"], reverse=True)
    kept = []
    for candidate in boxes_info:
        is_duplicate = any(
            candidate["cls"] == kept_box["cls"] and _box_iou(candidate["xyxy"], kept_box["xyxy"]) > iou_threshold
            for kept_box in kept
        )
        if not is_duplicate:
            kept.append(candidate)
    return kept


def _load_detector(detector_path: Path):
    detector_path = Path(detector_path)
    if not detector_path.exists():
        raise ModelNotAvailableError(
            f"No trained detector found at {detector_path}. Train one first with "
            "scripts/train_detector.py -- this system does not fabricate detections."
        )
    from ultralytics import YOLO
    try:
        return YOLO(str(detector_path))
    except Exception as exc:
        raise ModelArtifactError(
            f"Detector file at {detector_path} could not be loaded (corrupt or "
            "incompatible with the installed ultralytics version?).",
            details=f"{type(exc).__name__}: {exc}",
        ) from exc


class WholeSmearAnalyzer:
    def __init__(self, detector_path: Path, classifier_path: Path = None,
                 detector_confidence_threshold: float = 0.25,
                 low_confidence_display_threshold: float = 0.5,
                 single_cell_analyzer: "SingleCellAnalyzer" = None):
        """Pass an existing `single_cell_analyzer` (e.g. the one already
        cached by app/utils/model_access.get_single_cell_analyzer) to avoid
        loading the same classifier weights into memory twice -- spec
        section 32's "optimize... memory consumption" requirement. If not
        given, classifier_path is required and a new instance is built.

        `detector_confidence_threshold` is the HARD filter passed to YOLO's
        predict() -- boxes below it are discarded before this class ever sees
        them. `low_confidence_display_threshold` is a separate, higher bar
        used only to flag a detection that survived the hard filter but is
        still not very confident (spec section 4 point 10's "identify
        uncertain/ambiguous detections"). These must stay two different
        numbers: comparing a detection's confidence back against the same
        hard filter it already passed can never be true, which would
        silently disable ambiguous-detection flagging for RBC/Platelet.
        """
        self.detector = _load_detector(detector_path)
        self.detector_confidence_threshold = detector_confidence_threshold
        self.low_confidence_display_threshold = low_confidence_display_threshold

        if single_cell_analyzer is not None:
            self.single_cell_analyzer = single_cell_analyzer
        elif classifier_path is not None:
            self.single_cell_analyzer = SingleCellAnalyzer(classifier_path)
        else:
            raise ValueError("Provide either single_cell_analyzer or classifier_path.")

    def _detect(self, bgr_image: np.ndarray):
        try:
            return self.detector.predict(
                bgr_image, conf=self.detector_confidence_threshold, verbose=False
            )[0]
        except Exception as exc:
            raise InferenceError(
                "The cell detector failed on this image.",
                details=f"{type(exc).__name__}: {exc}",
            ) from exc

    def _classify_wbc_crop(self, crop: np.ndarray, det_conf: float):
        """Resolves a WBC crop to a specific subtype via the shared classifier.
        Returns (final_label, final_confidence, is_low_confidence). A per-cell
        classifier failure degrades to wbc_unresolved with a low-confidence
        flag -- it must never abort the whole smear or fabricate a label."""
        try:
            cell_result = self.single_cell_analyzer.analyze(crop)
        except Exception:
            # Detector said WBC but the classifier could not process the crop;
            # the detection itself is real, the subtype is honestly unknown.
            return contracts.LABEL_WBC_UNRESOLVED, det_conf, True

        if cell_result["status"] == contracts.STATUS_IMAGE_QUALITY_REJECTED:
            return contracts.LABEL_WBC_UNRESOLVED, det_conf, True
        pred = cell_result["prediction"]
        if pred["status"] == contracts.PREDICTION_LOW_CONFIDENCE:
            return contracts.LABEL_WBC_UNRESOLVED, pred["top1_probability"], True
        return pred["predicted_class"], pred["top1_probability"], False

    def analyze(self, bgr_image: np.ndarray) -> WholeSmearResult:
        quality = check_image_quality(bgr_image)
        if not quality.passed:
            return WholeSmearResult(
                status=contracts.STATUS_IMAGE_QUALITY_REJECTED,
                quality_reasons=quality.reasons,
                message="No clear blood cells were detected. " + " ".join(quality.reasons),
            )

        image_h, image_w = bgr_image.shape[:2]
        yolo_results = self._detect(bgr_image)

        if len(yolo_results.boxes) == 0:
            return WholeSmearResult(
                status=contracts.STATUS_NO_CELLS_DETECTED,
                message="No clear blood cells were detected. Try a sharper microscope "
                        "image with better illumination and appropriate magnification.",
            )

        class_names = yolo_results.names  # {0: 'WBC', 1: 'RBC', 2: 'Platelet'}
        raw_boxes = [
            {"cls": int(box.cls[0].item()), "conf": float(box.conf[0].item()),
             "xyxy": tuple(float(v) for v in box.xyxy[0].tolist())}
            for box in yolo_results.boxes
        ]
        deduped_boxes = _suppress_duplicate_boxes(raw_boxes)

        detections = []
        counts = {}

        for i, box in enumerate(deduped_boxes):
            det_conf = box["conf"]
            det_class = class_names[box["cls"]]

            clamped = _clamp_box(box["xyxy"], image_w, image_h)
            if clamped is None:
                continue  # degenerate/out-of-bounds box: skip, never fabricate
            x1, y1, x2, y2 = clamped

            if det_class in ("RBC", "Platelet"):
                final_label = "rbc" if det_class == "RBC" else "platelet"
                final_conf = det_conf
                is_low_conf = det_conf < self.low_confidence_display_threshold
            else:  # WBC -- resolve subtype via the 7-class classifier
                crop = bgr_image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                final_label, final_conf, is_low_conf = self._classify_wbc_crop(crop, det_conf)

            detections.append(CellDetection(
                box_xyxy=(x1, y1, x2, y2), detector_class=det_class, detector_confidence=det_conf,
                final_label=final_label, final_confidence=final_conf, is_low_confidence=is_low_conf,
                crop_index=i,
            ))
            counts[final_label] = counts.get(final_label, 0) + 1

        if not detections:
            return WholeSmearResult(
                status=contracts.STATUS_NO_CELLS_DETECTED,
                message="The detector returned boxes but none were usable after "
                        "clipping to the image bounds. Try a different image.",
            )

        return WholeSmearResult(status=contracts.STATUS_OK, detections=detections, counts=counts)


def draw_annotations(bgr_image: np.ndarray, result: WholeSmearResult) -> np.ndarray:
    """Draws boxes + compact labels. Labels are kept small and offset so they
    do not obscure dense fields of cells (spec section 5 requirement)."""
    annotated = bgr_image.copy()
    color_map = {
        "rbc": (60, 60, 220), "platelet": (200, 200, 60), contracts.LABEL_WBC_UNRESOLVED: (128, 128, 128),
    }
    for det in result.detections:
        x1, y1, x2, y2 = det.box_xyxy
        color = color_map.get(det.final_label, (60, 200, 60))
        thickness = 1 if det.is_low_confidence else 2  # confident boxes drawn bolder
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        label_text = det.final_label[:3].upper()
        if det.is_low_confidence:
            label_text += "?"
        cv2.putText(annotated, label_text, (x1, max(0, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return annotated
