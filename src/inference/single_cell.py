"""
Single-cell inference orchestration (Mode A -- spec section 3).

Ties together, for one uploaded cell image:
  1. image quality check (spec section 25)
  2. classifier forward pass -> logits
  3. confidence-gated, calibrated prediction (src/uncertainty/calibration.py)
  4. Grad-CAM "model evidence" (src/explainability/gradcam.py)
  5. image-derived morphology (src/morphology/features.py)

The three explanation sources (model evidence / measured morphology / static
scientific reference text) are kept in separate dict keys all the way out to
the caller -- the UI layer must not merge them into one unattributed string
(spec section 8).

No result here is fabricated: if the classifier/model weights are missing,
a typed ModelNotAvailableError is raised (src/errors.py) rather than
returning placeholder numbers; if inference itself fails, an
InferenceError carries the real cause; if segmentation fails, the
morphology result reports measurement-unavailable -- never fake values.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

from src.errors import DataQualityError, InferenceError, ModelArtifactError, ModelNotAvailableError
from src.explainability.gradcam import GradCAM, get_default_target_layer
from src.inference import contracts
from src.morphology.features import compute_morphology
from src.uncertainty.calibration import classify_with_confidence_gate, DEFAULT_CONFIDENCE_THRESHOLD
from scripts.train_classifier import build_model, IMAGENET_MEAN, IMAGENET_STD

MIN_RESOLUTION_PX = 32
# Laplacian-variance blur threshold. Measured on the real held-out test split
# (datasets/processed/classification_v1/test/, 3599 images) before fixing this
# value: at the previous threshold of 15.0, 53.1% of genuine RBC images and
# 32.6% of genuine platelet images were being wrongly rejected as "too
# blurry" -- RBCs and platelets are morphologically smooth, low-texture
# objects, so they produce low Laplacian variance even when perfectly in
# focus; the check was conflating "low texture" with "out of focus". Every
# one of the 7 classes measured 0.0% below 3.0 (min observed value across all
# classes was 4.19, on rbc), so 3.0 is set with a safety margin below that
# real minimum -- it still catches genuinely degenerate/near-uniform input
# (e.g. a blank or corrupted crop) without falsely rejecting real cells of
# any supported class. This is a documented heuristic, not a clinical
# standard, and deliberately errs toward not blocking a valid upload -- the
# classifier's own confidence gate is the second line of defense against a
# genuinely poor-quality image.
BLUR_LAPLACIAN_VAR_THRESHOLD = 3.0
DARK_MEAN_BRIGHTNESS_THRESHOLD = 20.0
OVEREXPOSED_MEAN_BRIGHTNESS_THRESHOLD = 235.0


@dataclass
class QualityCheckResult:
    passed: bool
    reasons: list


def check_image_quality(bgr_image: np.ndarray) -> QualityCheckResult:
    """Evaluates basic input quality: resolution, blur, brightness. Returns
    reasons as DATA QUALITY findings, never as generic errors."""
    if bgr_image is None:
        raise DataQualityError("Image could not be decoded (empty input).")
    if bgr_image.ndim < 2 or bgr_image.shape[0] == 0 or bgr_image.shape[1] == 0:
        raise DataQualityError("Image is empty or has zero width/height.")

    reasons = []
    h, w = bgr_image.shape[:2]
    if h < MIN_RESOLUTION_PX or w < MIN_RESOLUTION_PX:
        reasons.append(f"Resolution too low ({w}x{h}px, minimum {MIN_RESOLUTION_PX}px per side).")

    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < BLUR_LAPLACIAN_VAR_THRESHOLD:
        reasons.append(f"Image appears too blurry (sharpness score {laplacian_var:.1f}, "
                        f"below the {BLUR_LAPLACIAN_VAR_THRESHOLD} threshold).")

    mean_brightness = gray.mean()
    if mean_brightness < DARK_MEAN_BRIGHTNESS_THRESHOLD:
        reasons.append(f"Image is too dark (mean brightness {mean_brightness:.1f}/255).")
    elif mean_brightness > OVEREXPOSED_MEAN_BRIGHTNESS_THRESHOLD:
        reasons.append(f"Image is too bright/overexposed (mean brightness {mean_brightness:.1f}/255).")

    return QualityCheckResult(passed=(len(reasons) == 0), reasons=reasons)


def _load_checkpoint(model_path: Path) -> dict:
    """Loads and validates the classifier checkpoint. Raises typed
    MODEL STATE errors -- a corrupt or incomplete checkpoint must never be
    silently partially used."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise ModelNotAvailableError(
            f"No trained classifier found at {model_path}. Train one first with "
            "scripts/train_classifier.py -- this system does not fabricate predictions "
            "from an untrained model."
        )
    # weights_only=True restricts unpickling to tensors/primitives (str, int,
    # list, dict) -- this checkpoint contains only those, and this avoids
    # arbitrary code execution via torch.load's default unpickler.
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ModelArtifactError(
            f"Classifier file at {model_path} could not be loaded (corrupt or "
            "incomplete download/write?). Retrain with scripts/train_classifier.py.",
            details=f"{type(exc).__name__}: {exc}",
        ) from exc

    required_keys = ("model_state_dict", "class_names", "backbone", "img_size")
    missing = [k for k in required_keys if k not in checkpoint]
    if missing:
        raise ModelArtifactError(
            f"Classifier checkpoint at {model_path} is missing required keys: {missing}. "
            "It was likely written by an older script version -- retrain with "
            "scripts/train_classifier.py.",
        )
    if not isinstance(checkpoint["class_names"], list) or not checkpoint["class_names"]:
        raise ModelArtifactError(f"Checkpoint at {model_path} has an empty/invalid class_names list.")
    return checkpoint


class SingleCellAnalyzer:
    def __init__(self, model_path: Path, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD):
        checkpoint = _load_checkpoint(model_path)
        self.model_path = model_path
        self.class_names = checkpoint["class_names"]
        self.backbone = checkpoint["backbone"]
        self.img_size = checkpoint["img_size"]
        self.confidence_threshold = confidence_threshold

        try:
            self.model = build_model(len(self.class_names), self.backbone)
            self.model.load_state_dict(checkpoint["model_state_dict"])
        except (KeyError, RuntimeError, ValueError) as exc:
            raise ModelArtifactError(
                f"Classifier weights at {model_path} do not match the architecture this "
                "code builds. Retrain with scripts/train_classifier.py.",
                details=f"{type(exc).__name__}: {exc}",
            ) from exc
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        try:
            target_layer = get_default_target_layer(self.model, self.backbone)
        except ValueError as exc:
            raise ModelArtifactError(
                f"Grad-CAM is not configured for this checkpoint's backbone ({self.backbone!r}).",
                details=str(exc),
            ) from exc
        self.gradcam = GradCAM(self.model, target_layer)

    def _forward_logits(self, input_tensor: torch.Tensor) -> torch.Tensor:
        try:
            with torch.no_grad():
                logits = self.model(input_tensor)[0]
        except Exception as exc:
            raise InferenceError(
                "The classifier failed during its forward pass on this image.",
                details=f"{type(exc).__name__}: {exc}",
            ) from exc
        if logits.ndim != 1 or logits.shape[0] != len(self.class_names):
            raise InferenceError(
                "Classifier produced logits of unexpected shape "
                f"{tuple(logits.shape)} for {len(self.class_names)} classes.",
            )
        return logits

    def _compute_gradcam(self, input_tensor: torch.Tensor, class_index: int):
        try:
            return self.gradcam(input_tensor, class_index=class_index)
        except Exception as exc:
            # Explainability is an addition to the prediction, not a reason to
            # discard a valid one -- degrade explicitly, never fabricate.
            raise InferenceError(
                "Grad-CAM model-evidence computation failed (prediction itself is unaffected).",
                details=f"{type(exc).__name__}: {exc}",
            ) from exc

    def analyze(self, bgr_image: np.ndarray, temperature: float = 1.0) -> dict:
        quality = check_image_quality(bgr_image)
        if not quality.passed:
            return {
                "status": contracts.STATUS_IMAGE_QUALITY_REJECTED,
                "quality_reasons": quality.reasons,
                "message": "No clear blood cell could be analyzed. " + " ".join(quality.reasons),
            }

        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        try:
            input_tensor = self.transform(rgb).unsqueeze(0)
        except Exception as exc:
            raise InferenceError(
                "Image preprocessing (resize/normalize) failed for this upload.",
                details=f"{type(exc).__name__}: {exc}",
            ) from exc

        logits = self._forward_logits(input_tensor)

        try:
            prediction = classify_with_confidence_gate(
                logits, self.class_names, temperature=temperature, threshold=self.confidence_threshold
            )
        except Exception as exc:
            raise InferenceError(
                "Confidence calibration failed for this prediction.",
                details=f"{type(exc).__name__}: {exc}",
            ) from exc

        gradcam_result = None
        if prediction.status == contracts.PREDICTION_CONFIDENT:
            target_idx = self.class_names.index(prediction.predicted_class)
            # A Grad-CAM failure must not destroy a valid prediction: degrade
            # to an explicit "unavailable" note instead (no fake heatmap).
            try:
                gradcam_result = self._compute_gradcam(input_tensor, target_idx)
            except InferenceError as exc:
                gradcam_result = None
                gradcam_note = f"Model evidence unavailable for this prediction. ({exc.message})"
        else:
            gradcam_note = None

        morphology = compute_morphology(bgr_image)

        model_evidence = None
        if gradcam_result is not None:
            model_evidence = {
                "gradcam_heatmap": gradcam_result.heatmap.tolist(),
                "note": "Grad-CAM shows which pixels the trained classifier weighted most heavily "
                        "for its predicted class. This is model evidence, not a direct measurement.",
            }
        elif gradcam_note:
            model_evidence = {"gradcam_heatmap": None, "note": gradcam_note}

        return {
            "status": contracts.STATUS_OK,
            "prediction": prediction.to_dict(),
            "model_evidence": model_evidence,
            # .to_dict() -- consumed via dict-style access (morph["segmentation_status"])
            # by app/pages/2_analyze_cell.py and 4_cell_explorer.py; matches the
            # to_dict() convention already used by WholeSmearResult/CellDetection/prediction
            # elsewhere in this refactor. Returning the raw dataclass here would raise
            # TypeError: 'MorphologyResult' object is not subscriptable at those call sites.
            "measured_morphology": morphology.to_dict(),
            "quality_reasons": quality.reasons,
        }
