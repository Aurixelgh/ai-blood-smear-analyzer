"""
Image-derived morphology engine (spec section 7).

Computes ONLY features that are actually measurable from the pixels of a
single-cell crop via classical computer vision -- no learned model, no
biological ground truth. Every returned feature is either a numeric value
computed from a real segmentation of this specific image, or explicitly
`None` with a `reason` string when it could not be reliably computed. The
caller (UI layer) must never substitute a placeholder value for `None` --
that would violate the project's no-fabrication rule (spec section 35).

Two feature groups:
  1. Whole-cell shape/color/texture features (segmentation via Otsu
     thresholding on the HSV saturation channel -- blood-stain preparations
     reliably separate stained cell from lighter background on this channel).
  2. WBC-specific nucleus/cytoplasm split (k-means on masked-pixel hue/value --
     nuclei stain darker/more saturated purple-blue than cytoplasm). This
     split is a classical-CV heuristic, NOT validated against expert ground
     truth in this codebase yet (see reports/architecture_decision.md section 6
     for why: only Raabin-WBC's 1,145 masks exist publicly, and they are not
     yet downloaded/wired into an automated IoU/Dice validation script). Until
     that validation script exists and reports real IoU/Dice, nucleus/cytoplasm
     outputs are labeled "unvalidated_proxy" in their result dict -- the UI
     must surface that label, not hide it.
"""
from dataclasses import asdict, dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass
class MorphologyResult:
    # Shape features (None + reason if not computable)
    area_px: Optional[float] = None
    perimeter_px: Optional[float] = None
    equivalent_diameter_px: Optional[float] = None
    circularity: Optional[float] = None
    aspect_ratio: Optional[float] = None
    solidity: Optional[float] = None
    extent: Optional[float] = None
    eccentricity: Optional[float] = None
    bounding_box_wh_px: Optional[tuple] = None

    # Color statistics (RGB + HSV means/stds within the segmented cell mask)
    mean_rgb: Optional[tuple] = None
    std_rgb: Optional[tuple] = None
    mean_hsv: Optional[tuple] = None
    std_hsv: Optional[tuple] = None

    # Texture (simple, honestly-labeled proxies -- not a validated texture model)
    intensity_std: Optional[float] = None
    edge_density: Optional[float] = None

    # Nucleus/cytoplasm split (WBC only, unvalidated proxy -- see module docstring)
    nucleus_area_px: Optional[float] = None
    cytoplasm_area_px: Optional[float] = None
    nucleus_to_cell_ratio: Optional[float] = None

    segmentation_status: str = "not_attempted"
    unavailable_reasons: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Plain-dict form of this result for the UI/serialization layer.
        None stays None -- never substituted by a placeholder value."""
        return asdict(self)


def _segment_cell_mask(bgr_image: np.ndarray) -> Optional[np.ndarray]:
    """Otsu threshold on the HSV saturation channel + morphological cleanup.
    Returns a binary mask of the single largest foreground contour, or None
    if no plausible cell-sized contour is found."""
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    _, mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = bgr_image.shape[:2]
    image_area = h * w
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    # Guard against degenerate segmentations (near-zero or near-whole-image).
    # The 2% floor was picked by intuition and, when measured against the
    # real held-out test set (datasets/processed/classification_v1/test/,
    # 100 images/class), turned out to reject 94% of genuine platelet images
    # -- platelets are real, small blood-cell fragments (median ~1.1% of a
    # 360x363 crop; min observed 0.428%), not a segmentation failure. No
    # other class showed any meaningful area below 2% (all medians >4%,
    # worst 5th percentile 3.37% on neutrophil). 0.3% keeps a margin below
    # platelet's observed minimum while still rejecting true degenerate
    # segmentations (RBC showed ~1-2% of images below even 0.5%, which is
    # the expected rate of genuinely unreliable segmentation this guard
    # exists to catch honestly rather than fabricate a measurement for).
    if area < 0.003 * image_area or area > 0.98 * image_area:
        return None

    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, [largest], -1, 255, thickness=cv2.FILLED)
    return clean_mask


def compute_morphology(bgr_image: np.ndarray, attempt_nucleus_split: bool = True) -> MorphologyResult:
    result = MorphologyResult()

    mask = _segment_cell_mask(bgr_image)
    if mask is None:
        result.segmentation_status = "failed"
        result.unavailable_reasons.append(
            "Could not reliably segment the cell from background (image quality, "
            "staining, or framing may be atypical). All shape/color features unavailable."
        )
        return result
    result.segmentation_status = "ok"

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, closed=True)
    result.area_px = float(area)
    result.perimeter_px = float(perimeter)
    result.equivalent_diameter_px = float(np.sqrt(4 * area / np.pi))

    if perimeter > 0:
        result.circularity = float(4 * np.pi * area / (perimeter ** 2))
    else:
        result.unavailable_reasons.append("circularity: zero perimeter")

    x, y, w, h = cv2.boundingRect(contour)
    result.bounding_box_wh_px = (int(w), int(h))
    result.aspect_ratio = float(max(w, h) / max(min(w, h), 1))
    result.extent = float(area / (w * h)) if w * h > 0 else None

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    result.solidity = float(area / hull_area) if hull_area > 0 else None

    if len(contour) >= 5:
        (_, _), (major, minor), _ = cv2.fitEllipse(contour)
        major, minor = max(major, minor), min(major, minor)
        if major > 0:
            ratio = (minor / major) ** 2
            result.eccentricity = float(np.sqrt(max(0.0, 1 - ratio)))
    else:
        result.unavailable_reasons.append("eccentricity: too few contour points to fit an ellipse")

    # Color statistics within the mask
    mask_bool = mask.astype(bool)
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    result.mean_rgb = tuple(float(v) for v in rgb[mask_bool].mean(axis=0))
    result.std_rgb = tuple(float(v) for v in rgb[mask_bool].std(axis=0))
    result.mean_hsv = tuple(float(v) for v in hsv[mask_bool].mean(axis=0))
    result.std_hsv = tuple(float(v) for v in hsv[mask_bool].std(axis=0))

    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    result.intensity_std = float(gray[mask_bool].std())
    edges = cv2.Canny(gray, 50, 150)
    result.edge_density = float((edges[mask_bool] > 0).mean())

    if attempt_nucleus_split:
        _attempt_nucleus_cytoplasm_split(bgr_image, mask, result)

    return result


def _attempt_nucleus_cytoplasm_split(bgr_image: np.ndarray, cell_mask: np.ndarray, result: MorphologyResult):
    """K-means (k=2) on masked-pixel HSV to separate darker/more-saturated
    nucleus from lighter cytoplasm. Labeled unvalidated_proxy -- see module
    docstring. Skips (leaves nucleus fields as None) if the cell mask is too
    small for a stable 2-cluster fit."""
    mask_bool = cell_mask.astype(bool)
    n_pixels = int(mask_bool.sum())
    if n_pixels < 200:
        result.unavailable_reasons.append(
            "nucleus/cytoplasm split: segmented region too small for a reliable 2-cluster fit"
        )
        return

    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV).astype(np.float32)
    pixels = hsv[mask_bool][:, 1:3]  # saturation, value -- nucleus tends darker (low V) + more saturated

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(pixels, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)

    # Nucleus cluster = lower mean "value" (darker) among the two clusters
    nucleus_cluster = int(np.argmin(centers[:, 1]))
    nucleus_pixel_count = int((labels.flatten() == nucleus_cluster).sum())
    cytoplasm_pixel_count = n_pixels - nucleus_pixel_count

    result.nucleus_area_px = float(nucleus_pixel_count)
    result.cytoplasm_area_px = float(cytoplasm_pixel_count)
    result.nucleus_to_cell_ratio = float(nucleus_pixel_count / n_pixels)
    result.unavailable_reasons.append(
        "nucleus/cytoplasm split is an UNVALIDATED classical-CV proxy (k-means on HSV), "
        "not checked against expert-labeled masks in this build -- treat as indicative, not precise"
    )
