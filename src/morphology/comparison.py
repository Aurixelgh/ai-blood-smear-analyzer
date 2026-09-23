"""
Cell comparison logic (spec section 9): Monocyte vs Lymphocyte, Neutrophil vs
Eosinophil, RBC vs Platelet, and any other pair among the 7 target classes.

Design constraint from spec section 9: "Where numerical measurements exist,
display actual measured values. Do not invent values." This module therefore
does two clearly separated things:
  1. `compare_measured(...)` -- diffs two REAL MorphologyResult objects
     (src/morphology/features.py) computed from two actual uploaded/selected
     cell images. Every number here came from a real segmentation of a real
     image.
  2. `GENERAL_KNOWLEDGE` -- static, textbook-level descriptive text about each
     class pair's typically-reported morphological differences. This is
     clearly labeled as general scientific knowledge, NOT a measurement of
     the two specific cells being compared (spec section 8's category C).
     The UI must render this in a visually distinct block from
     `compare_measured`'s output, and must never merge the two.
"""
from typing import Optional

from src.morphology.features import MorphologyResult

NUMERIC_FIELDS = [
    "area_px", "perimeter_px", "equivalent_diameter_px", "circularity",
    "aspect_ratio", "solidity", "extent", "eccentricity",
    "intensity_std", "edge_density", "nucleus_to_cell_ratio",
]


def compare_measured(result_a: MorphologyResult, label_a: str,
                      result_b: MorphologyResult, label_b: str) -> dict:
    """Returns per-feature {value_a, value_b, difference, available} -- never
    substitutes a guessed value when either side is None."""
    comparison = {}
    for field_name in NUMERIC_FIELDS:
        val_a = getattr(result_a, field_name)
        val_b = getattr(result_b, field_name)
        if val_a is None or val_b is None:
            comparison[field_name] = {
                "value_a": val_a, "value_b": val_b, "difference": None, "available": False,
            }
        else:
            comparison[field_name] = {
                "value_a": val_a, "value_b": val_b, "difference": float(val_a - val_b), "available": True,
            }
    return {
        "label_a": label_a, "label_b": label_b,
        "features": comparison,
        "segmentation_status_a": result_a.segmentation_status,
        "segmentation_status_b": result_b.segmentation_status,
    }


# Static, textbook-level reference text -- general scientific knowledge, not a
# measurement of any specific pair of uploaded cells. Sourced from standard
# hematology morphology descriptions (consistent with the class definitions
# used by the PBC/Raabin-WBC dataset papers cited in dataset_feasibility_audit.md).
GENERAL_KNOWLEDGE = {
    ("monocyte", "lymphocyte"): [
        "Monocytes are typically the largest normal peripheral blood leukocyte; "
        "lymphocytes (especially small lymphocytes) are typically smaller.",
        "Monocyte nuclei are often folded, indented, or kidney/horseshoe-shaped; "
        "small lymphocyte nuclei are typically round and dense/clumped.",
        "Monocytes usually show more abundant, gray-blue, sometimes vacuolated "
        "cytoplasm; small lymphocytes typically show a thin rim of scant cytoplasm.",
    ],
    ("neutrophil", "eosinophil"): [
        "Neutrophils typically show a multi-lobed (2-5 lobe) segmented nucleus "
        "connected by thin chromatin strands; eosinophils are typically bilobed.",
        "Eosinophils are characterized by large, coarse, orange-red (eosinophilic) "
        "cytoplasmic granules; neutrophil granules are finer and more neutral-staining.",
    ],
    ("rbc", "platelet"): [
        "Mature red blood cells are typically much larger than platelets and, in "
        "normal physiology, lack a nucleus (anucleate biconcave discs).",
        "Platelets are small anucleate cell fragments, typically a fraction of "
        "the diameter of a red blood cell, often appearing as small purple-staining specks.",
    ],
}


def get_general_knowledge(label_a: str, label_b: str) -> Optional[list]:
    key = (label_a, label_b)
    if key in GENERAL_KNOWLEDGE:
        return GENERAL_KNOWLEDGE[key]
    reversed_key = (label_b, label_a)
    if reversed_key in GENERAL_KNOWLEDGE:
        return GENERAL_KNOWLEDGE[reversed_key]
    return None  # No canned reference text for this pair -- do not fabricate one
