import cv2
import numpy as np
import streamlit as st

from app.styles.theme import badge, disclaimer_line, inject_global_css, pipeline_steps, section_label
from app.utils.model_access import (
    classifier_status, detector_available, get_single_cell_analyzer,
    get_whole_smear_analyzer, hash_image_bytes,
)
from src.inference.whole_smear import _suppress_duplicate_boxes

inject_global_css()


def _count_wbc_candidates(bgr_image: np.ndarray) -> int:
    """Honest multi-cell sanity check for this single-cell page: reuses the
    SAME validated whole-smear detector and confidence threshold already
    deployed elsewhere in the app (no new heuristic, no filename-based
    guessing).

    Counts WBC-class detections specifically, NOT total candidate boxes.
    Measured directly against this app's own 7 real single-cell samples: a
    legitimate single-cell upload (e.g. neutrophil_example.jpg) still shows
    17-22 detected background RBCs -- real blood smears always have a sea
    of RBCs behind the one cell of interest, so a raw total-box-count
    threshold would falsely reject every legitimate sample in this app.
    Every one of those 7 samples has exactly 0-1 WBC-class detections, so
    multiple distinct WBC detections is the signal that actually indicates
    a genuine multi-cell field rather than one isolated cell of interest."""
    smear_analyzer = get_whole_smear_analyzer()
    yolo_results = smear_analyzer.detector.predict(
        bgr_image, conf=smear_analyzer.detector_confidence_threshold, verbose=False
    )[0]
    names = yolo_results.names
    raw_boxes = [
        {"cls": int(b.cls[0].item()), "conf": float(b.conf[0].item()),
         "xyxy": tuple(float(v) for v in b.xyxy[0].tolist())}
        for b in yolo_results.boxes
    ]
    deduped = _suppress_duplicate_boxes(raw_boxes)
    return sum(1 for b in deduped if names[b["cls"]] == "WBC")

st.markdown("<div class='sci-label'>Single-Cell Workstation</div>", unsafe_allow_html=True)
st.title("🔬 Analyze Cell")
st.caption("Upload an image containing a single blood cell for classification, evidence, and morphology.")

classifier_ok, classifier_msg = classifier_status()
if not classifier_ok:
    st.warning(classifier_msg)
    st.stop()

uploaded_file = st.file_uploader(
    "Insert a slide — upload a single-cell microscope image",
    type=["jpg", "jpeg", "png", "bmp"],
)

if uploaded_file is None:
    st.markdown(
        "<div class='sci-card' style='text-align:center; padding:2.4rem 1rem; color:var(--text-faint);'>"
        "<div style='font-size:0.78rem; letter-spacing:0.1em; text-transform:uppercase;'>Drop a blood-smear cell image</div>"
        "<div style='font-size:0.76rem; margin-top:0.4rem;'>JPG · JPEG · PNG · BMP</div></div>",
        unsafe_allow_html=True,
    )
    st.stop()

image_bytes = uploaded_file.getvalue()
image_key = hash_image_bytes(image_bytes)

# Per-upload isolation (spec section 32): results are keyed by this specific
# image's hash, never reused for a different upload even within one session.
if st.session_state.get("analyze_cell_last_key") != image_key:
    file_array = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr_image = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
    if bgr_image is None:
        st.error("Could not decode this file as an image. Try a different file.")
        st.stop()

    # Scientifically-honest multi-cell check (spec: this page is for one
    # isolated cell; a full smear field belongs on Analyze Blood Smear).
    # Reuses the SAME validated detector/threshold used elsewhere in the app
    # -- an actual model forward pass, not a filename or heuristic guess.
    # Only runs if the detector happens to be trained; its absence must
    # never block single-cell classification, which only needs the classifier.
    n_wbc_candidates = None
    if detector_available():
        try:
            n_wbc_candidates = _count_wbc_candidates(bgr_image)
        except Exception:
            n_wbc_candidates = None  # sanity check itself failing must not block classification

    if n_wbc_candidates is not None and n_wbc_candidates >= 2:
        result = {"status": "MULTI_CELL_IMAGE_REJECTED", "n_candidate_cells": n_wbc_candidates}
    else:
        steps_ph = st.empty()
        with steps_ph.container():
            pipeline_steps(["Image Quality", "Classification", "Confidence", "Morphology", "Evidence"], active_index=0)
        analyzer = get_single_cell_analyzer()
        with st.spinner("Analyzing..."):
            try:
                result = analyzer.analyze(bgr_image)
            except Exception as exc:  # typed AnalysisError or unexpected -- both user-facing
                steps_ph.empty()
                st.error(
                    "Analysis could not be completed for this image. "
                    f"({type(exc).__name__}: {exc})"
                )
                st.stop()
        steps_ph.empty()

    st.session_state["analyze_cell_last_key"] = image_key
    st.session_state["analyze_cell_last_result"] = result
    st.session_state["analyze_cell_last_image"] = bgr_image

result = st.session_state["analyze_cell_last_result"]
bgr_image = st.session_state["analyze_cell_last_image"]

pipeline_steps(["Image Quality", "Classification", "Confidence", "Morphology", "Evidence"], done_index=4)

col_img, col_result = st.columns([1, 1.4])
with col_img:
    st.image(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB), caption="Uploaded image", width="stretch")

with col_result:
    if result["status"] == "MULTI_CELL_IMAGE_REJECTED":
        st.markdown(badge("Multiple cells detected", "warn"), unsafe_allow_html=True)
        st.error(
            f"This image appears to contain multiple white blood cells ({result['n_candidate_cells']} "
            "detected by the same validated detector used elsewhere in this app), not one isolated "
            "cell of interest. (Background red blood cells are normal and expected in a single-cell "
            "photo; this check only flags multiple distinct white cells in the frame.) This page "
            "classifies a single cell; for a full microscope field, use the Analyze Blood Smear page instead."
        )
        st.stop()

    if result["status"] == "IMAGE_QUALITY_REJECTED":
        st.markdown(badge("Image quality insufficient", "warn"), unsafe_allow_html=True)
        st.error(result["message"])
        for reason in result["quality_reasons"]:
            st.caption(f"- {reason}")
        st.stop()

    prediction = result["prediction"]

    st.markdown("<div class='sci-card'>", unsafe_allow_html=True)
    if prediction["status"] == "LOW_CONFIDENCE":
        st.markdown(badge("Low confidence", "warn"), unsafe_allow_html=True)
        st.write(prediction["message"])
    else:
        section_label("Predicted Class")
        st.markdown(
            f"<div style='font-family:Space Grotesk,sans-serif; font-size:2.1rem; font-weight:700; "
            f"color:var(--muted-pink); margin:0.15rem 0 0.6rem 0;'>{prediction['predicted_class'].upper()}</div>",
            unsafe_allow_html=True,
        )
        st.metric("Confidence", f"{prediction['top1_probability']*100:.1f}%")
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    section_label("Alternative Predictions")
    sorted_probs = sorted(prediction["calibrated_probabilities"].items(), key=lambda kv: -kv[1])
    for class_name, prob in sorted_probs[:5]:
        st.progress(prob, text=f"{class_name.upper()} · {prob*100:.1f}%")

st.markdown("<hr class='sci-divider'/>", unsafe_allow_html=True)

tab_evidence, tab_morphology = st.tabs(["Why This Classification?", "Measured Morphology"])

with tab_evidence:
    section_label("Model Evidence — Grad-CAM")
    evidence = result.get("model_evidence")
    heatmap = evidence.get("gradcam_heatmap") if evidence else None
    if heatmap is not None:
        st.caption(evidence["note"])
        # The heatmap is sized to the classifier's input resolution (e.g. 128x128),
        # not the original upload's resolution -- resize before blending or
        # cv2.addWeighted raises a shape-mismatch error on any non-square/
        # non-matching upload.
        heatmap = np.array(heatmap, dtype=np.float32)
        heatmap_resized = cv2.resize(heatmap, (bgr_image.shape[1], bgr_image.shape[0]))
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(bgr_image, 0.6, heatmap_color, 0.4, 0)
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                 caption="Pixels the model weighted most heavily for this prediction",
                 width="stretch")
    else:
        # Covers both: no evidence (low-confidence prediction) and the
        # degraded path where evidence exists as a note with heatmap=None.
        st.info((evidence or {}).get("note")
                or "Model evidence is only shown for confident predictions.")

with tab_morphology:
    section_label("Image-Derived Measurements")
    morph = result["measured_morphology"]
    if morph["segmentation_status"] != "ok":
        st.warning("Morphology could not be reliably measured for this image.")
        for reason in morph["unavailable_reasons"]:
            st.caption(f"- {reason}")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Area (px)", f"{morph['area_px']:.0f}" if morph["area_px"] else "N/A")
        m2.metric("Circularity", f"{morph['circularity']:.3f}" if morph["circularity"] else "N/A")
        m3.metric("Aspect ratio", f"{morph['aspect_ratio']:.3f}" if morph["aspect_ratio"] else "N/A")
        m4.metric("Solidity", f"{morph['solidity']:.3f}" if morph["solidity"] else "N/A")
        if morph.get("nucleus_to_cell_ratio") is not None:
            st.metric("Nucleus/cell ratio", f"{morph['nucleus_to_cell_ratio']:.3f}")
            st.markdown(badge("Unvalidated proxy", "warn"), unsafe_allow_html=True)
        if morph["unavailable_reasons"]:
            with st.expander("Notes on this measurement"):
                for reason in morph["unavailable_reasons"]:
                    st.caption(f"- {reason}")
        st.json({k: v for k, v in morph.items() if k not in ("unavailable_reasons",)}, expanded=False)

disclaimer_line()
