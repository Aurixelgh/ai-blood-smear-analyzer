import cv2
import numpy as np
import streamlit as st

from app.styles.theme import badge, inject_global_css, section_label
from src.morphology.features import compute_morphology

inject_global_css()

st.markdown("<div class='sci-label'>Microscopy Gallery</div>", unsafe_allow_html=True)
st.title("🔎 Cell Explorer")

result = st.session_state.get("smear_last_result")
bgr_image = st.session_state.get("smear_last_image")

if result is None or result.status != "OK" or not result.detections:
    st.info("Run an analysis on the Analyze Blood Smear page first -- detections "
            "from that page will appear here.")
    st.stop()

st.caption(f"{len(result.detections)} detected cells from the last whole-smear analysis.")

# --- Filters ---------------------------------------------------------------
with st.sidebar:
    section_label("Filters")
    all_labels = sorted({d.final_label for d in result.detections})
    type_filter = st.multiselect("Cell type", options=all_labels, default=all_labels)
    min_conf = st.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05)
    only_low_conf = st.checkbox("Low-confidence only", value=False)

filtered = [
    d for d in result.detections
    if d.final_label in type_filter
    and d.final_confidence >= min_conf
    and (d.is_low_confidence if only_low_conf else True)
]

if not filtered:
    st.warning("No detections match the current filters.")
    st.stop()

st.caption(f"Showing {len(filtered)} of {len(result.detections)} detections.")

col_list, col_detail = st.columns([1, 1.6])

with col_list:
    labels_for_display = [
        f"#{d.crop_index}  {d.final_label.upper()}"
        + (" ⚠" if d.is_low_confidence else "")
        for d in filtered
    ]
    selected_idx = st.radio("Detected Cells", options=list(range(len(filtered))),
                             format_func=lambda i: labels_for_display[i], label_visibility="collapsed")

detection = filtered[selected_idx]
x1, y1, x2, y2 = detection.box_xyxy
h, w = bgr_image.shape[:2]
crop = bgr_image[max(0, int(y1)):min(int(y2), h), max(0, int(x1)):min(int(x2), w)]

with col_detail:
    if crop.size == 0:
        st.error("This cell's crop could not be extracted from the stored image "
                 "(box outside image bounds).")
        st.stop()
    st.markdown(
        f"<div class='sci-card'><div style='display:flex;align-items:center;gap:0.6rem;'>"
        f"<span style='font-family:Space Grotesk,sans-serif;font-size:1.4rem;font-weight:700;color:var(--muted-pink);'>"
        f"#{detection.crop_index} {detection.final_label.upper()}</span>"
        f"{badge('Low confidence', 'warn') if detection.is_low_confidence else badge('Confident', 'accent')}"
        f"</div></div>",
        unsafe_allow_html=True,
    )
    st.write("")

    c1, c2 = st.columns(2)
    c1.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), caption="Crop", width="stretch")
    c2.metric("Detector confidence", f"{detection.detector_confidence*100:.1f}%")
    c2.metric("Final confidence", f"{detection.final_confidence*100:.1f}%")
    c2.caption(f"Detector's coarse class: {detection.detector_class}")

    if crop.size > 0:
        morph = compute_morphology(crop)
        section_label("Measured Morphology")
        if morph.segmentation_status == "ok":
            m1, m2, m3 = st.columns(3)
            m1.metric("Area (px)", f"{morph.area_px:.0f}" if morph.area_px else "N/A")
            m2.metric("Circularity", f"{morph.circularity:.3f}" if morph.circularity else "N/A")
            m3.metric("Aspect ratio", f"{morph.aspect_ratio:.3f}" if morph.aspect_ratio else "N/A")
        else:
            st.caption("Morphology unavailable for this crop: " + "; ".join(morph.unavailable_reasons))
