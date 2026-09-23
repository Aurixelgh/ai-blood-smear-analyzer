import cv2
import numpy as np
import streamlit as st

from app.styles.theme import badge, disclaimer_line, inject_global_css, pipeline_steps, section_label
from app.utils.model_access import classifier_status, detector_status, get_whole_smear_analyzer, hash_image_bytes
from src.inference.whole_smear import draw_annotations

inject_global_css()

st.markdown("<div class='sci-label'>Whole-Field Analysis</div>", unsafe_allow_html=True)
st.title("🩸 Analyze Blood Smear")
st.caption("Upload a full microscope blood-smear field to detect, classify, and count every cell.")

classifier_ok, classifier_msg = classifier_status()
detector_ok, detector_msg = detector_status()
if not (classifier_ok and detector_ok):
    st.warning("Whole-smear analysis needs both a trained detector and classifier.")
    if not detector_ok:
        st.caption(f"- {detector_msg}")
    if not classifier_ok:
        st.caption(f"- {classifier_msg}")
    st.stop()

uploaded_file = st.file_uploader("Insert a slide — upload a whole-smear microscope image", type=["jpg", "jpeg", "png", "bmp"])
if uploaded_file is None:
    st.markdown(
        "<div class='sci-card' style='text-align:center; padding:2.4rem 1rem; color:var(--text-faint);'>"
        "<div style='font-size:0.78rem; letter-spacing:0.1em; text-transform:uppercase;'>Drop a whole-smear field</div>"
        "<div style='font-size:0.76rem; margin-top:0.4rem;'>JPG · JPEG · PNG · BMP</div></div>",
        unsafe_allow_html=True,
    )
    st.stop()

image_bytes = uploaded_file.getvalue()
image_key = hash_image_bytes(image_bytes)

if st.session_state.get("smear_last_key") != image_key:
    file_array = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr_image = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
    if bgr_image is None:
        st.error("Could not decode this file as an image.")
        st.stop()

    steps_ph = st.empty()
    with steps_ph.container():
        pipeline_steps(["Image Quality", "Cell Detection", "Classification", "Counts", "Annotation"], active_index=1)
    analyzer = get_whole_smear_analyzer()
    with st.spinner("Detecting and classifying cells..."):
        try:
            result = analyzer.analyze(bgr_image)
        except Exception as exc:  # typed AnalysisError or unexpected -- both user-facing
            steps_ph.empty()
            st.error(
                "Whole-smear analysis could not be completed for this image. "
                f"({type(exc).__name__}: {exc})"
            )
            st.stop()
    steps_ph.empty()

    st.session_state["smear_last_key"] = image_key
    st.session_state["smear_last_result"] = result
    st.session_state["smear_last_image"] = bgr_image

result = st.session_state["smear_last_result"]
bgr_image = st.session_state["smear_last_image"]

if result.status in ("IMAGE_QUALITY_REJECTED", "NO_CELLS_DETECTED"):
    st.markdown(badge("No usable result", "warn"), unsafe_allow_html=True)
    st.error(result.message)
    st.stop()

pipeline_steps(["Image Quality", "Cell Detection", "Classification", "Counts", "Annotation"], done_index=4)

section_label("Detection Summary")
st.markdown(
    f"<div class='sci-card' style='display:flex;align-items:baseline;gap:0.6rem;margin-bottom:0.8rem;'>"
    f"<span style='font-family:Space Grotesk,sans-serif;font-size:2.4rem;font-weight:700;color:var(--muted-pink);'>{len(result.detections)}</span>"
    f"<span style='font-size:0.8rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.08em;'>cells detected</span>"
    f"</div>",
    unsafe_allow_html=True,
)

count_cols = st.columns(len(result.counts) or 1)
for col, (label, count) in zip(count_cols, sorted(result.counts.items(), key=lambda kv: -kv[1])):
    col.metric(label.upper(), count)

st.write("")
section_label("Annotated Field")
annotated = draw_annotations(bgr_image, result)
st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detections overlaid on the uploaded field", width="stretch")

n_low_conf = sum(1 for d in result.detections if d.is_low_confidence)
if n_low_conf:
    st.markdown(badge(f"{n_low_conf} low-confidence detection(s)", "warn"), unsafe_allow_html=True)
    st.caption("Shown with '?' and gray boxes above — see the Cells page to inspect them individually.")

st.caption("Open the Cells page to inspect each detected cell individually "
           "(crop, prediction, confidence, morphology).")
disclaimer_line()
