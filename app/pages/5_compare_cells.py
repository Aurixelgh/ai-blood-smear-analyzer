import cv2
import numpy as np
import streamlit as st

from app.styles.theme import badge, disclaimer_line, inject_global_css, section_label
from app.utils.model_access import classifier_status, get_single_cell_analyzer
from src.morphology.comparison import compare_measured, get_general_knowledge
from src.morphology.features import compute_morphology

inject_global_css()

st.markdown("<div class='sci-label'>Side-by-Side Analysis</div>", unsafe_allow_html=True)
st.title("⚖️ Compare Cells")
st.caption("Upload two single-cell images to compare their measured morphology and predictions.")

classifier_ok, classifier_msg = classifier_status()
if not classifier_ok:
    st.warning(classifier_msg)
    st.stop()

col_a, col_b = st.columns(2)
with col_a:
    file_a = st.file_uploader("Cell A", type=["jpg", "jpeg", "png", "bmp"], key="cmp_a")
with col_b:
    file_b = st.file_uploader("Cell B", type=["jpg", "jpeg", "png", "bmp"], key="cmp_b")

if not (file_a and file_b):
    st.info("Upload both images to compare. Try Monocyte vs Lymphocyte, Neutrophil vs Eosinophil, or RBC vs Platelet.")
    st.stop()


def decode(uploaded):
    arr = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


img_a, img_b = decode(file_a), decode(file_b)
if img_a is None or img_b is None:
    bad = "Cell A" if img_a is None else "Cell B"
    st.error(f"Could not decode {bad} as an image. Try a different file.")
    st.stop()

analyzer = get_single_cell_analyzer()
try:
    result_a, result_b = analyzer.analyze(img_a), analyzer.analyze(img_b)
except Exception as exc:  # typed AnalysisError or unexpected -- both user-facing
    st.error(f"Analysis could not be completed for these images. ({type(exc).__name__}: {exc})")
    st.stop()
morph_a, morph_b = compute_morphology(img_a), compute_morphology(img_b)

label_a = result_a.get("prediction", {}).get("predicted_class") or "unknown"
label_b = result_b.get("prediction", {}).get("predicted_class") or "unknown"

col_a, col_b = st.columns(2)
col_a.image(cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB), caption=f"Cell A: {label_a.upper()}", width="stretch")
col_b.image(cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB), caption=f"Cell B: {label_b.upper()}", width="stretch")

for col, res in ((col_a, result_a), (col_b, result_b)):
    with col:
        if res["status"] == "IMAGE_QUALITY_REJECTED":
            st.markdown(badge("Quality check failed", "warn"), unsafe_allow_html=True)
        elif res["prediction"]["status"] == "LOW_CONFIDENCE":
            st.markdown(badge("Low confidence", "warn"), unsafe_allow_html=True)
        else:
            st.metric("Confidence", f"{res['prediction']['top1_probability']*100:.1f}%")

st.markdown("<hr class='sci-divider'/>", unsafe_allow_html=True)
section_label("Measured Comparison — From These Two Specific Images")
comparison = compare_measured(morph_a, label_a, morph_b, label_b)
rows = []
for feature, vals in comparison["features"].items():
    rows.append({
        "feature": feature,
        "cell_a": f"{vals['value_a']:.3f}" if vals["available"] else "unavailable",
        "cell_b": f"{vals['value_b']:.3f}" if vals["available"] else "unavailable",
        "difference": f"{vals['difference']:.3f}" if vals["available"] else "-",
    })
st.dataframe(rows, width="stretch", hide_index=True)

st.markdown("<hr class='sci-divider'/>", unsafe_allow_html=True)
section_label("General Scientific Knowledge — Reference, Not a Measurement")
st.markdown(badge("Static reference text", "muted"), unsafe_allow_html=True)
st.caption("NOT a measurement of these two specific cells (see spec section 8).")
knowledge = get_general_knowledge(label_a, label_b)
if knowledge:
    for point in knowledge:
        st.markdown(f"- {point}")
else:
    st.caption("No curated reference text available for this specific class pair yet.")

disclaimer_line()
