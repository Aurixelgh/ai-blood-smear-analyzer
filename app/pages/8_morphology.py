"""
Morphology atlas + inspector.

Two clearly separated sections (spec: "clearly distinguish morphology shown
as biological reference/educational visualization vs. morphology actually
inferred by the current model"):
  1. An educational RBC shape atlas -- static reference illustrations with
     textbook descriptions, explicitly labeled as reference material the
     model does NOT classify into (see dataset_feasibility_audit.md section
     10/12: no adequately verified public dataset exists for RBC
     morphology-subtype classification, so this project does not claim to
     detect these shapes).
  2. A real morphology inspector -- upload an actual cell image and see the
     REAL, measured geometric features from src/morphology/features.py,
     exactly as computed for any other page, with the same honest
     "unavailable" reporting when segmentation fails.
"""
import cv2
import numpy as np
import streamlit as st

from app.styles.theme import badge, disclaimer_line, inject_global_css, section_label
from src.morphology.features import compute_morphology

inject_global_css()

st.markdown("<div class='sci-label'>Morphology</div>", unsafe_allow_html=True)
st.title("🧬 Cell Morphology")
st.caption("An educational shape atlas, and a real measurement inspector for your own images.")
disclaimer_line()

st.markdown("<hr class='sci-divider'/>", unsafe_allow_html=True)

section_label("Reference Atlas — Educational Illustration Only")
st.markdown(
    badge("Not detected by this model", "warn") +
    "&nbsp;&nbsp;<span style='font-size:0.85rem;color:var(--text-muted);'>"
    "These shapes are standard hematology reference categories, shown for education. "
    "This project's dataset audit found no adequately verified public dataset for RBC "
    "morphology-subtype classification, so the model does not classify cells into these "
    "categories -- see the Research page and dataset_feasibility_audit.md.</span>",
    unsafe_allow_html=True,
)

ATLAS = [
    ("Discocyte", "Normal biconcave disc shape.", "#C13A46", 1.0, 0.0),
    ("Echinocyte", "Multiple small, evenly spaced spiky projections.", "#B4483F", 0.82, 10),
    ("Spherocyte", "Sphere-like, lacking the central pallor of a biconcave disc.", "#A5303C", 0.72, 0.0),
    ("Stomatocyte", "A single slit-like (mouth-shaped) central depression.", "#8E2E38", 0.9, 0.0),
    ("Elliptocyte", "Elongated, elliptical/oval shape.", "#7C2A32", 1.0, 0.0),
    ("Target cell", "Central darker zone surrounded by a pale ring and dark rim.", "#6E1F2B", 1.0, 0.0),
    ("Schistocyte", "Irregular fragment, often with sharp angular edges.", "#5C1620", 0.6, 5),
]

cols = st.columns(4)
for i, (name, desc, color, squish, spikes) in enumerate(ATLAS):
    with cols[i % 4]:
        spike_paths = ""
        if spikes:
            import math
            pts = []
            for k in range(int(spikes)):
                angle = (2 * math.pi / spikes) * k
                x1, y1 = 60 + 34 * math.cos(angle), 50 + 34 * math.sin(angle)
                x2, y2 = 60 + 44 * math.cos(angle), 50 + 44 * math.sin(angle)
                pts.append(f"<line x1='{x1:.0f}' y1='{y1:.0f}' x2='{x2:.0f}' y2='{y2:.0f}' stroke='{color}' stroke-width='2'/>")
            spike_paths = "".join(pts)
        svg = (
            f"<svg width='100%' height='90' viewBox='0 0 120 100'>"
            f"<ellipse cx='60' cy='50' rx='{34*squish:.0f}' ry='34' fill='{color}' opacity='0.85'/>"
            f"{spike_paths}"
            f"</svg>"
        )
        st.markdown(
            f"<div class='sci-card' style='text-align:center; padding:0.8rem;'>{svg}"
            f"<div style='font-weight:600; margin-top:0.4rem;'>{name}</div>"
            f"<div style='font-size:0.76rem; color:var(--text-muted); margin-top:0.2rem;'>{desc}</div></div>",
            unsafe_allow_html=True,
        )

st.markdown("<hr class='sci-divider'/>", unsafe_allow_html=True)

section_label("Real Morphology Inspector")
st.caption("Upload any single-cell image to see its actual, measured geometric features.")

uploaded = st.file_uploader("Upload a cell image", type=["jpg", "jpeg", "png", "bmp"], key="morph_upload")
if uploaded is not None:
    arr = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        st.error("Could not decode this file as an image.")
    else:
        col_img, col_data = st.columns([1, 1.4])
        with col_img:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded image", width="stretch")
        with col_data:
            morph = compute_morphology(img)
            if morph.segmentation_status != "ok":
                st.warning("Morphology could not be reliably measured for this image.")
                for reason in morph.unavailable_reasons:
                    st.caption(f"- {reason}")
            else:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Area (px)", f"{morph.area_px:.0f}" if morph.area_px else "N/A")
                m2.metric("Perimeter (px)", f"{morph.perimeter_px:.0f}" if morph.perimeter_px else "N/A")
                m3.metric("Circularity", f"{morph.circularity:.3f}" if morph.circularity else "N/A")
                m4.metric("Solidity", f"{morph.solidity:.3f}" if morph.solidity else "N/A")
                m5, m6, m7 = st.columns(3)
                m5.metric("Aspect ratio", f"{morph.aspect_ratio:.3f}" if morph.aspect_ratio else "N/A")
                m6.metric("Eccentricity", f"{morph.eccentricity:.3f}" if morph.eccentricity is not None else "N/A")
                m7.metric("Extent", f"{morph.extent:.3f}" if morph.extent else "N/A")
                if morph.nucleus_to_cell_ratio is not None:
                    st.metric("Nucleus/cell ratio", f"{morph.nucleus_to_cell_ratio:.3f}")
                    st.caption(badge("Unvalidated proxy", "warn"), unsafe_allow_html=True)
                if morph.unavailable_reasons:
                    with st.expander("Notes on this measurement"):
                        for reason in morph.unavailable_reasons:
                            st.caption(f"- {reason}")
