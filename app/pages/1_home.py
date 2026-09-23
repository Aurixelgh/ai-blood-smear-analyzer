import streamlit as st

from app.styles.theme import (
    badge, disclaimer_line, hero_rbc_svg, html_block, inject_global_css,
    micro_particles_backdrop, section_label,
)
from app.utils.model_access import classifier_status, detector_status

inject_global_css()


def _render_page_link(page_path: str, label: str, icon: str) -> None:
    """st.page_link only resolves pages registered via st.Page/st.navigation
    (the app.py entrypoint). When this file is executed standalone -- e.g. by
    `streamlit run app/pages/1_home.py` or streamlit.testing.v1.AppTest --
    Streamlit raises StreamlitPageNotFoundError instead. Degrade to a plain
    caption there so the page still renders; navigation from the real app
    entrypoint keeps working links."""
    try:
        st.page_link(page_path, label=label, icon=icon)
    except st.errors.StreamlitPageNotFoundError:
        st.caption(f"{icon} {label} (available from the app navigation)")


# --- Hero ---------------------------------------------------------------
st.markdown(
    html_block(f"""
    <div style='position:relative; padding: 2.4rem 0 1.6rem 0; overflow:hidden;'>
      {micro_particles_backdrop(16)}
      <div style='position:relative; display:flex; align-items:center; gap:3rem; flex-wrap:wrap;'>
        <div style='flex:1; min-width:320px;'>
          <div class='sci-label' style='margin-bottom:0.9rem;'>Digital Microscopy &middot; Computer Vision &middot; AI Analysis</div>
          <h1 style='font-size:3.1rem; line-height:1.05; margin:0 0 1.1rem 0; font-weight:700;'>
            See the Blood Smear<br/>Differently.
          </h1>
          <p style='font-size:1.08rem; color:var(--text-muted); max-width:520px; line-height:1.55; margin-bottom:1.7rem;'>
            AI-assisted analysis of blood-cell morphology, classification, and
            image-derived cellular features &mdash; built for research and education.
          </p>
        </div>
        <div style='flex:0 0 auto; display:flex; justify-content:center; min-width:260px;'>
          <div style='filter: drop-shadow(0 18px 40px rgba(193,58,70,0.25));'>
            {hero_rbc_svg(260)}
          </div>
        </div>
      </div>
    </div>
    """),
    unsafe_allow_html=True,
)

col_cta1, col_cta2, col_cta3 = st.columns([1.1, 1.1, 2])
with col_cta1:
    if st.button("Analyze a Blood Smear", type="primary", use_container_width=True):
        st.switch_page("app/pages/3_analyze_smear.py")
with col_cta2:
    if st.button("Explore Cell Morphology", type="secondary", use_container_width=True):
        st.switch_page("app/pages/8_morphology.py")

disclaimer_line()

st.markdown("<hr class='sci-divider'/>", unsafe_allow_html=True)

# --- Pipeline overview ----------------------------------------------------
section_label("The Analysis Pipeline")
st.markdown(
    html_block("""
    <div style='display:flex; gap:0.4rem; flex-wrap:wrap; margin: 0.8rem 0 1.6rem 0;'>
    """) + "".join(
        f"<div class='sci-card' style='flex:1; min-width:118px; text-align:center; padding:0.9rem 0.5rem;'>"
        f"<div style='font-size:0.68rem; color:var(--text-faint); letter-spacing:0.08em; text-transform:uppercase; margin-bottom:0.3rem;'>{i+1:02d}</div>"
        f"<div style='font-size:0.8rem; font-weight:600;'>{step}</div></div>"
        for i, step in enumerate([
            "Quality Check", "Cell Detection", "Classification", "Confidence",
            "Morphology", "Explanation", "Counts + Map",
        ])
    ) + "</div>",
    unsafe_allow_html=True,
)

# --- Two primary entry points ---------------------------------------------
col1, col2 = st.columns(2)
with col1:
    st.markdown("<div class='sci-card'>", unsafe_allow_html=True)
    st.markdown("#### 🩸 Analyze Blood Smear")
    st.caption("Detect and classify every cell in a full microscope field.")
    ok, msg = detector_status()
    st.markdown(badge("Detector ready" if ok else "Pending training", "accent" if ok else "warn"), unsafe_allow_html=True)
    if not ok:
        st.caption(msg)
    st.write("")
    _render_page_link("app/pages/3_analyze_smear.py", "Open Analyze Blood Smear", "🩸")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='sci-card'>", unsafe_allow_html=True)
    st.markdown("#### 🔬 Analyze One Cell")
    st.caption("Classify a single blood cell with evidence and morphology.")
    ok, msg = classifier_status()
    st.markdown(badge("Classifier ready" if ok else "Pending training", "accent" if ok else "warn"), unsafe_allow_html=True)
    if not ok:
        st.caption(msg)
    st.write("")
    _render_page_link("app/pages/2_analyze_cell.py", "Open Analyze Cell", "🔬")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr class='sci-divider'/>", unsafe_allow_html=True)

# --- Supported classes -----------------------------------------------------
section_label("Supported Cell Classes")
class_cols = st.columns(7)
for col, name in zip(class_cols, ["RBC", "Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil", "Basophil", "Platelet"]):
    with col:
        st.markdown(
            f"<div style='text-align:center; padding:0.6rem 0.2rem; border:1px solid var(--border-subtle); "
            f"border-radius:8px; font-size:0.76rem; color:var(--text-muted);'>{name}</div>",
            unsafe_allow_html=True,
        )
st.caption(
    "A class is listed here only because a cited, expert-annotated dataset trained it and it "
    "was independently evaluated on held-out data -- see Research for real per-class metrics."
)
