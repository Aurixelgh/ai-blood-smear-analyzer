"""
RBC biology storytelling + microcirculation visualization.

Explicitly EDUCATIONAL/illustrative content, not model output (spec section
15/16): no simulated value here is presented as a real measurement. Every
number shown (e.g. "illustrative deformability index") is a hand-authored
demo value for the animation, clearly labeled as such and structurally
separated from src/morphology's real, measured features (which live on the
Morphology and Analyze pages).
"""
import textwrap

import streamlit as st

from app.styles.theme import badge, disclaimer_line, html_block, inject_global_css, section_label

inject_global_css()

st.markdown("<div class='sci-label'>RBC Biology</div>", unsafe_allow_html=True)
st.title("💠 Red Blood Cell Biology")
st.caption("Why shape and deformability matter for a red blood cell's function.")
disclaimer_line()
st.markdown(
    badge("Illustrative visualization — not measured by this model", "warn"),
    unsafe_allow_html=True,
)

st.markdown("<hr class='sci-divider'/>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1.2])
with col1:
    section_label("Structure")
    st.markdown(
        textwrap.dedent("""
        A healthy red blood cell (discocyte) is a biconcave disc: thinner in the
        center than at the rim. This shape maximizes surface area relative to
        volume, which supports efficient gas exchange, and gives the membrane
        the slack it needs to bend without stretching.

        The membrane's flexibility comes from a spectrin protein lattice just
        beneath the lipid bilayer. This scaffold is what lets the cell fold and
        recover its shape after passing through vessels narrower than its own
        resting diameter.
        """)
    )
with col2:
    section_label("Why It Matters")
    st.markdown(
        textwrap.dedent("""
        Capillaries and splenic slits can be narrower than a resting RBC.
        A healthy cell deforms to pass through, then relaxes back to its
        biconcave shape afterward.

        Conditions that stiffen the membrane or fix an abnormal shape
        (some hereditary and acquired disorders) can impair this passage,
        which is part of why RBC morphology is studied in blood smears in
        the first place -- shape is a visible proxy for underlying membrane
        and metabolic state, not a diagnosis on its own.
        """)
    )

st.markdown("<hr class='sci-divider'/>", unsafe_allow_html=True)

section_label("Microcirculation Passage — Illustrative Animation")
st.caption(
    "A hand-authored, illustrative animation of a red blood cell deforming to pass through a "
    "narrow channel. This is a teaching aid, not a simulation calibrated to real biomechanical data, "
    "and not an output of the trained model."
)

st.markdown(
    html_block("""
    <style>
    @keyframes rbc-travel {
        0%   { left: -8%;  transform: translateY(-50%) scaleX(1)    scaleY(1); }
        42%  { left: 40%;  transform: translateY(-50%) scaleX(1)    scaleY(1); }
        50%  { left: 48%;  transform: translateY(-50%) scaleX(1.55) scaleY(0.55); }
        58%  { left: 56%;  transform: translateY(-50%) scaleX(1)    scaleY(1); }
        100% { left: 104%; transform: translateY(-50%) scaleX(1)    scaleY(1); }
    }
    @media (prefers-reduced-motion: reduce) {
        .rbc-travel-cell { animation: none !important; left: 48% !important; }
    }
    </style>
    <div style='position:relative; height:150px; background:var(--bg-secondary); border:1px solid var(--border-subtle);
                border-radius:12px; overflow:hidden; margin-bottom:0.6rem;'>
      <div style='position:absolute; top:0; bottom:0; left:0; width:44%; height:100%;
                   background:linear-gradient(90deg, rgba(139,135,129,0.10), rgba(139,135,129,0.02));'></div>
      <div style='position:absolute; top:0; bottom:0; right:0; width:44%; height:100%;
                   background:linear-gradient(270deg, rgba(139,135,129,0.10), rgba(139,135,129,0.02));'></div>
      <div style='position:absolute; top:calc(50% - 11px); left:44%; width:12%; height:22px;
                   background:var(--bg-primary); border-top:2px solid var(--border-strong);
                   border-bottom:2px solid var(--border-strong);'></div>
      <div class='rbc-travel-cell' style='position:absolute; top:50%; width:64px; height:44px; border-radius:50%;
                   background:radial-gradient(circle at 40% 35%, #D6505C, #7C2A32 75%);
                   animation: rbc-travel 5.5s ease-in-out infinite;'></div>
    </div>
    """),
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
c1.metric("Illustrative deformability index", "0.87", help="A demo value for this animation only -- not a model output or a clinical measurement.")
c2.metric("Illustrative channel width", "0.6× resting diameter", help="Chosen for visual clarity, not derived from patient data.")
c3.metric("Model-measured value", "Not available", help="This system does not currently measure deformability from images.")
st.caption(
    "The three values above are deliberately shown side-by-side: the first two are illustrative "
    "constants for this animation, the third states plainly that no such measurement exists in this "
    "build -- see reports/dataset_feasibility_audit.md section 11 for why deformability/membrane "
    "integrity are not implemented as real measurements here."
)
