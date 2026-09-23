"""
AI Blood Smear Analyzer -- Streamlit entry point.

Research/education positioning (spec section 28): this application is a
research-use and educational AI-assisted image analysis tool. It is not a
clinical diagnostic system and must never present a disease diagnosis.
"""
import streamlit as st

st.set_page_config(
    page_title="AI Blood Smear Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Navigation ORDER here drives the top nav bar's left-to-right order -- it is
# independent of the on-disk filenames/numbers, so pages can be reprioritized
# (Analyze Blood Smear first, per product priority) without renaming files
# that tests and internal page_link()s already reference.
pages = [
    st.Page("app/pages/1_home.py", title="Home", icon="🏠", default=True),
    st.Page("app/pages/3_analyze_smear.py", title="Analyze", icon="🩸"),
    st.Page("app/pages/2_analyze_cell.py", title="Analyze Cell", icon="🔬"),
    st.Page("app/pages/4_cell_explorer.py", title="Cells", icon="🔎"),
    st.Page("app/pages/8_morphology.py", title="Morphology", icon="🧬"),
    st.Page("app/pages/5_compare_cells.py", title="Compare", icon="⚖️"),
    st.Page("app/pages/9_rbc_biology.py", title="RBC Biology", icon="💠"),
    st.Page("app/pages/6_research_metrics.py", title="Research", icon="📊"),
    st.Page("app/pages/7_about.py", title="About", icon="📄"),
]

nav = st.navigation(pages, position="top")
nav.run()
