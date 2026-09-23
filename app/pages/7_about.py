import streamlit as st

from app.styles.theme import inject_global_css

inject_global_css()

st.markdown("<div class='sci-label'>Methodology</div>", unsafe_allow_html=True)
st.title("📄 About / Methodology")

st.warning(
    "**This is a research-use and educational AI-assisted image analysis tool. "
    "It is NOT a clinical diagnostic system.** It does not diagnose disease, and "
    "no output on this site should be interpreted as a medical finding about any "
    "person. Predictions are outputs of a machine learning model trained on public "
    "research datasets, evaluated on held-out data from those same datasets -- not "
    "validated against independent clinical ground truth."
)

st.markdown(
    """
### What this system does

Given a microscope image, it:
1. checks basic image quality (resolution, blur, brightness),
2. detects individual cells in whole-smear images (YOLO-nano detector),
3. classifies each cell into one of 7 supported classes (RBC, Neutrophil,
   Lymphocyte, Monocyte, Eosinophil, Basophil, Platelet),
4. reports a calibrated confidence and flags low-confidence predictions
   rather than forcing an answer,
5. shows Grad-CAM model evidence for the prediction,
6. computes image-derived morphology features via classical computer vision,
7. lets you compare two cells' measured morphology side by side.

### Supported classes and known limitations

A class is only listed as "supported" because a real, cited, expert-annotated
dataset trained it and it was independently evaluated -- not merely because a
source dataset's folder was named after it. See the **Research / Metrics**
page for the actual measured per-class precision/recall/F1 from this build.

**Known limitations (stated plainly, not hidden):**
- RBC *morphology subtype* classification (discocyte/echinocyte/spherocyte/
  sickle/target-cell) is **not implemented** -- no sufficiently reliable public
  dataset was found for it. Only generic geometric descriptors (circularity,
  aspect ratio, etc.) are reported for RBCs, explicitly as unlabeled proxies.
- RBC training crops for the classifier come from a different acquisition
  pipeline (whole-smear detection crops) than the WBC/platelet training images
  (dedicated single-cell captures) -- a documented domain-shift risk.
- Basophils are the smallest class in every source dataset used; expect wider
  uncertainty on this class specifically.
- Nucleus/cytoplasm segmentation is an unvalidated classical computer-vision
  proxy in this build, not checked against expert-labeled ground truth masks.
- No deformability, membrane-integrity, or other stress-response feature is
  presented as a validated clinical measurement -- any such feature, if shown,
  is explicitly labeled an *image-derived morphology proxy*.

Full detail, citations, and the exact license/provenance of every dataset used
are in `reports/dataset_feasibility_audit.md`. Architecture rationale is in
`reports/architecture_decision.md`. Both are also viewable from the
Research / Metrics page.
"""
)
