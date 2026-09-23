"""
Automated Streamlit UI tests using streamlit.testing.v1.AppTest -- runs the
real page scripts in a simulated session, not a mock. Spec section 31
requires testing "UI-critical flows" and requires that "frontend changes
must not silently break inference."

Two kinds of coverage here:
  1. Model-absent error states (always testable, regardless of training
     status): every analysis page must show a clear, specific error and
     must NOT raise an unhandled exception when the required model file(s)
     do not exist yet -- this is exactly spec section 33's "do not expose
     raw Python tracebacks" requirement, and it is real coverage right now
     because no model has been trained yet in this environment.
  2. Real-model flows: once a checkpoint exists, some tests drive the ACTUAL
     Streamlit widget tree with a real held-out sample image via
     FileUploader.upload() -- not just calling the underlying analyzer
     function directly (that direct-call coverage lives in
     test_end_to_end.py). This is the strongest UI-critical-flow check
     available without a live browser.
"""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[1]
CLASSIFIER_PATH = REPO_ROOT / "models" / "classifier_v1.pt"
DETECTOR_PATH = REPO_ROOT / "models" / "detector_v1.pt"


def _run_page(relative_path: str, timeout: int = 30):
    at = AppTest.from_file(str(REPO_ROOT / relative_path), default_timeout=timeout)
    at.run()
    return at


@pytest.mark.skipif(CLASSIFIER_PATH.exists(), reason="classifier now trained -- this tests the absent-model path specifically")
def test_analyze_cell_page_shows_clear_error_when_classifier_missing():
    at = _run_page("app/pages/2_analyze_cell.py")
    assert not at.exception, f"Page raised an unhandled exception: {at.exception}"
    assert any("classifier_v1.pt" in w.value for w in at.warning)


@pytest.mark.skipif(not CLASSIFIER_PATH.exists(), reason="requires a trained classifier_v1.pt")
def test_analyze_cell_page_full_upload_flow_with_real_image():
    """Drives the ACTUAL Streamlit widget tree (not just the underlying
    Python function) with a real held-out sample image via
    FileUploader.upload() -- the strongest form of UI-critical-flow coverage
    available without a live browser: real file bytes -> real file_uploader
    widget -> real page script -> real model -> real rendered output."""
    samples_dir = REPO_ROOT / "samples" / "single_cell"
    image_path = samples_dir / "neutrophil_example.jpg"
    assert image_path.exists()

    at = AppTest.from_file(str(REPO_ROOT / "app/pages/2_analyze_cell.py"), default_timeout=60)
    at.run()
    at.file_uploader[0].upload("neutrophil_example.jpg", image_path.read_bytes(), "image/jpeg").run()

    assert not at.exception, f"Page raised an unhandled exception on a real upload: {at.exception}"
    assert any("NEUTROPHIL" in m.value for m in at.markdown), \
        "expected the predicted class heading to render for a real, confidently-classified image"
    confidence_metrics = [m for m in at.metric if m.label == "Confidence"]
    assert confidence_metrics, "expected a rendered Confidence metric"
    assert confidence_metrics[0].value.endswith("%")
    # Morphology tab must render real measured numbers, not placeholders.
    area_metrics = [m for m in at.metric if m.label == "Area (px)"]
    assert area_metrics and area_metrics[0].value != "N/A"


@pytest.mark.skipif(CLASSIFIER_PATH.exists() and DETECTOR_PATH.exists(),
                     reason="both models now trained -- this tests the absent-model path specifically")
def test_analyze_smear_page_shows_clear_error_when_models_missing():
    at = _run_page("app/pages/3_analyze_smear.py")
    assert not at.exception, f"Page raised an unhandled exception: {at.exception}"
    assert any("detector_v1.pt" in c.value or "classifier_v1.pt" in c.value for c in at.caption)


@pytest.mark.skipif(CLASSIFIER_PATH.exists(), reason="classifier now trained -- this tests the absent-model path specifically")
def test_compare_cells_page_shows_clear_error_when_classifier_missing():
    at = _run_page("app/pages/5_compare_cells.py")
    assert not at.exception, f"Page raised an unhandled exception: {at.exception}"
    assert any("classifier" in w.value.lower() for w in at.warning)


def test_cell_explorer_page_shows_info_with_no_prior_smear_analysis():
    """No st.session_state is seeded -- this is the real first-visit state."""
    at = _run_page("app/pages/4_cell_explorer.py")
    assert not at.exception, f"Page raised an unhandled exception: {at.exception}"
    assert len(at.info) > 0


def test_home_page_renders_without_error():
    # Run via the real entrypoint (app.py), not the page file in isolation:
    # 1_home.py calls st.page_link() to other pages, which Streamlit only
    # resolves against pages actually registered by st.navigation in
    # app.py -- exercising app.py here matches real `streamlit run app.py`
    # behavior instead of a standalone-page fiction that would pass even if
    # app.py's navigation wiring were broken.
    at = _run_page("app.py")
    assert not at.exception, f"App raised an unhandled exception: {at.exception}"
    # The home page's hero uses a custom-styled HTML heading (app/styles/theme.py)
    # rather than st.title(), for layout control over the hero -- so this checks
    # the actual rendered headline text instead of the st.title() element list.
    assert any("See the Blood Smear" in m.value for m in at.markdown), \
        "expected the hero headline to render on the home page"


def test_research_metrics_page_renders_without_error_before_training():
    """Should show honest 'not trained yet' warnings, never fabricated numbers,
    and never crash, regardless of whether models exist yet."""
    at = _run_page("app/pages/6_research_metrics.py")
    assert not at.exception, f"Page raised an unhandled exception: {at.exception}"


def test_about_page_states_non_diagnostic_positioning():
    at = _run_page("app/pages/7_about.py")
    assert not at.exception, f"Page raised an unhandled exception: {at.exception}"
    full_text = " ".join(w.value for w in at.warning) + " ".join(m.value for m in at.markdown)
    assert "not a clinical diagnostic system" in full_text.lower() or \
           "not a clinical diagnostic" in full_text.lower()


def test_morphology_page_renders_without_error():
    """The reference atlas (illustrative) must render without needing a
    trained model at all -- only the real-inspector section below it
    depends on an uploaded image, and that path is untouched until upload."""
    at = _run_page("app/pages/8_morphology.py")
    assert not at.exception, f"Page raised an unhandled exception: {at.exception}"
    full_text = " ".join(m.value for m in at.markdown)
    assert "Not detected by this model" in full_text, \
        "the illustrative atlas must be explicitly labeled as not model-detected"


def test_rbc_biology_page_renders_without_error():
    """Educational content only -- must render with no model dependency, and
    must keep illustrative demo values clearly separated from real model
    output (spec section 15/16's no-overclaiming rule)."""
    at = _run_page("app/pages/9_rbc_biology.py")
    assert not at.exception, f"Page raised an unhandled exception: {at.exception}"
    metric_labels = [m.label for m in at.metric]
    assert any("Illustrative" in lbl for lbl in metric_labels)
    assert any(lbl == "Model-measured value" for lbl in metric_labels)
    model_measured = next(m for m in at.metric if m.label == "Model-measured value")
    assert model_measured.value == "Not available", \
        "must not claim a real measurement the model does not produce"
