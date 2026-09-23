import json
from pathlib import Path

import streamlit as st

from app.styles.theme import badge, disclaimer_line, inject_global_css, section_label

inject_global_css()

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "evaluation"

st.markdown("<div class='sci-label'>Research</div>", unsafe_allow_html=True)
st.title("📊 Research / Metrics")
st.caption("Real, measured evaluation results from this build. Nothing on this page is fabricated -- "
           "if a metric is missing, it is because that model has not been trained/evaluated yet.")


def _load_eval_report(path: Path):
    """Loads an evaluation report, degrading to None with a message if the
    file is unreadable/malformed -- a broken report must never crash the
    page or tempt anyone to fill in placeholder numbers."""
    if not path.exists():
        return None, None
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (json.JSONDecodeError, OSError) as exc:
        return None, f"Report exists but could not be read/parsed: {exc}"


classifier_eval_path = EVAL_DIR / "classifier_v1_evaluation.json"
detector_eval_path = EVAL_DIR / "detector_v1_evaluation.json"
detector_detailed_path = EVAL_DIR / "detector_v1_detailed_evaluation.json"

section_label("Classifier — 7-Class Single-Cell")
report, load_error = _load_eval_report(classifier_eval_path)
if report is not None:
    tm = report.get("test_metrics")
    if not isinstance(tm, dict):
        st.warning("Classifier evaluation report exists but has no readable test metrics. "
                   "Re-run `scripts/train_classifier.py`.")
    else:
        st.markdown(badge("Evaluated on held-out test split", "accent"), unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{tm['accuracy']*100:.1f}%")
        c2.metric("Balanced accuracy", f"{tm['balanced_accuracy']*100:.1f}%")
        c3.metric("Macro F1", f"{tm['macro_f1']:.3f}")
        c4.metric("MCC", f"{tm['mcc']:.3f}")

        st.markdown("##### Per-class metrics (test set)")
        rows = [
            {"class": k, "precision": f"{v['precision']:.3f}", "recall": f"{v['recall']:.3f}",
             "f1": f"{v['f1']:.3f}", "support": v["support"]}
            for k, v in tm["per_class"].items()
        ]
        st.dataframe(rows, width="stretch", hide_index=True)

        with st.expander("Full classification report (text)"):
            st.text(tm["classification_report_text"])
        with st.expander("Training configuration and history"):
            st.json({k: v for k, v in report.items() if k != "test_metrics"})
        if "hardware" in report:
            st.caption(f"Hardware: {report['hardware']}")
elif load_error:
    st.error(load_error)
else:
    st.markdown(badge("Evaluation pending", "warn"), unsafe_allow_html=True)
    st.caption("Run `scripts/train_classifier.py`.")

st.markdown("<hr class='sci-divider'/>", unsafe_allow_html=True)
section_label("Detector — Whole-Smear Cell Detection")
report, load_error = _load_eval_report(detector_eval_path)
if report is not None:
    tm = report.get("test_set_metrics")
    if not isinstance(tm, dict):
        st.warning("Detector evaluation report exists but has no readable test metrics. "
                   "Re-run `scripts/train_detector.py`.")
    else:
        st.markdown(badge("Evaluated on held-out test split", "accent"), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("mAP50", f"{tm['map50']*100:.1f}%")
        c2.metric("mAP50-95", f"{tm['map50_95']*100:.1f}%")
        with st.expander("Full aggregate report (Ultralytics)"):
            st.json(report)
elif load_error:
    st.error(load_error)
else:
    st.markdown(badge("Evaluation pending", "warn"), unsafe_allow_html=True)
    st.caption("Run `scripts/train_detector.py`.")

detailed, detailed_error = _load_eval_report(detector_detailed_path)
if detailed is not None:
    st.markdown("##### Per-class detection performance (real IoU-matched, not aggregate mAP alone)")
    pc = detailed.get("per_class_metrics", {})
    rows = [
        {"class": k, "precision": f"{v['precision']:.3f}" if v["precision"] is not None else "N/A",
         "recall": f"{v['recall']:.3f}" if v["recall"] is not None else "N/A",
         "f1": f"{v['f1']:.3f}" if v["f1"] is not None else "N/A",
         "tp": v["tp"], "fp": v["fp"], "fn": v["fn"]}
        for k, v in pc.items()
    ]
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)
    st.caption(
        f"Confidence threshold {detailed.get('confidence_threshold')} (matches the deployed app), "
        f"IoU match threshold {detailed.get('iou_match_threshold')}. "
        "Recall stratified by object size / boundary proximity / touching-another-cell / image "
        "density is in the expander below -- reported per this project's rule against hiding weak "
        "performance behind an aggregate number."
    )
    with st.expander("Stratified recall (size, boundary, touching cells, density) and confusion patterns"):
        st.json(detailed)
elif detailed_error:
    st.error(detailed_error)

st.markdown("<hr class='sci-divider'/>", unsafe_allow_html=True)
section_label("Dataset Provenance & Feasibility Audit")
st.caption("See the full reports for real citations, licenses, and known limitations.")
audit_path = REPO_ROOT / "reports" / "dataset_feasibility_audit.md"
arch_path = REPO_ROOT / "reports" / "architecture_decision.md"
methodology_path = REPO_ROOT / "reports" / "methodology_report.md"
for label, path in [
    ("Dataset Feasibility Audit", audit_path),
    ("Architecture Decision Record", arch_path),
    ("Methodology Report", methodology_path),
]:
    if path.exists():
        with st.expander(f"{label} (full report)"):
            st.markdown(path.read_text(encoding="utf-8"))

disclaimer_line()
