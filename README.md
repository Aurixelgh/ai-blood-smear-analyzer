# AI Blood Smear Analyzer

A research-use and educational AI-assisted blood-smear image analysis platform.
**This is not a clinical diagnostic system.** See [About / Methodology](app/pages/7_about.py)
and the full reports in `reports/` for scientific caveats.

## What it does

Upload a microscope image and:
- **Analyze Cell** -- classify a single blood cell into one of 7 supported
  classes, with calibrated confidence, Grad-CAM model evidence, and measured
  morphology.
- **Analyze Blood Smear** -- detect and classify every cell in a whole-smear
  field, with per-class counts and an annotated image.
- **Cell Explorer** -- inspect any individual detection.
- **Compare Cells** -- side-by-side measured-morphology comparison of two cells.
- **Research / Metrics** -- the real, measured evaluation numbers for the
  trained models in this build (never fabricated -- if a number is missing,
  that model has not been trained yet).

## Supported cell classes

RBC, Neutrophil, Lymphocyte, Monocyte, Eosinophil, Basophil, Platelet.

A class is only "supported" because a cited, expert-annotated dataset trained
it and it was independently evaluated -- see `reports/dataset_feasibility_audit.md`
section 5 for exact per-class coverage, and the Research / Metrics page for
real per-class precision/recall/F1.

## Scientific motivation and datasets

Datasets, licenses, and full provenance are documented in
`reports/dataset_feasibility_audit.md` and `datasets/dataset_registry.{csv,yaml}`.
In brief, v1 uses:
- **PBC (Acevedo et al. 2020, Data in Brief, CC BY 4.0)** -- primary classification
  backbone for neutrophil/eosinophil/basophil/lymphocyte/monocyte/platelet.
- **Raabin-WBC (Kouzehkanan et al. 2022, Scientific Reports)** -- additional
  domain diversity.
- **TXL-PBC (Gan et al. 2025, Scientific Data, CC BY 4.0)** -- whole-smear
  detection training (WBC/RBC/platelet bounding boxes); also the source of
  RBC single-cell crops for the classifier, since no classification-oriented
  dataset in this audit contains mature single-cell RBC images.
- **BCCD (MIT license)** -- supplementary detection data.

## Architecture

See `reports/architecture_decision.md` for full rationale. Summary:
- **Detector:** YOLO-nano (Ultralytics), trained on TXL-PBC.
- **Classifier:** lightweight transfer-learned CNN (MobileNetV3-Small by
  default), trained on the pooled PBC + TXL-PBC-RBC dataset.
- **Segmentation:** classical computer vision (Otsu + watershed-style
  morphology), not a deep model -- only ~1,145 expert-labeled masks exist
  publicly (Raabin-WBC), too few to train a trustworthy segmentation network.
- **Explainability:** Grad-CAM.
- **Uncertainty:** temperature-scaled softmax + confidence-gated "LOW
  CONFIDENCE" output path.
- **Training compute:** this dev machine has no CUDA GPU (Intel Iris Xe
  integrated graphics only, confirmed by direct inspection) -- models were
  trained CPU-only locally; free-tier cloud GPU (Kaggle/Colab) is recommended
  for faster iteration, see the architecture doc.
- **Inference/deployment:** CPU-only throughout, targeting Streamlit
  Community Cloud.

## Project structure

```
app/            Streamlit UI (app.py entry point, app/pages/*, app/utils/*)
src/            Core logic: detection, classification via scripts/, morphology,
                explainability, uncertainty, inference orchestration
                (src/errors.py = typed error taxonomy; src/inference/contracts.py
                = result schemas + validators)
datasets/       raw/ (immutable downloads), processed/ (pipeline output),
                dataset_registry.csv/.yaml (provenance)
models/         Trained model artifacts (classifier_v1.pt, detector_v1.pt)
evaluation/     Real measured evaluation JSON reports (no fabricated metrics)
reports/        dataset_feasibility_audit.md, architecture_decision.md,
                methodology_report.md, parallel_engineering_handoff.md
scripts/        Data acquisition, pipeline, and training scripts
configs/        label_schema.yaml and other pipeline configs
tests/          Automated tests (pytest) -- 119 tests; model-dependent ones
                skip automatically until checkpoints exist
```

## Current model status

- **Classifier (7-class):** trained and evaluated on real held-out test data.
  Real measured numbers are in `evaluation/classifier_v1_evaluation.json`
  and on the app's Research / Metrics page -- never copied into docs, so
  they cannot go stale here.
- **Detector (whole-smear):** training in progress. Whole-smear pages show an
  explicit "detector weights are not available yet" pending state until
  `models/detector_v1.pt` exists; no placeholder detections are ever shown.

## Reproducing the pipeline

```bash
# 0. Environment: Python 3.14 (see runtime.txt), CPU-only is sufficient.
pip install -r requirements.txt

# 1. Acquire raw data (network access required; sources documented in the
#    dataset registry). See datasets/raw/*/ for what this build already has.

# 2. Build the pooled, leakage-audited classification dataset
python scripts/extract_rbc_crops_from_txlpbc.py
python scripts/build_classification_dataset.py

# 3. Train (fixed seed 42 for the split; see scripts for exact configs)
python scripts/train_classifier.py --epochs 8 --batch-size 64 --img-size 128
python scripts/train_detector.py --epochs 20 --model yolov8n.pt

# 4. Run tests (model-dependent tests skip automatically if checkpoints
#    are missing -- a skip is reported, never faked)
python -m pytest tests/ -v

# 5. Run the app
streamlit run app.py
```

## Honesty and limitations

This project follows a strict no-fabrication rule: every prediction,
confidence, count, and metric shown anywhere in the app comes from an actual
model inference or an actual measurement on the specific uploaded image --
never a placeholder or invented value. Where a feature cannot be reliably
computed (e.g., RBC morphology subtypes, for which no adequate public dataset
was found), the system says so explicitly rather than guessing. See
`reports/dataset_feasibility_audit.md` section 12 and the About page for the
full, current list of known limitations.
