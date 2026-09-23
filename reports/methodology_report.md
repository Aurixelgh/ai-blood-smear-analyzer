# Methodology Report — AI Blood Smear Analyzer

This report walks the actual pipeline used to produce the models in this
build, in the order it was executed. Every number below is either taken
verbatim from a generated JSON/CSV artifact in this repository or from a
directly-inspected system fact -- none are estimated or assumed. Where a
number is not yet available (a model not yet trained), that is stated.

```
Dataset  ->  Preprocessing  ->  Training  ->  Validation  ->  Testing  ->  Evaluation  ->  Results  ->  Limitations
```

## 1. Dataset

Primary sources (full citations and license terms in
`dataset_feasibility_audit.md` and `datasets/dataset_registry.csv`):
- PBC (Acevedo et al. 2020, Data in Brief) -- CC BY 4.0, 17,092 single-cell
  images across 8 native classes.
- TXL-PBC (Gan et al. 2025, Scientific Data) -- CC BY 4.0, 1,260 whole-smear
  images / 18,143 bounding boxes (WBC/RBC/Platelet), pre-split
  train(882)/val(252)/test(126).

Both were downloaded directly from their official repositories (Mendeley
Data DOI 10.17632/snkd93bnjr.1; Figshare DOI 10.6084/m9.figshare.27073186.v8)
in this session, verified by exact byte size (PBC: 281,366,219 bytes) and
exact per-class image counts (PBC: 1218+3117+1551+2895+1214+1420+3329+2348
= 17,092, matching the paper exactly).

## 2. Preprocessing

`scripts/extract_rbc_crops_from_txlpbc.py` extracts single-cell RBC and
platelet crops from TXL-PBC's YOLO-format bounding boxes (PBC has no
mature-RBC single-cell images -- see architecture_decision.md section
"open items" for why this is a documented domain-shift risk, not an
oversight). Actual extraction yielded (real counts from this run):
16,302 RBC crops, 543 platelet crops, 1,298 unclassified-WBC crops (train
908/val 257/test 133 for WBC; 11,220/3,383/1,699 for RBC; 382/112/49 for
platelet).

`scripts/build_classification_dataset.py` then pools PBC's 6 usable target
classes with the TXL-PBC RBC crops, per `configs/label_schema.yaml`:
- Quality control removed 1 corrupted/hidden file (a macOS `.DS_*` artifact
  accidentally bundled in the PBC zip).
- Duplicate detection (8x8 difference hash) removed 14 near-duplicate images
  found within PBC (listed by exact path in
  `datasets/processed/classification_v1/leakage_and_qc_report.json`).
- Split: PBC images used a fixed-seed (42) stratified random split at the
  image level, since PBC publishes no per-image patient/donor identifier
  (documented limitation, `dataset_feasibility_audit.md` section 9). RBC
  crops instead inherited the split already assigned to their TXL-PBC parent
  whole-smear image, so multiple crops from one source photograph never
  straddle train/val/test.

Real resulting class counts (train / val / test), from
`leakage_and_qc_report.json`:

| class | train | val | test |
|---|---|---|---|
| neutrophil | 2330 | 499 | 500 |
| eosinophil | 2177 | 466 | 467 |
| basophil | 848 | 181 | 183 |
| lymphocyte | 849 | 182 | 183 |
| monocyte | 993 | 212 | 214 |
| platelet | 1643 | 352 | 353 |
| rbc | 11220 | 3383 | 1699 |

This confirms, on real pooled data, the class-imbalance finding from the
feasibility audit: basophil and lymphocyte are the smallest classes; RBC is
by far the largest (a consequence of RBC crops coming from dense whole-smear
detection rather than curated single-cell capture).

## 3. Training

Classifier: `scripts/train_classifier.py`, MobileNetV3-Small
(ImageNet-pretrained, torchvision), fine-tuned end-to-end with a
class-weighted cross-entropy loss (inverse-frequency weights, directly
responding to the imbalance above), Adam optimizer, image size 128x128,
batch size 64, on this dev machine's CPU (no CUDA GPU present, confirmed by
direct inspection -- see architecture_decision.md section 1). Augmentation:
small rotations (+/-15 deg), brightness/contrast jitter, horizontal flip --
all realistic per spec section 18, no biologically unrealistic transforms.

Detector: `scripts/train_detector.py`, Ultralytics YOLOv8n, trained on
TXL-PBC's pre-existing train/val/test split, CPU.

## 4. Validation

Best-epoch model selection used validation macro-F1 (not raw accuracy,
given the class imbalance above) -- see spec section 21's requirement to
never report only overall accuracy.

## 5. Testing

Final metrics were computed once on the held-out test split only, after
model selection was frozen on validation data.

## 6. Evaluation

Real, measured results from the actual completed run (`evaluation/classifier_v1_evaluation.json`,
8 epochs, MobileNetV3-Small, CPU-only, ~7.5 min/epoch). Computed ONCE on the
held-out **test** split (3,599 images) -- never on train or validation data.
Live numbers are also rendered on the app's Research / Metrics page, which
reads this same JSON file, so the two never drift apart.

**Test-set headline metrics:**

| Metric | Value |
|---|---|
| Accuracy | 99.53% |
| Balanced accuracy | 99.26% |
| Macro F1 | 0.9903 |
| Weighted F1 | 0.9953 |
| MCC | 0.9935 |

**Per-class (test set, n=3599):**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| basophil | 0.934 | 1.000 | 0.966 | 183 |
| eosinophil | 1.000 | 0.994 | 0.997 | 467 |
| lymphocyte | 0.995 | 0.995 | 0.995 | 183 |
| monocyte | 0.995 | 0.977 | 0.986 | 214 |
| neutrophil | 0.996 | 0.986 | 0.991 | 500 |
| platelet | 1.000 | 0.997 | 0.999 | 353 |
| rbc | 1.000 | 1.000 | 1.000 | 1699 |

## 7. Results -- Honest Interpretation

Total test errors: 17 out of 3,599 (all other 3,582 correct). Reading the
confusion matrix directly (not just the summary metrics):

- **The single clearest weakness: basophil precision (93.4%), not recall.**
  Basophil recall is a perfect 1.000 -- the model never misses a true
  basophil -- but 13 non-basophil test images were wrongly predicted as
  basophil (3 eosinophil, 3 monocyte, 7 neutrophil), which is what drags
  precision down. This is a plausible, interpretable side effect of the
  class-weighted loss (spec section 21/`train_classifier.py`): basophil is
  the smallest training class (848 of 20,060), so its loss weight is the
  highest, which fixed the "model ignores the rare class" failure mode this
  weighting was added for, but appears to have overshot into "model leans
  toward the rare class under ambiguity" for a handful of atypical
  granulocytes. This is reported honestly rather than hidden behind the 99%+
  headline numbers, per this project's explicit instruction not to hide weak
  classes behind high overall accuracy.
- The only other confusable pairs observed: 1 lymphocyte -> monocyte, 2
  monocyte -> neutrophil, 1 platelet -> lymphocyte. Each is a single-digit
  count against a support of 183-500; not evidence of a systematic weakness,
  but reported for completeness.
- RBC and platelet reach perfect or near-perfect precision/recall. Read this
  in light of a real, separately-confirmed limitation (see the "Engineering
  QA Notes" bug below and section 8): RBC crops come from a visually very
  different acquisition pipeline (whole-smear detection crops) than the five
  WBC types and platelet (dedicated single-cell CellaVision captures) --
  RBC-vs-everything-else is a comparatively easy visual separation for the
  model, which is part of why accuracy on this specific test split is this
  high. It does not mean RBC classification is equally reliable on images
  acquired a different way (e.g., a genuine single-cell RBC photograph).
- **This test split is not patient-level held-out** for the PBC-sourced
  classes (no public per-image donor ID exists in PBC -- see
  dataset_feasibility_audit.md section 9); it is a fixed-seed stratified
  random split at the image level. The near-perfect numbers should be read
  with that in mind: they are honestly measured on this specific held-out
  split, but that split is weaker than a true unseen-patient evaluation.
- **Confidence values are NOT temperature-calibrated in this run.** The
  `TemperatureScaler` in `src/uncertainty/calibration.py` exists but
  `scripts/train_classifier.py` does not call it -- `SingleCellAnalyzer`
  therefore uses temperature=1.0 (raw softmax) everywhere by default. The
  LOW_CONFIDENCE gate (default threshold 0.55) still functions on these raw
  probabilities, but the reported percentages should be read as "the model's
  raw softmax output," not as a calibrated probability of correctness.
  Fitting the temperature scaler on the validation logits is a cheap
  (seconds, not a training run) candidate follow-up, not done in this pass
  per the instruction to prioritize a working v1 over marginal refinement.

**Verdict on v1-readiness (classifier):** recommended for the v1 demo. The
one real, honestly-identified weakness (basophil precision, driven by
false positives from other granulocytes) is well within acceptable range for
a research/education tool that already gates low-confidence predictions and
never hides its per-class numbers from the user (Research / Metrics page).

## 8. Limitations

Restated from `dataset_feasibility_audit.md` and `architecture_decision.md`
for convenience:
- RBC morphology sub-typing (echinocyte/spherocyte/sickle/target-cell/etc.)
  is not implemented -- no adequately verified public dataset was found.
- RBC classification training data comes from a different acquisition
  domain (whole-smear detection crops) than the WBC/platelet training data
  (dedicated single-cell CellaVision captures) -- a real, documented
  domain-shift risk that the per-class RBC metrics should be read in light of.
- Nucleus/cytoplasm segmentation is an unvalidated classical-CV proxy, not
  checked against the 1,145 Raabin-WBC expert masks in this build.
- PBC's train/val/test split is at the image level (no public per-patient
  key exists for PBC), not a true patient-level split -- weaker leakage
  protection than the KU-Optofil/C-NMC/AML-Cytomorphology sources that do
  have patient IDs but were not used in v1 (see feasibility audit for why).
- Basophil and lymphocyte are the smallest-N classes in the training set;
  their measured metrics should be read with that in mind.

## 9. Engineering QA Notes (bugs found and fixed during this build)

A code-review pass over the already-written pipeline (done deliberately
without competing for CPU while the classifier trained) found and fixed
several real defects, verified by tests, not just inspection:

- **Critical: `app.py` vs. `app/` package name collision.** The top-level
  entrypoint file `app.py` and the `app/` package directory shared a name.
  CPython's import resolution checks for a plain module (`app.py`) before
  falling back to a namespace package (`app/` with no `__init__.py`), so
  every `from app.utils... import ...` statement in every page silently
  failed with `ModuleNotFoundError: No module named 'app.utils'; 'app' is
  not a package` -- confirmed by direct reproduction, not just theory. This
  would have broken the entire deployed application, not merely a test.
  **Fix:** added `app/__init__.py`, `app/pages/__init__.py`,
  `app/utils/__init__.py` so `app/` is a real regular package, which
  CPython's import machinery checks ahead of a same-named module file.
  Verified by direct reproduction (`import app.utils.model_access` failed
  before the fix, succeeded after) and by the full Streamlit `AppTest` suite
  in `tests/test_app_pages.py` (0/7 passing before, all page-import-related
  failures; 7/7 passing after).
- **Grad-CAM heatmap/image size mismatch** in `app/pages/2_analyze_cell.py`:
  the heatmap is sized to the classifier's input resolution (128x128), but
  was being blended directly with the original uploaded image via
  `cv2.addWeighted`, which requires matching shapes -- this would raise on
  any upload not exactly 128x128. Fixed by resizing the heatmap to the
  source image's dimensions before blending.
- **Dead low-confidence flag for RBC/Platelet whole-smear detections** in
  `src/inference/whole_smear.py`: the flag compared a detection's confidence
  against the same hard threshold already used to filter YOLO's output, so
  the condition could never be true -- RBC/Platelet ambiguous detections
  were silently never flagged, contradicting spec section 4's requirement to
  identify uncertain detections. Fixed by introducing a separate, higher
  `low_confidence_display_threshold`. Covered by a regression assertion in
  `tests/test_end_to_end.py`.
- **Redundant/dead ternary** in `draw_annotations()` (`thickness = 1 if not x
  else 1`) that discarded the intended confident-vs-low-confidence visual
  distinction in annotated boxes. Fixed to actually vary by confidence.
- **Duplicate model loading**: the whole-smear analyzer was building a
  second, separate copy of the classifier instead of reusing the instance
  already cached for the single-cell page, doubling memory use for no
  benefit (spec section 32). Fixed to share one cached instance.
- **Unguarded corrupted-upload path** in `app/pages/5_compare_cells.py`:
  `cv2.imdecode` returning `None` for an invalid file was not checked before
  being passed to the analyzer/morphology functions, unlike the other two
  upload pages -- would have crashed instead of showing a clear error (spec
  section 33). Fixed with the same guard pattern used elsewhere.
- **Severe, demo-blocking: blur-detection threshold systematically rejected
  real RBC and platelet images.** Discovered when the real trained
  classifier's end-to-end test (`test_single_cell_pipeline_end_to_end_on_real_holdout_image[rbc]`)
  failed with `IMAGE_QUALITY_REJECTED` on a genuine, real, held-out test-set
  RBC image. Investigated by measuring the Laplacian-variance "sharpness"
  score across the *entire* real test split (3,599 images, all 7 classes),
  not just the one failing sample. At the original threshold (15.0, picked by
  intuition when the quality check was first written, before any real cell
  images existed to check it against): **53.1% of real RBC images and 32.6%
  of real platelet images measured below threshold and would have been
  wrongly rejected as "too blurry."** Root cause: RBCs and platelets are
  morphologically smooth, low-texture objects even when perfectly in focus,
  so they produce inherently low Laplacian variance -- the heuristic was
  conflating "low image texture" with "out of focus," and had never been
  validated against real cell images of every supported class before this.
  Every one of the 7 classes measured 0.0% below 3.0 (minimum observed value
  across all classes: 4.19, on rbc). **Fix:** lowered
  `BLUR_LAPLACIAN_VAR_THRESHOLD` from 15.0 to 3.0, with the full measurement
  documented inline in `src/inference/single_cell.py`. This is exactly the
  kind of fabrication-adjacent failure this project's honesty rule exists to
  prevent: a real, in-focus image being told it was blurry when it was not.
  A pre-existing synthetic test (`test_slightly_blurry_image_rejected`, now
  `test_heavily_blurred_image_rejected`) asserted a specific blur kernel
  should be rejected; that kernel's synthetic random-noise output (~4.2) was
  no longer below the corrected, real-data-validated threshold, so the test's
  blur kernel was strengthened (15x15 -> 35x35, measured ~1.6) to keep
  testing genuine blur rejection without reverting the real-data fix -- the
  test's intent was preserved, only its synthetic severity was recalibrated.
- **Same class of bug, second instance: morphology segmentation's minimum
  contour-size guard rejected 94% of real platelet images.** Found while
  smoke-testing the completed classifier end-to-end on all 7 real sample
  images (`samples/single_cell/`): classification succeeded for the
  platelet sample, but its morphology came back `segmentation_status:
  failed`. Measured the largest-detected-contour area (as % of image area)
  across 100 real test-set images per class: platelet's largest contour
  averaged ~1.1% of the crop (min observed 0.428%) -- correctly reflecting
  that platelets are genuinely small blood-cell fragments -- but the
  `_segment_cell_mask` degenerate-segmentation guard in
  `src/morphology/features.py` rejected anything below 2%, a threshold
  picked by intuition and never checked against a real platelet image
  before. Every other class measured comfortably above 3% at its worst
  5th percentile, so this was specific to platelets, not a general
  segmentation problem. **Fix:** lowered the floor from 2% to 0.3% (below
  platelet's observed minimum, still well above the near-zero/degenerate
  range), documented inline with the measurement. Verified: the same
  platelet sample now segments successfully (area_px=2168,
  circularity=0.57) and the full test suite (118 passed) is unaffected.

All of the above were confirmed with real, executed tests (`tests/test_app_pages.py`,
`tests/test_end_to_end.py`, `python -c "import ..."` reproductions, and for
the blur-threshold fix, a full-distribution measurement script run against
the actual test split), not by inspection alone -- consistent with this
project's no-fabrication rule extending to "the code works" claims, not only
to model outputs.

**Strongest available UI verification, once the classifier existed:**
`tests/test_app_pages.py::test_analyze_cell_page_full_upload_flow_with_real_image`
drives the actual Streamlit widget tree (`FileUploader.upload()`, not a
direct Python call to the analyzer) with a real held-out neutrophil sample.
Confirmed working end-to-end: real upload -> real quality check -> real
classifier forward pass -> "NEUTROPHIL" heading rendered -> 100% confidence
metric rendered -> Grad-CAM tab -> Morphology tab with real measured numbers
(area 5840px, circularity 0.751, aspect ratio 1.057, solidity 0.952,
nucleus/cell ratio 0.726 correctly labeled "unvalidated proxy" in the
rendered UI, not hidden). No unhandled exception.

## 10. Detector Results (YOLOv8n, TXL-PBC, 20 epochs, CPU, 2.247 hours)

Real, measured results from the completed training run. Two independent
measurements are reported because they measure genuinely different things --
neither is wrong, and reporting only one would hide real information:

**Ultralytics' own validation** (`evaluation/detector_v1_evaluation.json`),
computed across the full confidence range for mAP and at Ultralytics'
internal best-operating-point for P/R, on the held-out test split (126
images, 1868 GT instances):

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| all | 0.975 | 0.957 | 0.986 | 0.860 |
| WBC | 0.983 | 1.000 | 0.995 | 0.874 |
| RBC | 0.964 | 0.960 | 0.990 | 0.892 |
| Platelet | 0.978 | 0.912 | 0.975 | 0.814 |

**This project's own strict evaluation** (`scripts/evaluate_detector.py`,
`evaluation/detector_v1_detailed_evaluation.json`): real greedy IoU>=0.5
matching at the SAME confidence threshold (0.25) the deployed app actually
uses -- not an idealized threshold sweep:

| Class | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| WBC | 131 | 3 | 2 | 0.978 | 0.985 | 0.981 |
| RBC | 1676 | 576 | 23 | 0.744 | 0.986 | 0.848 |
| Platelet | 48 | 17 | 1 | 0.738 | 0.980 | 0.842 |

**Why these differ, and what it means honestly:** at the app's actual
deployed confidence threshold, precision for RBC and Platelet is
meaningfully lower than Ultralytics' own summary suggests. Recall stays
very high for both (0.986, 0.980) -- the model rarely misses a real cell.
Direct inspection on the densest real test sample (28 ground-truth cells)
confirmed the mechanism: the model predicted 33-35 raw RBC boxes where only
26 were labeled, i.e. a real tendency to over-detect/produce near-duplicate
boxes for RBCs in dense fields at this threshold, not a fabricated concern.
**This is reported as a genuine, characterized limitation, not hidden behind
the higher aggregate mAP50 numbers above** -- see spec's explicit
requirement not to hide weak performance behind aggregate metrics.

Stratified recall (own evaluation, real IoU-matched): small/medium/large
object tercile 0.984/0.986/0.989 (no meaningful size effect); boundary vs.
interior 0.991/0.981 (no meaningful edge effect); touching-another-cell vs.
isolated 0.985/0.995 (real, small penalty for touching/overlapping cells,
consistent with expectation, not severe); sparse/medium/dense image density
0.991/1.000/0.973 (a small, real recall drop in the densest real fields).
None of these stratified effects are large; the dominant, most important
finding is the RBC/Platelet precision gap described above.

**Bug found and fixed via this evaluation:** the whole-smear end-to-end test
(`tests/test_end_to_end.py`, duplicate-detection integrity check) caught a
real near-duplicate pair on the densest real test image: two YOLO-class-RBC
boxes with mutual IoU right at Ultralytics' own 0.7 NMS cutoff (0.700 on raw
model coordinates, 0.706 after this project's own pixel-rounding) both
survived the detector's internal per-class NMS -- most likely a
coordinate-space rounding effect between the model's internal 416x416
inference resolution and the original image's pixel coordinates. Direct
inspection confirmed the redundant box already carried lower confidence
(0.498, below the app's own 0.5 low-confidence display threshold) than its
counterpart (0.715), so the existing honesty-first confidence gate was
already flagging it as uncertain -- this was a real double-counting risk,
not a hidden one. **Fix:** added `_suppress_duplicate_boxes()` in
`src/inference/whole_smear.py`, an application-layer safety net (not a
model change, not a retrain) that keeps only the highest-confidence box in
any same-class group whose mutual IoU exceeds 0.65 (set with margin below
the observed 0.700/0.706 boundary case). Verified: the same strict test that
caught the bug now passes unmodified after the fix (its 0.7 assertion
threshold was never loosened).

**Real cross-model behavior observed (not a bug):** on the same dense
sample, the detector flagged one box as WBC (0.831 confidence); when that
exact crop was passed to the 7-class classifier for subtype resolution (as
designed), the classifier confidently (100%) predicted "rbc" instead. The
final reported label is "rbc" for this detection, not a WBC subtype -- this
is the pipeline working as designed (the classifier's verdict is
authoritative for WBC-subtype resolution), but it is a real, observed
instance of the already-documented RBC domain-shift limitation (RBC
classifier training crops all come from whole-smear-style images -- the
same visual domain as this WBC-detected crop -- while WBC training crops
come from dedicated single-cell captures), reported honestly rather than
smoothed over.

**Whole-smear pipeline integrity, verified on all 4 real samples (sparse 5
cells, medium 14, dense 28, plus the original held-out sample):** counts
always equal `len(detections)`; every box lies within image bounds (real
`_clamp_box` clamping); no duplicate detections after the fix above; low-
confidence detections are flagged, never silently hidden or silently
promoted to confident; morphology failures report `segmentation_status:
failed` with reasons, never a fabricated measurement.

**Detector-side performance (real, measured on this CPU):** whole-smear
analyzer (detector + shared classifier) load time 0.82s, ~503MB RSS;
full pipeline (detect + classify every WBC crop) on the densest real test
image (28 ground-truth cells, 34 detections after dedup) averaged 0.53s
(range 0.35-0.97s across 5 warm runs). No optimization performed --
comfortably within interactive latency for a research/demo tool.
