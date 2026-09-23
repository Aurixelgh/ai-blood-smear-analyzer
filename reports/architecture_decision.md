# Architecture Decision Record — AI Blood Smear Analyzer

**Status:** Phase 1 deliverable, written immediately after `dataset_feasibility_audit.md`, before any model training.
**Date:** 2026-09-05

This document explains the proposed system architecture and the evidence behind each choice. Where evidence was insufficient to make a confident call, that is stated explicitly rather than guessed.

---

## 1. Hardware & Environment Constraints (measured, not assumed)

Directly inspected on the development machine in this session:

- CPU: 13th-gen Intel Core i5-13500H, 12 cores / 16 threads
- GPU: Intel Iris Xe integrated graphics only — **no NVIDIA/CUDA GPU present**
- RAM: 15.75 GB
- Disk: ~84 GB free on C:, ~298 GB free on E:
- Python 3.14.7, pip 26.2.1, Git 2.55.0
- `pip install torch --dry-run` resolves a `torch-2.14.0-cp314-cp314-win_amd64.whl` (CPU build) — confirming local PyTorch works for CPU-only inference/light training, but there is no local path to GPU-accelerated training.

**This single fact — no local GPU — is the dominant constraint on every model-architecture decision below.** It rules out training large detection/segmentation/ViT models locally in reasonable time, and it means the deployed model must run acceptably fast on CPU (since Streamlit Community Cloud, the target deployment per spec §29, also provides CPU-only compute).

---

## 2. System Architecture Overview

```
MICROSCOPE IMAGE
       |
IMAGE QUALITY ASSESSMENT (classical CV: blur/brightness/resolution checks)
       |
PREPROCESSING (color normalization, resize, stain-aware augmentation at train time)
       |
CELL DETECTION (YOLO-nano family; whole-smear mode only)
       |
CELL CROPPING (from detections) -- OR -- direct single-cell input (Mode A)
       |
CLASSIFICATION (lightweight CNN, 7-class + immature/other bucket)
       |
MORPHOLOGY EXTRACTION (classical CV on cell mask: shape/color/texture descriptors)
       |
NUCLEUS/CYTOPLASM SEGMENTATION (classical watershed/GrabCut, validated against
       the 1,145 Raabin-WBC masks; deep U-Net deferred, see §6)
       |
EXPLAINABILITY (Grad-CAM on the CNN classifier)
       |
UNCERTAINTY (softmax temperature scaling + confidence thresholding; explicit
       "low confidence" output path)
       |
VISUALIZATION (Streamlit UI: annotated image, cell explorer, comparison, morphology)
```

This matches the master spec's suggested pipeline (spec §20) with one explicit deviation: full deep instance segmentation is **deferred**, replaced with classical CV validated against the small labeled mask set, because the audit found only 1,145 mask-labeled WBC crops (Raabin-WBC) — too few to train a trustworthy segmentation network from scratch, and not worth the risk of an unvalidated model fabricating nucleus/cytoplasm boundaries.

---

## 3. Detector Choice: YOLO-nano family (Ultralytics YOLOv8n / YOLOv11n)

**Alternatives considered:** RT-DETR, Faster R-CNN, SSD, larger YOLO variants (s/m/l).

**Evidence for the decision:**
- Every audited paper that benchmarked blood-cell detection on these exact datasets (Raabin-WBC, BCCD, LISC) used a YOLO-family one-stage detector and reported strong results — e.g., Han et al. 2023 (*Comput Biol Med*, DOI [10.1016/j.compbiomed.2023.106606](https://doi.org/10.1016/j.compbiomed.2023.106606)) built a YOLO-based one-stage detector specifically for WBC microscopy across Raabin-WBC/BCCD/LISC; Yao et al. 2021 (*J Healthcare Eng*, DOI [10.1155/2021/1615192](https://doi.org/10.1155/2021/1615192)) directly compared Faster R-CNN vs. YOLOv4 on BCCD and found YOLOv4 reached 95.75% accuracy at 60 FPS vs. Faster R-CNN's 96.25% at much lower throughput; Diaz et al. 2026 (*Curr Oncol*, DOI [10.3390/curroncol33080486](https://doi.org/10.3390/curroncol33080486)) used YOLOv11-Large on the LeukemiaAttri/LLD dataset and reached mAP50 93.9%.
- The pattern across the literature is consistent: one-stage YOLO detectors are the dominant, well-validated choice for this exact image domain, and they trade a small amount of two-stage-model accuracy for a large inference-speed advantage — which matters here because inference must run acceptably on CPU in Streamlit Cloud.
- **Nano-scale variant specifically** (not the larger YOLO variants used in the papers above, which assumed GPU inference) is chosen because this project's inference target is CPU, not a research GPU cluster. This is this project's own judgment call, not something directly evidenced in the literature (none of the audited papers benchmarked nano-scale variants) — flagged as a deliberate, documented deviation, to be validated empirically once training data is assembled (§7 below commits to measuring actual CPU latency before finalizing).

---

## 4. Classifier Choice: Lightweight transfer-learned CNN (EfficientNet-B0 / MobileNetV3-Large / ResNet18 — final pick by empirical bake-off)

**Alternatives considered:** Vision Transformers (ViT/Swin), ConvNeXt-Base/Large, training a CNN from scratch.

**Evidence for the decision:**
- Dataset scale from the audit (tens of thousands of images pooled, not millions) favors transfer learning over training from scratch or training a data-hungry ViT — this is standard, well-evidenced ML practice for small-to-medium medical imaging datasets, and is exactly what the audited literature does: Chen et al. 2022 (*BMC Bioinformatics*, DOI [10.1186/s12859-022-04824-6](https://doi.org/10.1186/s12859-022-04824-6)) used pre-trained ResNet+DenseNet with transfer learning on WBC classification across LISC/BCCD/Raabin-WBC and explicitly cite sample insufficiency as the reason.
- **No single "winner" architecture is evidenced strongly enough across the literature to hard-commit today** — different papers report different backbones winning on different dataset combinations (ResNet+DenseNet hybrid in Chen et al.; plain SVM+handcrafted features beating pretrained CNNs on generalization in Tavakoli et al. 2021, *Sci Rep*, DOI [10.1038/s41598-021-98599-0](https://doi.org/10.1038/s41598-021-98599-0)). Given no local GPU to run a large architecture search, the defensible choice is: **implement a small, swappable backbone bake-off (EfficientNet-B0, MobileNetV3-Large, ResNet18) on the actual pooled v1 dataset**, running each candidate on free-tier cloud GPU (§7), and report the winner with real validation numbers — not assume a winner from other papers' different data mixes. This is the "investigate further, document alternatives, choose the most defensible option" path spec §40 asks for when evidence is genuinely insufficient for a single confident pick.
- Whichever backbone wins, it must be small enough to export to ONNX/TorchScript and run at interactive latency on CPU — this rules out ConvNeXt-Large/Swin-Large regardless of accuracy, since they would make the deployed Streamlit app unusably slow on Community Cloud's CPU-only tier.

---

## 5. Explainability: Grad-CAM

**Decision:** Implement Grad-CAM (Gradient-weighted Class Activation Mapping) on the final convolutional layer of the trained classifier.

**Evidence:** Grad-CAM is already validated on this exact problem domain — Chen et al. 2022 (DOI [10.1186/s12859-022-04824-6](https://doi.org/10.1186/s12859-022-04824-6)) explicitly used Grad-CAM occlusion testing to improve interpretability of their WBC classifier on Raabin-WBC/LISC/BCCD. This directly satisfies spec §8's instruction to investigate and implement Grad-CAM-style explainability where appropriate, with a precedent specific to this exact data domain rather than a generic justification.

**Boundary to respect (spec §8):** Grad-CAM output is "model evidence" (what the network attended to), and must be presented as a distinct category from (a) image-derived measurements (morphology engine output) and (b) general scientific morphology knowledge (static reference text). The UI must never blend these into a single unattributed "why" statement.

---

## 6. Segmentation: Classical CV first, deep learning deferred

**Decision:** Nucleus/cytoplasm segmentation for WBCs uses classical computer vision (color-space thresholding + watershed, or GrabCut initialized from the classifier's attention/bounding region), validated quantitatively (IoU/Dice) against the 1,145 expert-labeled masks in Raabin-WBC.

**Evidence against a deep segmentation network for v1:** The audit found exactly one meaningfully-sized public mask-labeled set (Raabin-WBC, 1,145 cells) and one restricted-license set (LISC) with masks. 1,145 examples is thin for training a trustworthy U-Net-class model from scratch, and LISC's license forbids redistribution, complicating any pipeline that depends on it. Rather than train an under-validated deep segmentation model and risk fabricated-looking boundaries, the defensible v1 choice is classical CV with the same 1,145 masks used purely as a **validation set**, reporting real IoU/Dice numbers. A deep segmentation model (U-Net-family) remains a documented future option if either (a) KU-Optofil PBC or another source turns out to include additional mask annotations not yet surfaced in this audit, or (b) the classical baseline's measured IoU/Dice is too low to be useful, in which case the shortfall itself becomes the evidence to revisit this decision.

---

## 7. Training Location vs. Inference Location (the core infra decision)

**Decision:** Train on free-tier cloud GPU (Kaggle Notebooks and/or Google Colab, both of which provide free T4-class GPU quota); export trained weights to ONNX or TorchScript; run all inference (classification, detection, Grad-CAM, morphology, segmentation) on CPU, both for local development and for the Streamlit Community Cloud deployment target (spec §29).

**Evidence:**
- Measured fact: this machine has no CUDA GPU. Training even a lightweight CNN on 60-90k images purely on a 12-core CPU is achievable but slow relative to a free GPU tier that costs nothing extra — there is no evidence-based reason to accept that slowdown when a free alternative exists.
- Streamlit Community Cloud's compute tier is CPU-only and memory-constrained — this was already the deployment target specified by the user (spec §29), independent of this project's own hardware, so the inference-side architecture (small models, CPU-exportable) would be required regardless of the local dev machine's GPU situation.
- This means the "GPU vs. CPU" question is really two separate, already-settled questions: *train* where GPU is free and available (cloud notebook), *serve* where the deployment target requires CPU (Streamlit Cloud) — no architecture choice needs to straddle both.

---

## 8. Uncertainty Strategy

**Decision:** Temperature-scaled softmax calibration on the classifier (a well-established, low-risk calibration method) plus an explicit confidence-threshold gate in the inference pipeline that routes low-confidence or out-of-distribution-looking inputs to a "LOW CONFIDENCE" UI state rather than forcing a top-1 label.

**Rationale tied to the audit:** Basophils were flagged in the dataset audit (§10 of the feasibility report) as the smallest class in every source dataset — this is exactly the kind of class where naive softmax confidence is known to be poorly calibrated (overconfident on minority classes trained with few examples). Calibration is not optional polish here; it is required by the specific class-imbalance finding from the audit.

---

## 9. Frontend Architecture

**Decision:** Streamlit multipage app (`app/pages/`), per spec §27's suggested navigation (Home, Analyze Cell, Analyze Blood Smear, Results, Cell Explorer, Compare Cells, Morphology, Research/Metrics, About). Session state scoped per-upload so that (per spec §32) no cached result from one image can leak into another image's display — this is an explicit correctness requirement, not just a performance one, and will be covered by an automated test (task #10).

**Alternatives considered:** FastAPI + custom JS frontend, Gradio. Rejected for v1: the spec explicitly names Streamlit as acceptable for initial deployment (spec §29) and names Streamlit Community Cloud as the target (spec §29); building a separate frontend stack would add engineering surface with no evidenced benefit for a research/education-positioned tool at this stage.

---

## 10. Deployment Architecture

- Model artifacts (ONNX/TorchScript weights) committed to the repo or pulled from a release asset at startup — not re-trained at deploy time.
- `requirements.txt` pinned to versions confirmed compatible with Streamlit Community Cloud's supported Python versions (to be confirmed against Streamlit Cloud's current supported runtime at deploy time, since this dev machine's Python 3.14 is newer than what Streamlit Cloud is likely to offer — **explicit open item**, not assumed compatible).
- No GPU dependency anywhere in the deployed inference path.

---

## 11. Summary Table of Decisions

| Component | Decision | Confidence | Evidence basis |
|---|---|---|---|
| Detector | YOLO-nano (v8n/v11n) | High for "YOLO family," medium for "nano specifically" | Multiple published benchmarks on these exact datasets; nano-scale choice is this project's own CPU-latency judgment, to be empirically verified |
| Classifier | Transfer-learned lightweight CNN, backbone chosen by bake-off | Medium — no single backbone dominates the literature | Multiple papers, no consistent winner; bake-off is the defensible response to that ambiguity |
| Segmentation | Classical CV, validated against Raabin-WBC masks; deep model deferred | High | Only 1,145 labeled masks available; insufficient for trustworthy from-scratch deep segmentation |
| Explainability | Grad-CAM | High | Directly precedented on this data domain (Chen et al. 2022) |
| Uncertainty | Temperature scaling + confidence gate | High | Directly motivated by the audit's basophil-scarcity finding |
| Training compute | Free-tier cloud GPU | High | Measured absence of local GPU |
| Inference compute | CPU-only, ONNX/TorchScript | High | Streamlit Community Cloud is CPU-only; already the specified deployment target |
| Frontend | Streamlit multipage | High | Explicitly specified acceptable/target by the project spec |

---

## 12. Open Items Carried Forward

1. ~~Confirm Streamlit Community Cloud's currently supported Python version~~ **RESOLVED**: confirmed via Streamlit's own documentation and community reports that Streamlit currently supports Python 3.10-3.14, and Community Cloud has in practice been defaulting new deployments to Python 3.14.x already -- i.e. this dev machine's Python 3.14 is directly compatible. Pinned explicitly via `runtime.txt` (`python-3.14`) per Streamlit's documented mechanism, though community reports also note `runtime.txt` is sometimes ignored in favor of the Advanced Settings dialog's own version picker at deploy time -- set it there too if `runtime.txt` is not honored.
2. Run the classifier backbone bake-off on real pooled data before committing to a single architecture in code.
3. Empirically measure YOLO-nano CPU inference latency on representative smear images before finalizing the detector size class.
4. Resolve the two license-UNKNOWN datasets (KU-Optofil PBC, LeukemiaAttri/LLD) per the feasibility audit's action items before depending on them in the training pipeline.
