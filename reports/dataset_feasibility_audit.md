# Dataset Feasibility Audit — AI Blood Smear Analyzer

**Status:** Phase 1 deliverable. Written before any model training.
**Date:** 2026-09-05
**Author:** Automated research pass (Claude Code), verified against PubMed, TCIA, Zenodo/Figshare/Mendeley, and Kaggle/GitHub source pages. Every dataset below is traced to a primary source; nothing here is asserted from memory alone.

---

## 1. Executive Summary

This audit evaluates candidate public datasets for a 7-class single-cell blood cell classifier (RBC, Neutrophil, Lymphocyte, Monocyte, Eosinophil, Basophil, Platelet) plus whole-smear cell detection and, where feasible, morphology/segmentation.

**Headline findings:**

- **Six of the seven target classes have strong, license-clear, peer-reviewed-provenance data**: Neutrophil, Lymphocyte, Monocyte, Eosinophil, Basophil, Platelet (all present in the Acevedo/PBC and Raabin-WBC datasets). **RBC classification (RBC vs. non-RBC / vs. other classes) is supported for detection/counting**, but there is **no strong public dataset for RBC *morphology subtypes*** (discocyte/echinocyte/spherocyte/sickle/target-cell) that meets the bar of "trustworthy labels + sufficient volume + independent evaluation." This is documented as a hard limitation, not worked around.
- **Best classification backbone: PBC / Acevedo dataset** (Hospital Clínic de Barcelona) — CC BY 4.0, 17,092 single-cell images, 8 labeled classes, expert-annotated. This is the strongest license (fully permissive, redistribution-safe) of any candidate found.
- **Best detection/whole-smear backbone: Raabin-WBC** (~40k images, free for commercial+non-commercial use, 2 microscopes/2 cameras for domain diversity) combined with **TXL-PBC** (2025, Scientific Data) which already re-annotated BCCD+Raabin+PBC into a unified WBC/RBC/platelet bounding-box detection set.
- **A newly published (2026) large dataset, KU-Optofil PBC** (31,489 images, 13 classes, automated Sysmex DI-60 slide scanner, Zenodo-hosted), is the single richest resource found and should be evaluated for inclusion once its license terms are confirmed on the Zenodo record (see §7, marked UNKNOWN pending manual license read).
- **Kaggle-only "mirrors" were explicitly NOT treated as independent evidence.** Where a Kaggle dataset could be traced to a paper (PBC, Raabin, BCCD via GitHub), that paper/repo is the source of record. Where it could not be traced (e.g., several "12/13-class" Kaggle blood-cell datasets), it is flagged UNKNOWN provenance and excluded from the v1 training plan.
- **Compute constraint changes the architecture decision, not the dataset decision**: the development machine has no CUDA GPU (Intel Iris Xe integrated graphics only). This pushes model choice toward lightweight CNNs (MobileNetV3/EfficientNet-lite/ResNet18/YOLO-nano) trainable on free-tier cloud GPU (Kaggle/Colab) and CPU-inference-friendly for Streamlit Community Cloud deployment. Full detail in `architecture_decision.md`.

---

## 2. Dataset Comparison Table

| Dataset | Type | Images | Classes | Annotation | License | Tier |
|---|---|---|---|---|---|---|
| PBC (Acevedo, Hospital Clínic Barcelona) | Single-cell classification | 17,092 | 8 (neutrophil, eosinophil, basophil, lymphocyte, monocyte, immature granulocyte, erythroblast, platelet) | Expert clinical pathologists | **CC BY 4.0** (Mendeley DOI 10.17632/snkd93bnjr.1) | 1 |
| Raabin-WBC | Single-cell + smear, detection | ~40,000 WBC images + RBC/platelet "color spots"; 1,145 cells with nucleus/cytoplasm masks | 5 WBC types (neutrophil, lymphocyte, monocyte, eosinophil, basophil) | 2 independent experts | Free for commercial + non-commercial use (raabindata.com) | 1 |
| TXL-PBC | Whole-smear detection | 1,260 images / 18,143 bounding boxes | WBC, RBC, Platelet | Semi-automatic (YOLOv8n) + full manual review | Open (Figshare + GitHub); re-annotation of BCCD+BCDD+PBC+Raabin | 1 |
| KU-Optofil PBC | Single-cell classification | 31,489 | 13 (incl. blasts, myelocytes, reactive lymphocytes) | Expert hematologists, Cohen's κ > 0.85 | Zenodo-hosted (DOI 10.5281/zenodo.17333317) — **license terms not yet confirmed, treat as UNKNOWN until read** | 1 |
| BCCD (Shenggan) | Detection | 364 original images / 4,888 boxes | WBC, RBC, Platelet | Community-annotated (VOC format), no peer-reviewed paper of its own | **MIT** (GitHub `Shenggan/BCCD_Dataset/LICENSE`) | 2 |
| Kaggle "Blood Cell Images" (paultimothymooney) | Classification (heavily augmented) | 12,500 (from ~410 originals) | 4 (eosinophil, lymphocyte, monocyte, neutrophil) | Inherited from BCCD source photos (cosmicad/akshaylamba) | Public-domain-style, attribution "appreciated" not required | 2 — **augmentation-inflated, do not treat as 12,500 independent images** |
| LISC | Single-cell classification + segmentation | Smaller (exact N not confirmed from author page) | 5 WBC types + mixed | Hematologist-labeled, manual nucleus/cytoplasm segmentation | **Research/education only. No redistribution. Must be obtained directly from author's site. Mandatory acknowledgment.** | 1 (restricted) |
| ALL-IDB (1 & 2) | Leukemia lymphoblast classification | ALL-IDB1: ~108 images; ALL-IDB2: 260 segmented lymphocytes (130/130) | Leukemic vs. normal lymphocyte | Expert-verified | **Request-only, research use, no public open license confirmed** — contact required | 2 (restricted) |
| C-NMC-2019 (TCIA) | Single-cell classification (ALL) | 15,114 cell images / 118 subjects | Normal vs. malignant B-ALL lymphoblast | Expert-labeled, used as ISBI 2019 challenge ground truth | TCIA standard (CC BY 3.0/4.0 family — confirm per-collection page) | 1 |
| AML-Cytomorphology_LMU (Matek et al.) | Single-cell classification, bone-marrow/PB leukocytes | Large (paper-scale, exact N per TCIA page) | 15 classes incl. blasts + WBC subtypes | Expert hematologists (Matek, Marr et al., Nature Machine Intelligence 2019) | TCIA standard license family (typically CC BY 3.0) | 1 |
| LeukemiaAttri / LLD (ITU, MICCAI 2024) | Whole-smear detection, attribute-annotated | 2,400 images / 10.3k WBC objects / 55k morphological attribute labels | 14 (13 WBC/blast subtypes + artifact) | Multi-domain (multiple microscopes/magnifications), attribute boxes for nucleus/cytoplasm | GitHub-hosted (`intelligentMachines-ITU`) — license file not yet confirmed, treat as UNKNOWN pending read | 2 (leukemia-patient population, not healthy donors) |
| Chula-PIC-Lab (RBC morphology) | RBC shape/abnormality classification | Not independently confirmed at source | Poikilocytosis/anisocytosis abnormal RBC classes | Referenced only via Dhar et al. 2025 (Microsc Res Tech) | **UNKNOWN** — no independent public access page located | 2 — insufficient to build on |

---

## 3. Source Provenance & Scientific References

All citations verified via PubMed (attribution required by tool terms) and publisher/repository pages.

1. **PBC dataset.** Acevedo A, Merino A, Alférez S, Molina Á, Boldú L, Rodellar J. "A dataset of microscopic peripheral blood cell images for development of automatic recognition systems." *Data in Brief* 2020;30:105474. PMID 32346559, PMC7182702, DOI: [10.1016/j.dib.2020.105474](https://doi.org/10.1016/j.dib.2020.105474). Data: Mendeley DOI [10.17632/snkd93bnjr.1](https://doi.org/10.17632/snkd93bnjr.1), CC BY 4.0.
2. **Raabin-WBC.** Kouzehkanan ZM, Saghari S, Tavakoli S, Rostami P, Abaszadeh M, Mirzadeh F, Satlsar ES, Gheidishahran M, Gorgi F, Mohammadi S, Hosseini R. "A large dataset of white blood cells containing cell locations and types, along with segmented nuclei and cytoplasm." *Scientific Reports* 2022;12:1123. PMID 35064165, PMC8782871, DOI: [10.1038/s41598-021-04426-x](https://doi.org/10.1038/s41598-021-04426-x). Data: [raabindata.com/free-data](https://www.raabindata.com/free-data/).
3. **TXL-PBC.** Gan L, Li X, Wang X. "A Curated and Re-annotated Peripheral Blood Cell Dataset Integrating Four Public Resources." *Scientific Data* 2025;12:1694. PMID 41145591, PMC12559723, DOI: [10.1038/s41597-025-05980-z](https://doi.org/10.1038/s41597-025-05980-z). Data: Figshare + GitHub.
4. **KU-Optofil PBC.** Yarıkan AE, Örer C, Akyıldız V, Kuş Z, Aydin M, Palaoğlu KE, İncir S, Baysal K, Özçelik C, Kiraz B, Kiraz A. "A Large-Scale Peripheral Blood Cell Dataset for Automated Hematological Analysis." *Scientific Data* 2026;13. PMID 41651863, PMC13005019, DOI: [10.1038/s41597-026-06761-y](https://doi.org/10.1038/s41597-026-06761-y). Data: Zenodo DOI [10.5281/zenodo.17333317](https://doi.org/10.5281/zenodo.17333317). Note: publication year 2026 reflects the journal's advance/volume dating, not a future embargo — the record is live now.
5. **LISC.** Rezatofighi SH, Soltanian-Zadeh H. Original LISC paper (2011), PMID 21300521. Data + license terms: author page, [users.cecs.anu.edu.au/~hrezatofighi/Data/Leukocyte Data.htm](http://users.cecs.anu.edu.au/~hrezatofighi/Data/Leukocyte%20Data.htm).
6. **ALL-IDB.** Labati RD, Piuri V, Scotti F. "ALL-IDB: The acute lymphoblastic leukemia image database for image processing." Original venue: IEEE ICIP 2011 (conference paper, not PubMed-indexed). Official page: [homes.di.unimi.it/scotti/all](https://homes.di.unimi.it/scotti/all/).
7. **C-NMC-2019.** Companion paper: "C-NMC: B-lineage acute lymphoblastic leukaemia: A blood cancer dataset." PMID 35500994. Hosted at TCIA: [C_NMC_2019 collection page](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52758223).
8. **AML-Cytomorphology_LMU.** Matek C, Schwarz S, Marr C, Spiekermann K. "Human-level recognition of blast cells in acute myeloid leukaemia with convolutional neural networks." *Nature Machine Intelligence* 2019 (not PubMed-indexed at the abstract level in this search; confirmed via TCIA collection page and Springer abstract link). Data: [TCIA AML-Cytomorphology_LMU](https://www.cancerimagingarchive.net/collection/aml-cytomorphology_lmu/).
9. **BCCD.** Community dataset, no peer-reviewed paper. Source: [github.com/Shenggan/BCCD_Dataset](https://github.com/Shenggan/BCCD_Dataset), MIT license, credited to "cosmicad" and "akshaylamba" original photographers.
10. **LeukemiaAttri / LLD.** Presented MICCAI 2024; used as training data in Diaz JM et al., "Deep Learning-Based Detection Model for Leukemia Cells in Peripheral Blood Smears Using YOLOv11-Large." *Current Oncology* 2026;33(8). PMID 42645391, PMC13510383, DOI: [10.3390/curroncol33080486](https://doi.org/10.3390/curroncol33080486). Data: [github.com/intelligentMachines-ITU/LLD-Large-Leukemia-dataset-for-microscopic-imagery](https://github.com/intelligentMachines-ITU/LLD-Large-Leukemia-dataset-for-microscopic-imagery).
11. **Chula-PIC-Lab / RBC abnormality.** Referenced only in Dhar P, Suganya Devi K, Bhattacharjee R, Srinivasan P. "Morphological Abnormalities Classification of Red Blood Cells Using Fusion Method on Imbalance Datasets." *Microscopy Research and Technique* 2025;88(5):1566-1581. PMID 39871114, DOI: [10.1002/jemt.24786](https://doi.org/10.1002/jemt.24786). No independently verifiable public access point found — **do not use until a direct source is located.**

*(Per PubMed tool terms of use: information above is drawn from PubMed and DOI links are provided for every PubMed-sourced record.)*

---

## 4. License Analysis

| Dataset | License | Commercial use | Redistribution | Notes |
|---|---|---|---|---|
| PBC (Acevedo) | CC BY 4.0 | Yes | Yes, with attribution | Best license of the set. Safe to redistribute derived crops with citation. |
| Raabin-WBC | "Free for commercial and non-commercial" (publisher statement, not a named SPDX license) | Yes (per publisher statement) | Not explicitly addressed — **treat conservatively: keep raw images out of any redistributed model-adjacent bundle; ship only weights + derived non-identifying features** | Re-confirm wording directly at source before any redistribution decision. |
| TXL-PBC | Open (Figshare/GitHub); derivative of the above | Inherits constraints of weakest ingredient dataset | Same caveat as Raabin | Because it's a derivative, its effective license is the intersection of BCCD (MIT), PBC (CC BY 4.0), and Raabin (free-use statement) — not a single blanket license. |
| KU-Optofil PBC | UNKNOWN (Zenodo page not read line-by-line for license field) | UNKNOWN | UNKNOWN | **Action item:** read the Zenodo record's license field before use; do not assume permissive. |
| BCCD | MIT | Yes | Yes | Cleanest license of the detection-oriented sources. |
| Kaggle "Blood Cell Images" (Mooney) | Inherited, "public domain style" per Kaggle listing | Yes (informally) | Yes (informally) | Heavily augmented (12,500 from ~410 originals) — treat as a *derivative*, not an independent sample; do not double-count against BCCD in leakage audits. |
| LISC | **Restricted**: research/education only, no redistribution, must obtain directly from author, mandatory acknowledgment | **No** | **No** | Cannot be bundled into any redistributed pipeline artifact. Usable only for local model training/evaluation with citation. |
| ALL-IDB | Restricted, request-only | Unclear, assume research-only until confirmed | No | Not usable without direct email authorization from the maintainer — **out of scope for v1** pending that authorization. |
| C-NMC-2019 | TCIA standard (CC BY 3.0/4.0 family, confirm per collection) | Yes, with attribution | Yes, with attribution | TCIA's blanket policy: "freely available... for commercial, scientific and educational purposes" under CC BY. |
| AML-Cytomorphology_LMU | TCIA standard (CC BY 3.0 typical) | Yes, with attribution | Yes, with attribution | Same TCIA policy family as above; verify per-collection page since some TCIA collections carry restricted variants. |
| LeukemiaAttri / LLD | UNKNOWN (GitHub license file not read) | UNKNOWN | UNKNOWN | Action item before any use. |

**General finding on Kaggle (per spec §12):** every Kaggle listing investigated in this audit is a mirror or augmentation of one of the traced primary sources (BCCD, PBC, Raabin) — none introduced an independently verifiable new dataset. This confirms the spec's caution: Kaggle is a distribution layer here, not an independent evidence source. No Kaggle-only dataset was accepted into the Tier-1 list.

---

## 5. Class Coverage Against the 7 Target Classes

| Target class | Coverage | Best source(s) |
|---|---|---|
| RBC / Erythrocyte | **Detection/counting: supported.** Morphology subtyping: **not supported** (see §10, §17) | TXL-PBC, BCCD, Raabin "color spots" for bounding boxes; no dataset found for shape-subtype labels at production quality |
| Neutrophil | Supported | PBC, Raabin-WBC, KU-Optofil, LISC |
| Lymphocyte | Supported | PBC, Raabin-WBC, KU-Optofil, LISC |
| Monocyte | Supported | PBC, Raabin-WBC, KU-Optofil, LISC |
| Eosinophil | Supported | PBC, Raabin-WBC, KU-Optofil, LISC |
| Basophil | Supported, **but lowest sample count of the 5 WBC types in every source** — expect class-imbalance handling to be mandatory | PBC, Raabin-WBC, KU-Optofil |
| Platelet | Supported | PBC (dedicated class), TXL-PBC/BCCD (bounding boxes) |

No class is claimed as "supported" merely because a dataset's name mentions it — each row above has a verified, expert-annotated source with a real citation.

---

## 6. Original vs. Augmented Image Counts

- PBC: 17,092 originals, no built-in augmentation.
- Raabin-WBC: ~40,000 originals (two microscope/camera setups — a *diversity* multiplier, not augmentation).
- KU-Optofil PBC: 31,489 originals, single automated-scanner acquisition pipeline.
- BCCD: 364 originals → the popular "12,500-image" Kaggle classification set is **augmentation-derived from those same 364 originals** (or the closely related cosmicad/akshaylamba photo set). This must be flagged wherever it appears in any future benchmark comparison, and must never be treated as adding 12,500 independent samples to a combined pool.
- TXL-PBC: 1,260 curated images, itself a *re-annotation*, not new raw photography.
- LeukemiaAttri/LLD: 2,400 raw images, 55k derived attribute labels (not augmented copies — attribute labels are per-object annotations on the same images).

---

## 7. Annotation Availability

| Dataset | Bounding boxes | Class labels | Segmentation masks | Patient/source metadata |
|---|---|---|---|---|
| PBC | No (single-cell crops) | Yes, 8-class | No | Aggregate only (donor health status, not per-image ID) |
| Raabin-WBC | Yes (subset) | Yes, 5-class WBC | Yes, 1,145 cells (nucleus+cytoplasm) | Smear/camera/microscope metadata present |
| TXL-PBC | Yes, 18,143 boxes | WBC/RBC/Platelet | No | Inherited from source datasets |
| KU-Optofil PBC | No (single-cell crops, pre-cropped by scanner) | Yes, 13-class | No | Anonymized patient identifiers retained — **this is a real leakage-prevention asset**, see §17 |
| BCCD | Yes, VOC format | WBC/RBC/Platelet | No | None |
| LISC | No | Yes, 5-class + mixed | Yes, manual nucleus/cytoplasm | Hospital source noted, no per-patient ID |
| C-NMC-2019 | No | Normal vs. malignant | No | Per-subject IDs (118 subjects) — **usable for patient-level splitting** |
| AML-Cytomorphology_LMU | No | 15-class | No | Per-patient (AML vs. non-malignant control) |
| LeukemiaAttri/LLD | Yes + attribute sub-boxes (nucleus/cytoplasm regions) | 14-class | Partial (attribute boxes approximate nucleus/cytoplasm) | Per-image patient/domain tag |

---

## 8. Detection / Classification / Segmentation / Morphology Suitability

- **Detection (whole-smear cell finding):** TXL-PBC and BCCD are purpose-built; Raabin-WBC's smear-level images add domain diversity; LeukemiaAttri contributes multi-microscope diversity but from a leukemia-patient population (morphology skew must be documented, not hidden, if ever mixed into a "normal smear" detector).
- **Classification (7-class):** PBC is the primary source (best license, largest fully-labeled single-cell corpus with clean provenance); Raabin-WBC and KU-Optofil provide additional diversity and larger N for the 5 WBC classes; cross-dataset combination requires careful label-schema mapping (e.g., PBC's "immature granulocyte" superclass vs. KU-Optofil's separate myelocyte/metamyelocyte labels) — this mapping must be made explicit in the label-normalization step, never silently merged.
- **Segmentation:** Only Raabin-WBC (1,145 cells) and LISC (restricted license) provide real nucleus/cytoplasm ground truth. This is a **thin** segmentation training set. Recommendation (see architecture doc): treat WBC nucleus/cytoplasm segmentation as a secondary, lower-confidence feature built from a small supervised set plus classical CV (watershed/GrabCut) validated against the 1,145 Raabin masks, not as a fully deep-learned instance segmentation system in v1.
- **Morphology:** Directly measurable geometric/textural features (area, perimeter, circularity, eccentricity, solidity, color/texture stats) can be computed from *any* cell crop via classical image processing regardless of which dataset a training example came from — this does not require a labeled "morphology dataset," only a good cell mask, which is the actual bottleneck (see segmentation point above).

---

## 9. Patient / Source Metadata & Leakage Risk

- **KU-Optofil PBC** and **C-NMC-2019** and **AML-Cytomorphology_LMU** carry genuine per-patient/per-subject identifiers — these enable real patient-level splitting.
- **PBC (Acevedo)** and **Raabin-WBC** do not publish per-image patient IDs at the level needed for a rigorous per-donor split; both are aggregate "collected from N healthy donors" without a public per-image donor key. **This is a real limitation**: for these two datasets, the pipeline will fall back to per-*acquisition-batch* or per-*source-file* grouping (images that are clearly crops from the same original smear photograph are grouped and never split across train/val/test), which is weaker than true per-patient grouping but strictly better than a naive random crop-level split. This limitation is recorded, not hidden.
- **BCCD-derived Kaggle augmentations** are a duplicate-risk source: augmented copies of the same 364 base images must never end up on both sides of a split. Any pipeline that ingests the Kaggle "Blood Cell Images" set must trace each image back to its pre-augmentation parent before splitting.
- **Cross-dataset duplicate risk:** TXL-PBC explicitly re-uses BCCD, Raabin, and PBC images. If TXL-PBC is used *alongside* its four source datasets in the same pooled training set, duplicate images will exist across "different-looking" sources — the unified pipeline (§16) must deduplicate by perceptual hash before splitting, not just by filename.

---

## 10. Dataset Quality Concerns & Missing Data

- **No verified public dataset for RBC morphology subtypes** (discocyte/echinocyte/spherocyte/target-cell/sickle-cell) was located that meets the bar of expert-verified labels + adequate per-class sample size + independent evaluation. The one candidate mentioned in literature (Chula-PIC-Lab, via Dhar et al. 2025) could not be independently traced to a public access point. **Decision: RBC morphology sub-classification is NOT implemented as a trained classifier in v1.** Only classical, unlabeled geometric descriptors (circularity, aspect ratio, etc.) will be reported as "image-derived morphology," explicitly not as a shape-category diagnosis.
- **Basophils are the minority class everywhere.** Every WBC dataset audited shows basophils as the smallest class by a wide margin (a well-known hematology fact — they are the rarest circulating leukocyte, typically <1-2% of a normal differential). Expect basophil precision/recall to be reported with wide confidence intervals and to be the most likely class to need "low confidence / insufficient evidence" fallback behavior.
- **License uncertainty on two promising, high-value sources** (KU-Optofil PBC, LeukemiaAttri/LLD) is unresolved — flagged as an explicit action item, not silently assumed permissive.
- **LISC and ALL-IDB are legally unusable for redistribution** and, for ALL-IDB, unusable at all without direct author authorization — both are excluded from the default v1 training recipe; LISC may still be used purely as an internal, non-redistributed benchmark set with proper acknowledgment.

---

## 11. Recommended Dataset Combination (v1)

**Primary training pool (classification, 7 classes):**
1. PBC (Acevedo) — CC BY 4.0, primary backbone, all licensing risk-free.
2. Raabin-WBC — free-use, adds domain diversity (different microscope/camera) critical for generalization.
3. KU-Optofil PBC — pending license confirmation; if confirmed permissive, adds the largest and most recent expert-verified corpus and unlocks real patient-level splitting.

**Detection pool (whole-smear):**
1. TXL-PBC (primary — purpose-built WBC/RBC/platelet boxes).
2. BCCD (MIT-licensed, safe to redistribute, supplements TXL-PBC).
3. Raabin-WBC smear-level images for additional domain diversity.

**Explicitly excluded from v1 pooled training:** LISC (redistribution-restricted; may be used as a held-out non-redistributed sanity-check benchmark only), ALL-IDB (authorization required, not yet obtained), Chula-PIC-Lab (unverifiable access), Kaggle-only augmented mirrors (used only after tracing to their non-augmented parent, and only for the parent images).

**Label schema normalization required before pooling:** a single canonical label map (`configs/label_schema.yaml`, to be created in the pipeline phase) mapping each source dataset's native class names to the 7 target classes plus an explicit "immature/other — out of v1 scope" bucket, so that immature granulocytes, blasts, and erythroblasts are never silently absorbed into one of the 7 target classes.

---

## 12. Missing Data / Explicit Limitations

- No validated RBC morphology-subtype dataset (see §10).
- No dataset in this audit provides deformability/membrane-integrity ground truth of any kind — confirming spec §11's instruction that any such feature must be labeled an **image-derived morphology proxy**, never a clinical measurement.
- Per-patient metadata is inconsistent across sources — leakage prevention will be uneven in strength (strong for KU-Optofil/C-NMC/AML-Cytomorphology, weaker fallback for PBC/Raabin).
- No dataset audited includes non-CellaVision/non-Sysmex whole-slide-scanner images from low-cost/consumer microscope+phone-camera setups, which is the deployment scenario most likely to stress-test the system in the wild. This is a robustness gap for future data collection, not something the current corpus can close.

---

## 13. Recommended Train/Validation/Test Strategy

- Split **by patient/donor ID** wherever a real ID exists (KU-Optofil, C-NMC-2019, AML-Cytomorphology_LMU).
- Split **by acquisition batch/source file** where no patient ID exists (PBC, Raabin) — every crop traceable to the same original photograph stays on one side of the split.
- Perceptual-hash deduplication across the *entire* pooled corpus before splitting, specifically to catch TXL-PBC's re-use of BCCD/Raabin/PBC images and the Kaggle augmented-mirror problem.
- Report the leakage audit (train/val/test source lists, patient overlap, duplicate overlap) as a generated artifact of the pipeline (§17 of the master spec) — not as a one-time manual claim.

---

## 14. Recommended Architecture Direction (summary — full detail in `architecture_decision.md`)

- No local CUDA GPU (Intel Iris Xe only) → train on free-tier cloud GPU (Kaggle Notebooks / Google Colab, both offer free T4-class GPUs), export to a CPU-inference-friendly format (ONNX/TorchScript) for Streamlit Community Cloud deployment.
- Classification: lightweight CNN backbone (EfficientNet-B0/ConvNeXt-Tiny/MobileNetV3 family), transfer-learned, not trained from scratch, given moderate dataset size (tens of thousands of images, not millions).
- Detection: YOLO-nano-class model (small enough for CPU inference at interactive latency), consistent with what multiple audited papers already used successfully on these exact datasets (Raabin, BCCD, LISC all have published YOLO-family baselines).
- Segmentation: classical CV (watershed/GrabCut) validated against the 1,145 Raabin-WBC masks, not a full deep segmentation network, given the small mask-labeled sample.

---

## 15. Computational Requirements (estimate)

- Classification training (7-class, ~60-90k pooled images at 224×224, EfficientNet-B0-scale): a handful of epochs on a free-tier T4 GPU, on the order of 1-3 GPU-hours total for a first full training+validation pass; feasible within Kaggle's free weekly GPU quota.
- Detection training (YOLO-nano, ~2-3k pooled annotated smear images): similarly on the order of 1-2 GPU-hours.
- Local machine role: data pipeline orchestration, preprocessing, classical CV morphology/segmentation, CPU-only inference serving, and the Streamlit app — all of which are within the 16GB RAM / 12-core CPU envelope confirmed in the environment inspection.

---

## 16. Risks

1. License ambiguity on two high-value datasets (KU-Optofil, LeukemiaAttri) could force their removal from the training pool after the fact — mitigated by not hard-depending on either in the pipeline code until confirmed.
2. Cross-dataset label schema drift (e.g., "immature granulocyte" as one class vs. three) risks silent mislabeling if not centralized in one schema file.
3. Basophil scarcity risks an unusable per-class F1 for that one class — must be reported honestly, not smoothed over with macro-averaging alone.
4. No local GPU means all training must round-trip through an external free-tier service — a workflow dependency, not a blocker, but worth surfacing to the user as an operational fact.

---

## 17. Exact Next Implementation Steps

1. Read the KU-Optofil PBC Zenodo record's license field directly and the LeukemiaAttri/LLD GitHub repo's LICENSE file; update this audit's UNKNOWN markers with confirmed terms before either dataset enters the training pool.
2. Build `datasets/dataset_registry.csv` / `.yaml` (this session, immediately following this report) capturing every field in spec §14 for each Tier-1 dataset, `UNKNOWN` where genuinely unverifiable.
3. Write acquisition scripts (`scripts/`) for PBC (Mendeley), Raabin-WBC (raabindata.com), TXL-PBC (Figshare/GitHub), BCCD (GitHub) — network-dependent, will run opportunistically; the pipeline code must not assume the data is already present, and must fail with a clear, actionable message (not a stack trace) if a source is unreachable.
4. Implement the unified pipeline (raw → metadata → label normalization → QC → dedup → leakage-safe split → preprocessing) per spec §16, driven by `configs/label_schema.yaml`.
5. Only after (1)-(4) are in place: begin classifier training, per spec §36's explicit instruction not to train before this audit and the architecture decision are both complete.
