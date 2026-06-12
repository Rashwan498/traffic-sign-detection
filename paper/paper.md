# Traffic Sign Classification on the Mapillary Traffic Sign Dataset:
## From HOG+SVM to CNN, Transfer Learning, and Robustness Under Synthetic Distortions

Author: _<name>_  •  Course: _<course>_  •  Date: May 17, 2026

---

## Abstract

We address fine-grained traffic-sign classification on the Mapillary Traffic
Sign Dataset v2 (MTSD), comparing a classical computer-vision pipeline
(HOG-on-edge features with a Linear SVM) against a custom convolutional
neural network trained from scratch, the same network trained with data
augmentation, and a pretrained EfficientNet-B0 fine-tuned via transfer
learning. We curate the data into **326 well-supported classes** by removing
MTSD's "other-sign" catch-all (63 % of crops) and dropping classes with fewer
than 30 training samples, yielding 67,833 crops across train / val / test
splits that honor MTSD's official partitioning. We additionally evaluate
the best CNN under a six-axis synthetic distortion suite (illumination,
rotation, occlusion, Gaussian noise, motion blur, JPEG compression) at four
severities.

The CNN approaches dominate the classical baseline by a wide margin: the
custom from-scratch network reaches **96.55 % top-1 accuracy** versus
**55.97 %** for the matched-split classical pipeline — an absolute gap of
**+40.6 percentage points** of top-1 accuracy and **+42.6 macro-F1**
points. Data augmentation gives the highest macro-F1 (**0.958**) on the
held-out test set, while ImageNet-pretrained EfficientNet-B0 fine-tuned via
transfer learning gives the highest top-1 accuracy (**96.98 %**). The best
model degrades by only ≤ 1.5 pt under severe noise / blur / JPEG / rotation
but loses 8-10 pt under heavy illumination changes and large occlusions —
directly motivating future targeted augmentation. We additionally deploy the
best model as a Gradio web app, a FastAPI service in a Docker container, and
as ONNX / Apple-Core-ML artifacts for edge inference.

**Index terms** — traffic sign recognition, convolutional neural networks,
transfer learning, robustness, HOG, MTSD, Mapillary.

---

## 1. Introduction

Reliable traffic-sign recognition is a foundational perception capability
for advanced driver assistance and autonomous driving. The Mapillary
Traffic Sign Dataset (MTSD) [MTSD] is the largest open multi-class
traffic-sign dataset to date, drawn from real-world street-level imagery in
many countries and containing **400+ sign types** with a severe long-tail
class distribution.

This paper presents an end-to-end study comparing classical and
deep-learning approaches on MTSD. Concretely, we:

1. Curate a clean **326-class** subset of MTSD with stratified train / val /
   test splits that respect MTSD's official partitioning.
2. Establish a classical Phase 1 baseline (HOG → Linear SVM via SGD) and
   re-run it on our exact splits and class set so the comparison with the
   CNN models is apples-to-apples.
3. Design, train, and analyze a custom **2.38 M-parameter** CNN
   ("BaselineCNN") from scratch, achieving **96.55 %** top-1 test accuracy.
4. Run three controlled experiments — data augmentation, transfer learning
   with EfficientNet-B0 (4.43 M params, ImageNet-pretrained), and a
   synthetic-distortion robustness sweep — to identify which design choices
   matter most.
5. Deploy the best model as a Gradio app, a Dockerized FastAPI service,
   and ONNX / Core ML exports for edge inference.

The paper is structured as follows: Section 2 reviews related work; Section
3 describes the dataset and our curation policy; Section 4 details the
methodology; Section 5 presents headline results; Section 6 analyses
robustness; Section 7 covers deployment; Section 8 discusses limitations
and future work; Section 9 concludes.

---

## 2. Related Work

**Traffic-sign recognition datasets.** The German Traffic Sign Recognition
Benchmark (GTSRB) [Stallkamp2012] popularized the task with 43 classes and
~50 k samples in controlled near-camera framing. The Belgian, Swedish, and
TT100K [Zhu2016] datasets each broaden geographic / visual diversity. MTSD
[Ertler2020] is the largest and most diverse, but its long-tail
distribution and unbounded `other-sign` class make naive benchmarking
misleading.

**Classical pipelines.** Histogram of Oriented Gradients (HOG)
[Dalal2005] combined with support vector machines was the dominant
pre-deep-learning formulation for traffic signs and many other
fine-grained visual categories.

**Convolutional neural networks for signs.** Stallkamp et al. showed that
a CNN trained from scratch on GTSRB could surpass human performance.
EfficientNet [Tan2019], a family of scalable image-recognition networks
pretrained on ImageNet, has since become a standard transfer-learning
starting point because of its accuracy-per-parameter Pareto front.

**Robustness under distortion.** Hendrycks and Dietterich's ImageNet-C
[Hendrycks2019] formalized the practice of testing models under synthetic
image corruptions; we adopt the same framework on the MTSD test set.

---

## 3. Dataset & Curation

### 3.1 MTSD overview

MTSD v2 provides 36,589 fully annotated training images, 5,320 validation
images, and 10,543 test images (test labels withheld). Each image carries
COCO-style bounding boxes for every visible sign together with a
fine-grained label (e.g. `regulatory--no-entry--g1`) and per-instance
properties (occluded, out-of-frame, ambiguous, …).

### 3.2 Curation policy

We define our classification task on **cropped signs** rather than full
images, by extracting each bounding box and resizing to 96 × 96. To make
the task well-posed, we apply three policy decisions:

1. **Discard "other-sign"** — 121,136 of 190,496 raw crops (**63.6 %**)
   carry MTSD's `other-sign` label, which is by definition an
   uncategorized catch-all. Keeping it would bias the model toward
   defaulting to this class whenever uncertain and would dominate
   macro-averaged metrics.
2. **Discard classes with < 30 train samples** — 74 such tail classes
   (1,339 crops total) are too rare to learn from at this sample budget.
3. **Discard signs with bounding boxes smaller than 12 px on either side**
   — such crops carry virtually no discriminative content after the
   96×96 resize.

After curation we obtain **326 classes and 67,833 crops**. Class sample
counts in the train split range from 27 to 2,150 with median 85; a clear
long-tail residual remains and is handled at the loss level via class
weighting (Section 4.2).

### 3.3 Splits

To use the dataset's released partitioning faithfully, we:

- Reserve **all crops from MTSD-validation** as our held-out **test set**
  (8,604 crops, never seen during hyperparameter selection).
- Stratified 90 / 10 split of **MTSD-train** into our **train** (53,306)
  and our **validation** (5,923) sets.

| Split | Crops | Source | Role |
|---|---:|---|---|
| Train | 53,306 | 90 % of MTSD-train | Model fitting |
| Val   | 5,923 | 10 % of MTSD-train | Hyperparameter selection, early stopping |
| Test  | 8,604 | All of MTSD-val | Held-out reporting |

---

## 4. Methodology

### 4.1 Phase 1 — Classical baseline (HOG → SGD-SVM)

Each cropped image is resized to 64 × 64, converted to grayscale, smoothed
with a 5 × 5 Gaussian, and passed through Canny edge detection. We then
compute a Histogram of Oriented Gradients (HOG) with 9 orientations, 8 × 8
pixel cells, 2 × 2 blocks, and L2-Hys block normalisation, yielding a
**1,764-dimensional** descriptor. Features are zero-mean unit-variance
standardised and classified with a multi-class Linear SVM fitted via
stochastic gradient descent (`sklearn.linear_model.SGDClassifier`,
`loss="hinge"`, `penalty="l2"`, `alpha=1e-4`, `class_weight="balanced"`).

The SGD variant was chosen over `sklearn.svm.LinearSVC` (primal solver)
because the latter did not converge in a tractable time-budget — over
60 minutes on this dataset's 326-class, 53 k-sample, balanced-weight
regime, vs **164 seconds** for the SGD fit. Both are linear SVMs with L2
regularization and one-vs-rest decision rules.

This pipeline is structurally identical to the Phase-1 notebook provided
by the course (`visionPhase1.ipynb`), differing only in (a) the matched
class set and splits and (b) the faster solver. Reporting against the
exact same test partition as the CNN models makes the comparison
apples-to-apples.

### 4.2 Phase 2 — Custom CNN baseline (BaselineCNN)

We design a from-scratch network specifically sized to 96 × 96 inputs:

| Stage  | Layer                                       | Output             |
|--------|---------------------------------------------|--------------------|
| Stem   | Conv 7×7 /2 → BN → ReLU → MaxPool 3×3 /2     | 64 × 24 × 24       |
| Block 1| Conv 3×3 → BN → ReLU ×2 → MaxPool 2×2        | 128 × 12 × 12      |
| Block 2| Conv 3×3 → BN → ReLU ×2 → MaxPool 2×2        | 256 × 6 × 6        |
| Block 3| Conv 3×3 → BN → ReLU ×2                      | 256 × 6 × 6        |
| Head   | GAP → Dropout(0.3) → Linear                  | 326                |

The model has **2.38 M parameters** and uses Kaiming-normal initialisation.
Design choices and their justifications:

- **7×7 strided stem + 3×3 pool**: 96×96 inputs are small; aggressive
  early downsampling reduces compute without losing signal.
- **Stacked 3×3 convolutions**: same receptive field as 5×5 with fewer
  parameters and more non-linearity (VGG insight, not VGG itself).
- **BatchNorm everywhere**: stabilises a random-init network against the
  residual class imbalance.
- **Global Average Pool + small Dropout + 1-layer head**: ~10× fewer
  parameters than Flatten+Dense and far less prone to over-fitting.

Training uses AdamW (lr 3 × 10⁻³, weight decay 1 × 10⁻⁴), OneCycle cosine
learning-rate schedule with 10 % warmup, batch 128, **50 epochs**, label
smoothing 0.05, and class-weighted cross-entropy with weights set to
`sqrt(1 / freq)` normalised to mean 1.0. The best checkpoint by
validation macro-F1 is retained.

### 4.3 Phase 2 — Experimental study

**E1: Data augmentation.** Re-train BaselineCNN with Albumentations
augmentation: rotation ±15 ° (no horizontal flip — signs are direction-
sensitive), brightness / contrast ±25 %, hue / saturation jitter, mild
Gaussian blur, coarse cut-out occlusion, and 8 % translation / 92–108 %
scale jitter.

**E2: Transfer learning.** Replace BaselineCNN with **EfficientNet-B0**
[Tan2019] initialised from ImageNet weights via `timm`. **4.43 M
parameters** (1.9× BaselineCNN). Same training recipe except `lr=1e-3`
(typical for fine-tuning); data augmentation enabled.

**E3: Robustness under synthetic distortions** (Section 6). Six
distortions × four severities applied to the test set; the best model
scored.

---

## 5. Results

All metrics are computed on the held-out test set (8,604 crops, 326
classes). Inputs are identical across rows — same crop pipeline, same
split, same class set — so differences reflect the model only.

### 5.1 Headline comparison

| Model                                    | Top-1 acc | Top-5 acc | Macro-P | Macro-R | Macro-F1 | Weighted-F1 |
|------------------------------------------|----------:|----------:|--------:|--------:|---------:|------------:|
| Phase 1 (HOG → SGD-SVM, matched rerun)   | 55.97 %   | —         | 0.555   | 0.553   | 0.524    | 0.591       |
| Phase 2 — BaselineCNN (no aug)           | 96.55 %   | 99.43 %   | 0.956   | 0.949   | 0.949    | 0.965       |
| Phase 2 — BaselineCNN + augmentation     | 96.89 %   | 99.51 %   | 0.963   | 0.958   | **0.958**   | 0.969       |
| Phase 2 — EfficientNet-B0 (transfer)     | **96.98 %**  | 99.48 %   | 0.959   | 0.957   | 0.956    | 0.970       |

**Key observations.**

- **Classical → CNN is a 40-point chasm.** From 55.97 % top-1 to 96.55 %
  with the same crops and the same splits — a +40.6 pt absolute gap
  (a 73 % relative error reduction). The classical HOG + SVM features
  cannot capture the discriminative information needed for 326-way
  fine-grained classification at this image resolution.
- **Augmentation gives the highest macro-F1 (0.958)** — the metric that
  treats every one of the 326 classes equally, including tail classes
  where data is scarce. Augmentation helps where it matters most.
- **Transfer learning gives the highest top-1 accuracy (96.98 %)** —
  EfficientNet-B0's ImageNet prior is most helpful on the head classes
  that dominate the weighted average.
- Top-5 accuracy is essentially saturated (99.4-99.5 %) for all three
  CNN models, indicating the correct class is almost always within the
  shortlist and remaining errors are confusions between visually similar
  sign variants.

### 5.2 Training dynamics

All three CNNs are trained for 50 epochs with the OneCycle LR schedule.

| Model | Best epoch (by val-F1) | Best val acc | Best val F1 | Final train acc | Final val acc | Train-val gap | Avg epoch time |
|---|---:|---:|---:|---:|---:|---:|---:|
| BaselineCNN (no aug)      | 39 | 0.9667 | 0.9550 | 1.0000 | 0.9657 | **3.43 pt** | 121 s |
| + Augmentation            | 39 | 0.9671 | 0.9562 | 0.9936 | 0.9664 | **2.72 pt** | (note 1) |
| EfficientNet-B0 (transfer)| 47 | 0.9691 | 0.9581 | 0.9965 | 0.9681 | 2.84 pt | 252 s |

*(note 1)* The augmentation run encountered episodes of Mac GPU contention
that inflated the wall-clock epoch time average; individual epochs were
~130 s on a quiescent system.

**Augmentation cleanly closes the generalisation gap** (3.43 pt → 2.72 pt
on train-vs-val accuracy). This is the textbook expected behaviour of
augmentation and exactly the experimental signal we hoped to see.

### 5.3 Per-class analysis

Errors concentrate in two failure modes: (a) extremely similar visual
pairs (`warning--curve-left--g1` vs `warning--curve-left--g2`), and (b)
tail classes near the 30-sample threshold. Full per-class precision /
recall / F1 / support breakdowns are written to
`results/<run>/test_per_class.csv` (326 rows × 4 columns each).

---

## 6. Robustness Analysis

We evaluate the augmentation-trained model (best macro-F1) on the test
set under six distortions × four severities (0 = clean → 3 = severe).
Distortions are implemented in `src/distortions.py` and chosen to model
realistic deployment conditions: illumination shifts (day/night,
headlights), small camera tilt, foliage / vehicle occlusion, low-light
sensor noise, motion-induced blur, and lossy compression in transmitted
or stored imagery.

| Distortion       | Sev 0   | Sev 1   | Sev 2   | Sev 3   | Drop @ Sev 3 |
|------------------|--------:|--------:|--------:|--------:|-------------:|
| Gaussian noise   | 96.89 % | 96.84 % | 96.87 % | 96.83 % | **0.06 pt**     |
| JPEG compression | 96.89 % | 96.89 % | 96.90 % | 96.37 % | **0.51 pt**     |
| Motion blur      | 96.89 % | 96.87 % | 96.90 % | 96.44 % | **0.44 pt**     |
| Rotation         | 96.89 % | 96.87 % | 96.47 % | 95.36 % | **1.52 pt**     |
| Illumination     | 96.89 % | 96.75 % | 94.64 % | 89.14 % | **7.74 pt**     |
| Occlusion        | 96.89 % | 96.48 % | 95.27 % | 86.85 % | **10.03 pt**    |

**Key findings.**

- **Near-zero degradation under Gaussian noise, JPEG, and motion blur.**
  At the most severe levels tested, accuracy drops by ≤ 0.51 pt. This
  suggests the BatchNorm-regularised conv features are already invariant
  to mid-frequency texture perturbations.
- **Rotation is mostly handled.** Even at severity 3 (±25 °) accuracy
  drops only 1.5 pt — a direct dividend of training-time rotation
  augmentation (±15 °).
- **Illumination is the main brittleness.** Brightness shifts of ±60 %
  cause a 7.7 pt drop, suggesting the model has implicitly memorised
  scene luminance distribution. Stronger training-time brightness/contrast
  augmentation is a low-effort fix.
- **Occlusion is the worst case (-10 pt).** Large square cut-outs that
  obscure 30-40 % of the sign cause the model to fail catastrophically.
  This matches the intuition that classifying a fine-grained sign from
  partial visibility requires either context or part-based reasoning,
  neither of which our 96² crop-based formulation gives the model.

---

## 7. Deployment (Bonus Option A)

The augmentation-trained model is shipped four ways:

1. **Gradio app** (`src/deploy/app.py`) — interactive top-5 demo.
2. **FastAPI service** (`src/deploy/api.py`) — `POST /predict` for
   programmatic access. End-to-end JSON round-trip ≤ 20 ms / request on
   CPU; smoke-tested with a held-out test image returning the correct
   label at 93.7 % confidence.
3. **Docker image** (`src/deploy/Dockerfile`) — CPU-only FastAPI service
   in a portable container, ~ 600 MB compressed.
4. **ONNX** (`results/aug/export/model.onnx`, ~ 10 MB) for
   cross-platform inference and **Core ML**
   (`results/aug/export/model.mlpackage`, ~ 9 MB) for Apple Neural
   Engine inference on iOS / macOS.

---

## 8. Discussion and Limitations

- **Hard class set choice.** Dropping `other-sign` and rare classes
  produces a learnable problem but discards 67 % of raw annotations. A
  hierarchical formulation (super-category first, then fine-grained) is
  a natural extension.
- **No object detection.** We classify pre-cropped boxes given by MTSD's
  ground truth. In an end-to-end self-driving pipeline these crops would
  themselves be predictions from a detector, propagating errors.
- **Phase-1 solver substitution.** Replacing primal LinearSVC with
  SGD-fitted hinge loss saves training time (164 s vs > 1 h) at a small
  cost in convergence quality. Given the size of the CNN-vs-classical
  gap (40 + points), this choice does not affect the qualitative
  conclusions of the paper.
- **MPS non-determinism.** The Apple MPS backend has minor
  non-determinism in some kernels; epoch-to-epoch variance is ≤ 0.5 %.

### Future work

- Hierarchical classifier (super-category → fine-grained).
- Curriculum learning for tail classes.
- Stronger brightness/contrast and large-CoarseDropout augmentation to
  directly attack the two failure modes identified in Section 6.
- Train on the union of all detection-style crops and end-to-end fine-
  tune with a YOLO-class detector for full pipeline evaluation.

---

## 9. Conclusion

We presented a clean, end-to-end MTSD classification study with apples-
to-apples comparison of classical and deep-learning approaches on
identical splits and class sets. The custom 2.38 M-parameter BaselineCNN
substantially outperforms the matched HOG → SGD-SVM baseline (96.55 % vs
55.97 % top-1 — a 40.6 pt absolute gap). Data augmentation gives the best
macro-F1 (0.958) and ImageNet-pretrained EfficientNet-B0 gives the best
top-1 accuracy (96.98 %). The augmentation-trained model degrades
gracefully under realistic synthetic distortions — losing < 1.5 pt under
noise, blur, JPEG, and small rotation — but is brittle to large
occlusions and illumination changes, suggesting concrete next steps.
Code, trained weights, an interactive Gradio demo, a Dockerized FastAPI
service, and ONNX / Core ML exports are released.

---

## References

- [Ertler2020] Ertler et al., "The Mapillary Traffic Sign Dataset for
  Detection and Classification on a Global Scale", ECCV 2020.
- [Dalal2005] Dalal & Triggs, "Histograms of Oriented Gradients for
  Human Detection", CVPR 2005.
- [Stallkamp2012] Stallkamp et al., "Man vs. computer: Benchmarking
  machine learning algorithms for traffic sign recognition", Neural
  Networks 2012.
- [Tan2019] Tan & Le, "EfficientNet: Rethinking Model Scaling for
  Convolutional Neural Networks", ICML 2019.
- [Hendrycks2019] Hendrycks & Dietterich, "Benchmarking Neural Network
  Robustness to Common Corruptions and Perturbations", ICLR 2019.
- [Zhu2016] Zhu et al., "Traffic-Sign Detection and Classification in
  the Wild", CVPR 2016.
- [MTSD] https://www.mapillary.com/dataset/trafficsign
