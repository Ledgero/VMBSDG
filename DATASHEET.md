# Datasheet for the SynLoc Dataset

Following the *Datasheets for Datasets* template by Gebru et al. (2021).
This datasheet accompanies the SynLoc release archived at Zenodo
(DOI: [10.5281/zenodo.20589296](https://doi.org/10.5281/zenodo.20589296))
and the paper *"SynLoc: Viewpoint-Diverse Synthetic Urban Data with Covisibility Filtering for Few-Shot Visual Localization"* (Ou, Xu, Jiang, Song; The Visual
Computer, 2026).

---

## 1. Motivation

**For what purpose was the dataset created?**
SynLoc was created to provide a large-scale, geometrically accurate
synthetic corpus for training and evaluating feature-matching and
visual-localization models, alleviating the cost and label noise of
real-world capture and enabling few-shot real-data fine-tuning.

**Who created the dataset and on behalf of which entity?**
The dataset was created by Zhiquan Ou, Cheng Xu, Wei Jiang and Mofei
Song at Southeast University and China Unicom Digital Technology
Co., Ltd.


---

## 2. Composition

**What do the instances represent?**
Each instance is a synthetic image pair `(I_a, I_b)` rendered from a
procedurally generated city, accompanied by:
- RGB images,
- per-pixel depth maps,
- camera intrinsics `K`,
- camera extrinsics `T_wc` (world-to-camera, right-handed),
- a forward / backward overlap pair `(O_{a→b}, O_{b→a})`,
- the binary covisibility-filter decision.

**How many instances are there in total?**
Across the full release, there are 274 rendered scenes and 53,719 retained
image pairs after covisibility filtering.

| Subset           | #Scenes | #Pairs (after filter) |
|------------------|--------:|----------------------:|
| town-level       |       8 |                 2,437 |
| district-level   |      64 |                23,253 |
| city-level       |     202 |                28,029 |
| **Total**        |     274 |                53,719 |

**Does the dataset contain all possible instances or a sample?**
A sample drawn from the ViewDiversify-Pose proposal distribution and
filtered by the bidirectional Covisibility-Filter (acceptance region
detailed in §3.3 of the paper).

**What data does each instance consist of?**
Raw image (PNG, 8-bit sRGB), 32-bit-float depth (EXR), pose / intrinsic
metadata (JSON), pair list (CSV).

**Is there a label or target associated with each instance?**
Camera pose, depth and covisibility serve as ground-truth labels.

**Is any information missing from individual instances?**
No. All instances carry complete pose, depth, intrinsics and overlap.

**Are relationships between individual instances made explicit?**
Yes -- the pair list (`pairs.csv`) explicitly enumerates every retained
image pair and its covisibility statistics.

**Are there recommended data splits?**
Yes. Standard train/val/test splits at scene level are provided in
`splits/{train,val,test}.txt`. Splits are scene-disjoint to prevent
leakage.

**Are there any errors, sources of noise or redundancies?**
The dataset is procedurally generated and free of annotation noise.
Limitations include: lack of fine-grained micro-geometry, no dynamic
objects (pedestrians/vehicles), no real sensor noise / motion blur.

**Is the dataset self-contained?**
Yes -- once downloaded from Zenodo, no external resource is required to
*use* the dataset. Regenerating new scenes requires Unreal Engine 4.27
and AirSim (see `ASSETS.md`).

**Does the dataset contain confidential or sensitive information?**
No. Procedurally generated content; no personal data, no real persons,
no offensive content.

---

## 3. Collection Process

**How was the data acquired?**
Images and depth are rendered in Unreal Engine 4.27 driven by AirSim;
camera poses are proposed by the *ViewDiversify-Pose* planner described
in §3.2 and accepted/rejected by the bidirectional *Covisibility-Filter*
described in §3.3.

**What mechanisms or procedures were used?**
A Python pipeline orchestrating: (i) CFG-rule city generation from OSM
footprints, (ii) AirSim-driven rendering, (iii) overlap computation via
forward/backward depth re-projection, (iv) acceptance-region filtering,
(v) packaging into the released archive.

**Who was involved in data collection?**
The four paper authors. No crowdworkers or third parties.


**Were any ethical-review processes conducted?**
Not applicable -- procedurally generated synthetic content; no human
subjects, no animal data.

---

## 4. Preprocessing / Cleaning / Labelling

Pose, depth and covisibility are computed analytically, not annotated
post-hoc; no manual labelling is involved. Pairs failing the
acceptance region (Eqs. 9-10 in the paper) are removed prior to
release. Removal statistics are reported in §4 (R2-J / R2-P).

Raw (pre-filter) pose proposals and depth re-projections are also
released so users may re-derive the splits with alternative thresholds.

---

## 5. Uses

**For what tasks has the dataset been used?**
- Training / evaluating feature-matching networks (LoFTR, RoMa, ...);
- Synthetic pre-training followed by few-shot real-data fine-tuning,
  evaluated on MegaDepth-1500;
- Ablation of pose-sampling strategies (Wander vs. Zoom) and
  covisibility filtering.

**Is there a repository linking papers and systems that use the dataset?**
Yes -- see the *Citation* section of the GitHub README and the
project page.

**What other tasks could the dataset be used for?**
Multi-view stereo, novel-view synthesis, neural rendering, visual SLAM
benchmarking and AR/VR pose-registration evaluation (see *Broader
Relevance to Visual Computing* in §5.2).

**Are there tasks for which the dataset should not be used?**
Tasks requiring real sensor noise, dynamic objects (pedestrians,
vehicles) or fine-grained material realism should *not* rely on SynLoc
alone -- a few-shot real-data fine-tuning stage is recommended.

---

## 6. Distribution

**How will the dataset be distributed?**
- Versioned archive at Zenodo: <https://doi.org/10.5281/zenodo.20589296>
- Generation / evaluation code on GitHub:
  <https://github.com/Ledgero/VMBSDG>

**When will the dataset be distributed?**
Already distributed (v1.0.0).

**Will the dataset be distributed under a copyright or other IP licence?**
- Dataset: CC-BY-4.0 (see `LICENSE-DATA`).
- Code:    MIT       (see `LICENSE`).
- Third-party assets retain their original licences (see `ASSETS.md`).

**Have any third parties imposed IP-based or other restrictions?**
Yes -- Unreal Engine, AirSim, Quixel Megascans and OSM impose their
own terms; we therefore do **not** redistribute engine binaries,
proprietary 3D assets or OSM dumps. Only our pipeline outputs and code
are redistributed.

**Do any export controls or regulatory restrictions apply?**
None known.

---

## 7. Maintenance

**Who is supporting / hosting / maintaining the dataset?**
The corresponding author (Mofei Song, songmf@seu.edu.cn) .

**How can the manager be contacted?**
By GitHub issue (<https://github.com/Ledgero/VMBSDG/issues>) or email.

**Will the dataset be updated?**
Bug-fix releases will be archived at Zenodo with new version DOIs;
the *concept DOI* (10.5281/zenodo.20589296) always resolves to the
latest version.

**Will older versions continue to be supported?**
All released versions remain permanently archived on Zenodo.

**If others want to extend / augment / contribute, is there a mechanism?**
Yes -- pull requests on GitHub are welcome.
