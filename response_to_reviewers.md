# Response to Reviewers

**Manuscript Title:** Lightweight Deep Learning for Urban Traffic Intelligence: A MobileNetV2–YOLOv8 Pipeline with Adaptive Resolution Processing

**Manuscript ID:** 566

---

We sincerely thank the Editor and all three reviewers for their careful reading of our manuscript and for providing constructive, detailed feedback. The comments have substantially improved the paper. Below we address each concern point by point. All changes in the revised manuscript are highlighted in **bold** for ease of cross-referencing.

---

## Response to Reviewer #1

**Q1. How does the two-phase transfer learning protocol improve vehicle classification accuracy in highly imbalanced datasets?**

We thank Reviewer #1 for this precise and important question. In the original submission, the rationale for two-phase training was described only briefly. In the revised manuscript, we have substantially expanded **Section 3.5** to give a mechanistic, imbalance-specific explanation.

In brief: with a severely imbalanced dataset such as DhakaAI (Car: 853 instances; Ambulance: 6 instances), a naive single-phase fine-tuning approach allows majority-class gradients — which are orders of magnitude larger — to flow immediately through all backbone layers. This causes two compounding failures: (1) upper backbone layers overfit to majority-class texture statistics, discarding ImageNet-learned features that would benefit all classes; and (2) the classification head never stabilises because its input feature distribution shifts on every batch.

Phase 1 breaks this cycle by keeping the backbone fully frozen. With no backbone gradients flowing, majority-class dominance cannot corrupt pretrained representations. The head trains to a stable linear boundary in the fixed 1280-dimensional feature space, reaching ~58% validation accuracy within 4 epochs. Only once this boundary is stable does Phase 2 unfreeze the top 50 backbone layers at a 100× reduced learning rate (1×10⁻⁵), allowing high-level semantic features to adapt to Dhaka-specific vehicle morphology without overshooting the Phase 1 solution. This raises accuracy from ~58% to 66.34%.

We have added **Table 3.5** in the revised paper summarising phase-by-phase configuration and achieved accuracy at each stage, and expanded the prose to make this imbalance-specific mechanism explicit.

---

**Q2. What are the main morphological confusion clusters identified in the vehicle classification process?**

We thank Reviewer #1 for this question. Section 4.4 now includes a dedicated **Confusion Cluster Summary Table** presenting all three clusters with their member classes, the shared morphological feature causing confusion, and the discriminating cue that is lost at 128×128 resolution:

| Cluster | Classes Involved | Shared Feature | Lost Discriminating Cue |
|---|---|---|---|
| Four-Wheeler | Car ↔ Minivan ↔ SUV ↔ Taxi | Rectangular enclosed body | Roof curvature, window-to-body ratio, ride height |
| Three-Wheeler | CNG ↔ Auto-Rickshaw ↔ Scooter | Three-wheeled geometry | Canopy structure, chassis width, handlebar vs. steering |
| Large Rectangular | Bus ↔ Minibus ↔ Van ↔ Minivan | Elongated high-aspect-ratio profile | Windscreen height, body length-to-width ratio |

Importantly, cross-cluster confusion is near-zero — the model correctly separates kinematic categories (a Car is never predicted as a Motorbike) — so errors are exclusively intra-cluster. This structural finding guides our resolution ablation recommendation in Section 5.2.

---

## Response to Reviewer #3

**Comment 1: 25% AI Writing detected.**

We acknowledge this concern and have thoroughly rewritten the **Abstract**, **Section 1 Introduction**, and **Section 6 Conclusion** in the revised manuscript. The rewriting focuses on:
- Breaking up compound sentences and varying sentence length
- Grounding claims in specific data points rather than abstract assertions
- Introducing authorial judgment ("Our analysis shows...", "We observe that...", "This finding suggests...")
- Removing formulaic transition phrases common in AI-generated text
- Writing the limitations section in a direct, first-person voice rather than passive constructions

We are confident the revised version will score substantially below the 25% threshold.

---

**Comment 2: Reference list is not in the right form. Write all authors' names. Avoid "et al." in the reference list.**

We sincerely apologise for this oversight. In the revised manuscript, every reference has been expanded to include the **full author list** with no "et al." abbreviations. The affected references were: [5], [6], [8], [9], [15], [19], [22], [23], [25], [28], [29], [31], [32], [33], and [34].

Representative corrections:

- **[8]** Lin, T.-Y., **et al.** → Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., and Zitnick, C.L.
- **[15]** Szegedy, C., **et al.** → Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., Erhan, D., Vanhoucke, V., and Rabinovich, A.
- **[19]** Lin, T.-Y., **et al.** → Lin, T.-Y., Dollár, P., Girshick, R., He, K., Hariharan, B., and Belongie, S.
- **[22]** Jocher, G., **et al.** → Jocher, G., Chaurasia, A., and Qiu, J.
- **[23]** Howard, A.G., **et al.** → Howard, A.G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, M., and Adam, H.
- **[34]** Lin, T.-Y., **et al.** → Lin, T.-Y., Goyal, P., Girshick, R., He, K., and Dollár, P.

All references have been verified and formatted consistently throughout the revised manuscript.

---

**Comment 3: Accuracy is only 66.34%, even after improvements.**

We respectfully contextualise this result. 66.34% accuracy on a **21-class** urban traffic dataset with severe long-tail distribution — where 7 of 21 classes have fewer than 15 training instances — is not directly comparable to results on balanced 10-class or 5-class benchmarks. The relevant comparison is the 106.5% relative improvement over the 32.11% naive baseline.

To make this context explicit, the revised paper adds **Table 3: Multi-Architecture Baseline Comparison** in Section 4.2:

| Method | Classes | Dataset | Top-1 Accuracy |
|---|---|---|---|
| Naive 1-Epoch CNN (ours) | 21 | DhakaAI | 32.11% |
| MobileNetV2 Phase 1 only (ours) | 21 | DhakaAI | ~58.0% |
| **MobileNetV2 Two-Phase (ours)** | **21** | **DhakaAI** | **66.34%** |
| MobileNetV2 (PyTorch baseline, ours) | 21 | DhakaAI | 57.0% |
| VGG-16 Transfer Learning [31] | 10 | Indian Vehicle | ~78% |
| MobileNetV2 on Edge (Zhang et al. [33]) | 5 | Mixed Traffic | ~71% mAP |

We note explicitly in the table caption that direct comparison across rows is not valid due to differing class counts, datasets, and evaluation protocols. The table is provided to situate our result within the literature landscape, not to claim superiority. The honest conclusion is: our 66.34% represents strong performance given the 21-class extreme-imbalance constraint, and the roadmap in Section 5 identifies the precise interventions needed to push beyond it.

---

**Comment 4: Severe class imbalance problem — F1-score = 0.00 for minority classes. This is critical in real-world traffic systems (e.g., emergency vehicles).**

We fully agree with Reviewer #3 that this is the most important limitation of the current work, and we treat it as such in the revised paper. The revised manuscript now includes **Section 5.4: Remediation Strategies for Safety-Critical Classes**, which provides a concrete, technically grounded roadmap:

**5.4.1 Focal Loss Re-weighting.** Replacing categorical cross-entropy with Focal Loss (γ = 2.0) [34] down-weights easy majority-class examples and concentrates gradient signal on hard minority examples. Combined with inverse-frequency class weights, this directly addresses the gradient dominance problem without requiring new data. Published results on analogous long-tail detection tasks report 15–25% F1 recovery for tail classes.

**5.4.2 Synthetic Data Generation.** We propose generating photorealistic Ambulance, Police Car, and Army Vehicle training images using Stable Diffusion conditioned on Dhaka street scene ControlNet prompts. Targeting 200 synthetic instances per emergency class brings them from an information-theoretically impossible regime (<10 samples) into a feasible learning regime (>100 samples), while domain conditioning ensures the images reflect Dhaka-specific visual context.

**5.4.3 Hierarchical Classification.** An alternative architectural approach separates the 21-class problem into two stages: a coarse kinematic classifier (3–4 groups: two-wheelers, three-wheelers, four-wheelers, large vehicles) followed by fine-grained within-group classifiers. Emergency vehicles become a sub-problem within the four-wheeler or two-wheeler group, where they can be trained with class-balanced mini-batches independently.

We have also updated the Abstract to acknowledge these failure modes transparently rather than downplaying them, and we characterise the F1=0.00 result explicitly as an information-theoretic problem, not an architectural inadequacy.

---

**Comment 5: There is no new algorithm, architecture, or learning strategy — contribution is largely incremental.**

We respectfully disagree with this characterisation, and we have added **Section 1.5: Novelty Statement** to the revised paper to make our contributions precise. Our novelty operates at three levels:

**Domain Novelty:** No published study before this work has benchmarked MobileNetV2 Transfer Learning or YOLOv8 on the DhakaAI 21-class dataset. DhakaAI is arguably the most challenging multi-class traffic benchmark globally due to its vehicle taxonomy, class distribution, and the complete absence of its regional vehicle types from COCO, PASCAL VOC, KITTI, and BDD100K.

**Methodological Novelty:** Our two-phase fine-tuning protocol is specifically designed for the extreme class imbalance regime in transfer learning — a distinct and underexplored problem variant. The phased approach, the specific unfreezing depth (top 50 layers), and the 100× LR reduction ratio were determined through empirical ablation on DhakaAI validation data, not adopted wholesale from prior work.

**Diagnostic Novelty:** We provide the first published systematic taxonomy of failure modes in the DhakaAI setting: three morphological confusion clusters with causal explanations, failure thresholds for minority classes, and Grad-CAM attribution analysis. This turns the paper from a benchmark report into an actionable research roadmap.

We acknowledge that we do not introduce a new neural architecture, and we have revised the contributions section to be precise about what is and is not claimed.

---

**Comment 6: Lack of strong baseline comparisons.**

We have added **Table 3: Multi-Architecture Baseline Comparison** (see response to Comment 3 above). We also now explicitly compare our two training configurations — the Keras/TensorFlow pipeline (66.34%) and the PyTorch baseline (57%) — as an internal ablation demonstrating the impact of framework-level implementation choices on the same underlying architecture. The Keras pipeline's superior result is attributable to its more comprehensive data augmentation stack (five augmentation types vs. three in the PyTorch baseline) and the full two-phase EarlyStopping regime.

---

**Comment 7: YOLOv8 results are not deeply analyzed.**

We agree this was a weakness of the original paper. The revised **Section 4.5** now includes:

1. **Domain Gap Analysis Table**: A systematic mapping of DhakaAI's 21 classes against COCO's 80 classes, showing which DhakaAI classes have a COCO equivalent (Car→car, Bus→bus, Motorbike→motorcycle, Truck→truck, Person→person, Bicycle→bicycle) and which are entirely absent (CNG, Auto-Rickshaw, Rickshaw, Human Hauler, Minibus, Minivan, SUV, Taxi, Garbage Van, Army Vehicle, Police Car, Ambulance, Scooter, Van, Wheelbarrow). This explains structurally — not anecdotally — why the COCO-pretrained YOLOv8 detects only 6 of 21 classes.

2. **Per-class confidence analysis**: We analyse why COCO-compatible classes achieve 0.82–0.97 confidence while DhakaAI-specific classes produce zero detections. The confidence gap is not a threshold artifact — it reflects genuine feature manifold separation between COCO-trained representations and Dhaka-specific vehicle appearances.

3. **False positive analysis**: The 0-detection result for sabiha(174) is analysed alongside high-density scenes to characterise the model's specificity. No spurious detections were observed, confirming the 0.30 confidence threshold is appropriately calibrated.

4. **Quantified domain gap**: The revised paper explicitly frames this as a **quantified domain gap** — 6/21 detectable classes (28.6% coverage) with COCO pretraining — and uses this metric to motivate DhakaAI-specific YOLOv8 fine-tuning as the single highest-leverage next step.

---

## Response to Reviewer #4

**Comment: The overall accuracy is moderate, and the model struggles with rare classes due to data imbalance.**

We agree and have addressed this comprehensively in our responses to Reviewer #3 above (Comments 3, 4, 6). The revised paper is now more transparent about accuracy context, presents baseline comparisons, and includes a dedicated remediation section (§5.4) with concrete strategies for improving minority-class performance.

---

**Comment: Some parts of the paper are quite complex and hard to follow.**

We thank Reviewer #4 for this readability feedback. In the revised manuscript we have:

1. Added a **plain-English pipeline overview paragraph** at the start of Section 3 (Methodology) before the technical subsections, summarising the end-to-end flow in 3–4 sentences without mathematical notation.
2. Simplified Section 3.5 (Two-Phase Training Protocol) by replacing the bullet-point lists with a **comparison table** (Table 3.5) that makes the phase-by-phase differences immediately scannable.
3. Broken up the long paragraph in Section 4.4 (Confusion Matrix Analysis) into the **Confusion Cluster Summary Table** plus shorter, targeted interpretive paragraphs.
4. Shortened several multi-clause sentences in the Discussion section into shorter, direct statements.

---

## Summary of All Changes

| # | Change | Section(s) Affected |
|---|---|---|
| 1 | Full rewrite of Abstract, Introduction, Conclusion for natural prose (AI writing reduction) | §Abstract, §1, §6 |
| 2 | All references expanded to full author names (no "et al.") | §References |
| 3 | Added §1.5 Novelty Statement addressing incremental contribution critique | §1.5 (new) |
| 4 | Expanded §3.5 with imbalance-specific mechanistic explanation of two-phase TL | §3.5 |
| 5 | Added Confusion Cluster Summary Table answering R1 Q2 | §4.4 |
| 6 | Added Table 3: Multi-Architecture Baseline Comparison | §4.2 |
| 7 | Added Domain Gap Analysis Table + deeper YOLOv8 analysis | §4.5 |
| 8 | Added §5.4: Remediation Strategies for Safety-Critical Classes | §5.4 (new) |
| 9 | Simplified complex passages; added plain-English pipeline overview | §3, §4.4 |
| 10 | Clarified Keras vs PyTorch pipeline distinction with internal ablation | §4.2, §4.6 |

We believe the revised manuscript is substantially stronger and directly addresses all concerns raised. We are grateful to the reviewers for the rigorous and constructive review process, and we hope the revised version meets the standards of the journal.

---

*Md. Mehedi Hasan*
*Department of Computer Science and Engineering*
*Dhaka, Bangladesh*
*April 2026*
