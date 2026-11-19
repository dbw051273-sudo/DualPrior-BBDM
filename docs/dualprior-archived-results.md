# Archived DualPrior comparisons

UNI-StainDG is the main method in the manuscript. DualPrior is the variant that adds CONCH reference retrieval from the source-domain library to UNI-StainDG. This page records the DualPrior results used in earlier versions of the main comparison figures.

## Held-out breast

The earlier qualitative comparison used six-step DualPrior outputs for three views: gland or duct, stroma, and dense nuclei. The updated figure uses six-step UNI-StainDG outputs for those same views, with matched PI, reference H&E, and the same 104 × 104 pixel regions of interest.

The following scores are from the archived evaluation of 5,636 held-out breast pairs at the 190,000-step checkpoint, without target H&E references.

| Model | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ | Dice ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| DualPrior, six steps | 18.31 | 0.783 | 0.210 | 38.23 | 0.546 |
| UNI-StainDG, six steps | 17.62 | 0.772 | 0.213 | 36.71 | 0.552 |

## Reference conditions in the earlier PSNR figure

The earlier figure displayed DualPrior as the three-source model. The revised figure displays UNI-StainDG. P2 and P3 used target H&E information for post hoc sensitivity analysis.

| Model | P0: no target refs | P2: five unpaired target refs, seed 7 | P3: 150 target H&E images |
| --- | ---: | ---: | ---: |
| DualPrior | 18.31 | 22.37 | 22.40 |
| UNI-StainDG | 17.62 | 22.76 | 23.11 |

Values are PSNR in dB. Across the three single-source donors (liver, kidney, brain) and DualPrior, the observed max–min spans were 6.76, 1.72, and 0.72 dB for P0, P2, and P3, respectively. With UNI-StainDG in the four-model comparison, the spans are 6.76, 1.72, and 1.10 dB. Each span summarizes the observed model scores under its own reference condition.

The historical 2-, 6-, 20-, and 200-step sweep used DualPrior. The six-step UNI-StainDG result is a single measured operating point.

## Source-only CONCH retrieval and decoder injection

The optional DualPrior branch uses a frozen CONCH image encoder. For each of 9,000 paired source fields, a 4 × 4 nonoverlapping partition gives 144,000 paired crops. A bank entry stores the L2-normalized CONCH embedding of the source PI crop as its key and the normalized embedding of its paired source H&E crop as its value.

At training and sampling time, the input PI supplies the query. FAISS exact inner-product search retrieves the eight highest-scoring source keys. At the 32 × 32 and 64 × 64 decoder stages, Swin features supply attention queries, and the eight retrieved H&E embeddings supply projected keys and values. For a decoder query q, attention uses softmax(qKᵀ / √d)V. This bank uses source-domain pairs; target H&E reference calibration is a separate post hoc procedure. UNI-StainDG uses the frozen UNI features of the current PI and has no CONCH retrieval branch.

### Fixed-weight retrieval substitution

| Evaluation set | Top-8 PSNR (dB) | Random retrieval PSNR (dB) | Retrieval branch off PSNR (dB) |
| --- | ---: | ---: | ---: |
| Breast subset, 1,000 pairs | 17.9176 | 17.9139 | 17.9253 |
| Within-domain kidney, 800 pairs | 25.580 | 25.581 | — |

These substitutions use the same trained DualPrior weights. Disabling the retrieval branch at inference bypasses its cross-attention operation. UNI-StainDG in the main comparison was trained separately without that branch.
