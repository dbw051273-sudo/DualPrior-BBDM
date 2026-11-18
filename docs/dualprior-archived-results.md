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
