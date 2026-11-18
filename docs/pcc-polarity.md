# Signed PCC and input polarity

This diagnostic compares the grayscale intensity of a generated H&E image with the corresponding input PI image. It is distinct from the correlation between generated H&E and paired reference H&E. Pearson's (r) measures the direction and strength of the linear association between the two sets of pixel intensities. Signs are retained from the archived evaluation.

PCC does not have a universal higher-is-better interpretation across organs. In the held-out liver split, both model outputs have a negative correlation with PI, consistent with the direction observed between paired reference H&E and PI. The reference H&E–PI mean PCC was −0.81 for liver and +0.85 for kidney in the archived 150-image-per-organ diagnostic. These reference means and the full-test results below cover different sample sets.

## Held-out breast (5,636 pairs)

### Three source organs

| Model | PCC |
| --- | ---: |
| Pix2Pix | +0.818 |
| CycleGAN | −0.654 |
| Conditional DDIM (50 steps) | +0.739 |
| I2SB (20 steps) | +0.833 |
| Base | +0.835 |
| Base + StainAug | +0.831 |
| Base + DANN | +0.774 |
| Base + MixStyle | +0.822 |
| Base + FOMLDG | +0.826 |
| Base + SWAD (validation loss valley) | +0.806 |
| UNIStainNet (local PI-to-H&E adaptation) | +0.829 |
| UNI-StainDG | +0.834 |
| DualPrior (6 steps) | +0.861 |
| DualPrior (2 steps) | +0.865 |

### Kidney-only donor reference

| Model | PCC |
| --- | ---: |
| CycleGAN | +0.742 |
| Pix2Pix | +0.817 |
| Base | +0.867 |

The joint-source and kidney-only rows are separate training protocols. The joint-source CycleGAN comparison figure in the paper includes generated images and their paired H&E references. Its visible polarity mismatch is assessed against those references.

## Leave-one-organ-out comparison

Each row uses weights trained on the other three organs. These values describe the archived full-test runs.

| Held-out organ (number of pairs) | Base | UNI-StainDG |
| --- | ---: | ---: |
| Kidney (800) | +0.911 | +0.876 |
| Brain (1,000) | +0.825 | +0.790 |
| Liver (750) | −0.852 | −0.856 |
| Breast (5,636) | +0.835 | +0.834 |
