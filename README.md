# Input Polarity and Stain Statistics: Controlled Attribution and a Dual-Prior Brownian Bridge Diffusion Framework for Cross-Organ Zero-Shot Virtual Staining
# 输入极性与染色统计：跨器官零样本虚拟染色的受控归因与双先验布朗桥扩散框架

Official PyTorch implementation, 190k-step controlled evaluation protocol, 4-level automated pathology evaluation suite, and dataset split manifests for **"输入极性与染色统计：跨器官零样本虚拟染色的受控归因与双先验布朗桥扩散框架"** (*Input Polarity and Stain Statistics: Controlled Attribution and a Dual-Prior Brownian Bridge Diffusion Framework for Cross-Organ Zero-Shot Virtual Staining*).

---

## 1. Overview / 项目简介

Mainstream label-free virtual staining models (e.g., Phase Imaging / Photoacoustic Imaging $\to$ H&E) are predominantly trained and evaluated within a single organ. Under a strict **leave-one-organ-out zero-shot** setting—where the model is trained exclusively on source organs (**Liver, Kidney, Brain**; 9,000 paired patches) and evaluated blindly on an unseen target organ (**Breast**; 5,636 paired patches)—conventional generative backbones suffer from severe nuclear topology collapse and chromatic drift.

This repository provides:
1. **Dual-Prior Brownian Bridge Diffusion Model (Dual-Prior BBDM)**:
   - **Backbone**: 303M-parameter Swin-UNet Brownian Bridge Diffusion Model (`BBDM`) modeling deterministic boundary-pinned trajectories between PI inputs and H&E targets.
   - **Structural Prior (`SpatialPriorInjector`)**: Dense spatial tokens from a frozen pathology foundation model (**UNI**) injected via zero-initialized $1\times1$ convolutions (ControlNet-style), preventing nuclear topology collapse on unseen organs ($-4.23\text{ dB}$ PSNR if ablated).
   - **Texture Prior (`TextureCrossAttention`)**: Self-queried $k$-NN prototype retrieval ($k=8$) over a **source-only CONCH memory bank** ($N=144,000$ patches across 3 source organs) using Gibbs-kernel cross-attention.
2. **Two-Factor Controlled Attribution (Polarity Alignment & Stain Statistics Calibration)**:
   - **Stage I (Zero-Shot Input Polarity Alignment)**: Aligning donor-recipient optical contrast polarity lifts the single-organ Liver donor on unseen Breast from `8.23 dB` to `20.04 dB` without any retraining or target H&E access.
   - **Stage II (Unpaired Site Reference Calibration)**: Lightweight Beer–Lambert optical density (OD) affine calibration using $n \ge 3$ unpaired archive H&E slides compresses all 4 donor models (`Liver`, `Kidney`, `Brain`, `Joint`) into a `22.01–23.39 dB` band ($\Delta < 1.4\text{ dB}$).
3. **4-Level Automated Pathology Evaluation Suite (`official_eval.py`)**:
   - **(1) Pixel & Photometric Fidelity**: PSNR, SSIM, LPIPS;
   - **(2) Distributional Realism**: Full-sample & patch-level FID, KID;
   - **(3) Nuclear Topology & Optical Density**: **Cellpose** instance segmentation nuclear mask **Dice** and Beer–Lambert Hematoxylin-channel **PCC**;
   - **(4) Foundation-Model Proxy & Dual-Observer Zero-Shot Consistency**: FD-UNI, CS-CONCH, and malignant-likelihood agreement ($r, \kappa, \text{AUC}$) across an independent observer (**PLIP**) and prior-sharing observer (**CONCH**).

---

## 2. Repository Structure / 仓库目录结构

```text
DualPrior-BBDM-GitHub/
├── configs/                        # Training & evaluation YAML configurations
│   ├── full_e3.yaml                # 3-Organ Joint Dual-Prior BBDM (Main Model, 190k steps)
│   ├── abl_nouni_laptop.yaml       # Ablation: w/o UNI structural prior (-UNI)
│   ├── abl_noconch_laptop.yaml     # Ablation: w/o CONCH retrieval prior (-CONCH)
│   ├── abl_dualoff_server.yaml     # Ablation: Pure BBDM backbone (w/o both priors)
│   ├── base_liver_server.yaml      # Single-organ Donor: Liver-only (190k steps)
│   ├── base_kidney_server.yaml     # Single-organ Donor: Kidney-only (190k steps)
│   ├── base_brain_server.yaml      # Single-organ Donor: Brain-only (190k steps)
│   ├── i2sb_joint_190k.yaml        # Baseline: I2SB (Schrödinger Bridge, 190k steps)
│   ├── gddpm_joint_240.yaml        # Baseline: Conditional DDIM
│   └── moe_joint_server.yaml       # Negative Evidence: Organ-MoE soft routing
├── data_splits/                    # Complete deterministic split filelists & metadata
│   ├── dataset_manifest.json       # Exact counts, optical polarity conventions & access policy
│   ├── liver_train_filelist.txt    # 3,000 train / 279 val / 750 test filenames
│   ├── kidney_train_filelist.txt   # 2,000 train / 194 val / 800 test filenames
│   ├── brain_train_filelist.txt    # 4,000 train / 863 val / 1,000 test filenames
│   └── breast_test_blind_filelist.txt # 5,636 blind test filenames
├── Datasets_demo/                  # 2 paired demo patches per organ (8 pairs total) for smoke testing
├── datasets/                       # Multi-organ paired dataset loaders & polarity/photometric transforms
├── model/                          # Swin-UNet BBDM, UNI SpatialPriorInjector & CONCH TextureCrossAttention
├── runners/                        # Training, checkpointing, and 6-step/2-step bridge sampling loops
├── main.py                         # Unified entry point for training and testing
├── precompute_uni.py               # Offline UNI dense token extraction (accelerates training)
├── build_conch_bank.py             # Offline source-only CONCH memory bank builder
├── gen_cross.py                    # Cross-organ zero-shot inference & polarity alignment script
├── eval_lib.py                     # Core evaluation & Beer-Lambert OD site calibration library
└── official_eval.py                # 4-Level automated evaluation script (PSNR/SSIM/LPIPS/FID/KID/Dice/PCC)
```

---

## 3. Dataset Organization & Data Availability / 数据集结构与获取政策

### Data Availability Statement (数据可用性说明)
- **Full Raw Dataset (全量原始数据)**: The complete multi-organ paired PI$\to$H&E dataset (**15,222 paired patches** across Liver, Kidney, Brain, and Breast) was acquired under institutional collaboration protocols at Hainan University. Due to institutional data privacy policies, the full raw image archive is not hosted directly in this public repository, but is available from the corresponding author upon reasonable academic request and advisor approval.
- **Reproducibility Assets Included Here (本仓库已公开的复现资产)**:
  1. **`data_splits/`**: Exact deterministic file lists (`*_filelist.txt`) and optical polarity metadata (`dataset_manifest.json`) for every split of all 4 organs.
  2. **`Datasets_demo/`**: 2 representative paired `input` (PI) and `target` (H&E) images per organ so users and reviewers can immediately verify the code pipeline out-of-the-box. *(Note: If strict zero-image hosting is required prior to publication, simply remove `Datasets_demo/` before pushing).*

| Organ (器官) | Role (角色) | Train (训练) | Val (验证) | Test (测试) | Native PI Polarity (原生 PI 极性约定) |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Liver (肝)** | Source Donor | 3,000 | 279 | 750 | Negative contrast (dark background, bright nuclei) |
| **Kidney (肾)** | Source Donor | 2,000 | 194 | 800 | Positive contrast (bright background, dark nuclei) |
| **Brain (脑)** | Source Donor | 4,000 | 200 (863 raw) | 1,000 | Positive contrast (bright background, dark nuclei) |
| **Breast (乳腺)** | **Held-Out Target** | — | — | **5,636** | Positive contrast (bright background, dark nuclei) |

When preparing the full local dataset (`Datasets_processed/`), organize it as follows:
```text
Datasets_processed/
├── liver/
│   ├── train/{input, target}/
│   ├── val/{input, target}/
│   └── test/{input, target}/
├── Kidney/
│   ├── train/{input, target}/
│   ├── val/{input, target}/
│   └── test/{input, target}/
├── Brain/
│   ├── train/{input, target}/
│   ├── val/{input, target}/
│   └── test/{input, target}/
└── Breast/
    ├── input/
    └── target/
```

---

## 4. Quick Start / 快速上手

### 4.1 Environment Setup (环境安装)
```bash
conda create -n vstain python=3.9 -y
conda activate vstain
pip install -r requirements.txt
```

### 4.2 Precompute Source-Only Priors (预提取源域 UNI 特征与 CONCH 检索库，仅训练加速用)
```bash
# 1. Extract frozen UNI dense tokens (16x16x1024) for source training organs
python precompute_uni.py --data_root /path/to/Datasets_processed --out_dir ./feat_cache_dir

# 2. Build source-only CONCH memory bank (144,000 vectors x 512d)
python build_conch_bank.py --data_root /path/to/Datasets_processed --out_dir ./feat_cache_dir
```

### 4.3 Training (190k-Step Controlled Protocol / 模型训练)
```bash
# Train 3-Organ Joint Dual-Prior BBDM (Main Model)
CUDA_VISIBLE_DEVICES=0 python main.py -c configs/full_e3.yaml --train --gpu_ids 0

# Train Single-Organ Donor Models (190k steps)
CUDA_VISIBLE_DEVICES=0 python main.py -c configs/base_liver_server.yaml --train --gpu_ids 0
CUDA_VISIBLE_DEVICES=0 python main.py -c configs/base_kidney_server.yaml --train --gpu_ids 0
CUDA_VISIBLE_DEVICES=0 python main.py -c configs/base_brain_server.yaml --train --gpu_ids 0
```

### 4.4 Cross-Organ Zero-Shot Inference & 4-Level Evaluation (跨器官零样本推理与四维评测)
```bash
# Run cross-organ inference + automatic 4-level evaluation (PSNR, SSIM, LPIPS, FID, KID, Cellpose Dice, H-channel PCC)
python gen_cross.py configs/base_brain_server.yaml ckpt_cross/brainonly190k.pth kidney 800 brainonly190k_k800 neg

# Or directly invoke the standalone 4-level evaluator on any generated folder vs. ground-truth folder
python official_eval.py \
    --gen_dir gens_screen/brainonly190k_k800 \
    --gt_dir eval_kidney/test/B \
    --out_json official_screen_brainonly190k_k800.json
```

---

## 5. Key Benchmark Results (24-Cell Donor × Recipient Matrix at 190k Steps)

PSNR (dB) across all 4 donors (`Liver-only`, `Kidney-only`, `Brain-only`, `3-Organ Joint`) and 4 recipients (`Liver`, `Kidney`, `Brain`, `Held-Out Breast`):

| Donor Model (190k steps) | $\to$ Liver (750) | $\to$ Kidney (800) | $\to$ Brain (1000) | $\to$ **Breast (5636, Held-Out)** |
| :--- | :---: | :---: | :---: | :---: |
| **Liver-only** (Native / Polarity-Aligned / Site-Calibrated) | *26.81* / — / — | 16.92 / — / — | 16.21 / — / — | 8.23 / **20.04** / 22.01 |
| **Kidney-only** (Native / Polarity-Aligned / Site-Calibrated) | 8.70 / 16.05 / — | *26.82* / — / — | 15.86 / — / — | 19.93 / 19.93 / 22.12 |
| **Brain-only** (Native / Polarity-Aligned / Site-Calibrated) | 9.57 / 13.98 / — | 14.80 / 14.80 / — | *25.58* / — / — | 13.28 / 13.28 / **23.39** |
| **3-Organ Joint (Ours)** (Native / Polarity-Aligned / Site-Calibrated) | *27.69* / — / — | *25.58* / — / — | *26.41* / — / — | 18.31 / 18.31 / **22.37** |

---

## 6. License & Citation / 开源协议与引用

This code is released under the MIT License (see [`LICENSE`](./LICENSE)). Foundation model weights (**UNI** and **CONCH**) are subject to the Mahmood Lab non-commercial academic license on HuggingFace.
