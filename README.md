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
2. **Controlled Domain Generalization (DG) Benchmarks (6 Major Categories)**:
   - Systematically reproduces 6 classical DG baseline families on the identical `Swin-UNet BBDM` backbone under single-variable controlled 190,000-step training:
     - **Class 1 (Data Augmentation)**: *StainAug* (Tellez et al., IEEE TMI 2019) Beer–Lambert stain perturbation;
     - **Class 2 (Feature Alignment)**: *DANN* (Ganin et al., JMLR 2016) Gradient Reversal Layer on 2048-dim bottleneck;
     - **Class 3 (Feature Disentanglement)**: *MixStyle* (Zhou et al., ICLR 2021) feature statistics mixing;
     - **Class 4 (Foundation Model Baseline)**: *UNIStainNet* multi-modal cross-attention;
     - **Class 5 (Meta-Learning)**: *MLDG* (Li et al., AAAI 2018) episodic virtual-test optimization;
     - **Class 6 (Flat Minima Optimization)**: *SWAD* (Cha et al., NeurIPS 2021) stochastic dense weight averaging.
3. **Two-Factor Controlled Attribution (Polarity Alignment & Stain Statistics Calibration)**:
   - **Stage I (Zero-Shot Input Polarity Alignment)**: Aligning donor-recipient optical contrast polarity lifts the single-organ Liver donor on unseen Breast from `8.23 dB` to `20.04 dB` without any retraining or target H&E access.
   - **Stage II (Unpaired Site Reference Calibration)**: Lightweight Beer–Lambert optical density (OD) affine calibration using $n \ge 3$ unpaired archive H&E slides compresses all 4 donor models (`Liver`, `Kidney`, `Brain`, `Joint`) into a `22.01–23.39 dB` band ($\Delta < 1.4\text{ dB}$).
4. **4-Level Automated Pathology Evaluation Suite (`fast_official_eval.py`)**:
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
│   ├── stainaug_190k_server.yaml   # DG Class 1: StainAug (Optical Density stain jitter)
│   ├── dann_190k_server.yaml       # DG Class 2: DANN (GRL + 3-Organ Domain Discriminator)
│   ├── mixstyle_190k_server.yaml   # DG Class 3: MixStyle (Feature statistics mixing)
│   ├── unistain_190k_server.yaml   # DG Class 4: UNIStainNet (Cross-modal foundation prior)
│   ├── mldg_190k_server.yaml       # DG Class 5: MLDG (Meta-learning domain generalization)
│   ├── swad_190k_server.yaml       # DG Class 6: SWAD (Dense flat-minima weight averaging)
│   ├── i2sb_joint_190k.yaml        # Baseline: I2SB (Schrödinger Bridge, 190k steps)
│   ├── gddpm_joint_240.yaml        # Baseline: Conditional DDIM
│   └── moe_joint_server.yaml       # Negative Evidence: Organ-MoE soft routing
├── data_splits/                    # Complete deterministic split filelists & metadata
│   ├── dataset_manifest.json       # Exact counts, optical polarity conventions & access policy
│   ├── liver_train_filelist.txt    # 3,000 train / 279 val / 750 test filenames
│   ├── kidney_train_filelist.txt   # 2,000 train / 194 val / 800 test filenames
│   ├── brain_train_filelist.txt    # 4,000 train / 863 val / 1,000 test filenames
│   └── breast_test_blind_filelist.txt # 5,636 blind test filenames
├── datasets/                       # Multi-organ paired dataset loaders & stain augmentations
│   ├── custom.py                   # Paired multi-organ dataset with polarity transforms
│   └── stain_aug.py                # Ruifrok-Beer-Lambert optical stain augmentor (StainAug)
├── model/                          # Swin-UNet BBDM, UNI SpatialPriorInjector & GRL/MixStyle modules
│   └── BrownianBridge/
│       ├── BrownianBridgeModel.py  # Core BBDM with dual-prior, GRL, and MLDG support
│       ├── grl.py                  # Gradient Reversal Layer & Domain Discriminator (DANN)
│       └── mixstyle.py             # Feature statistics mixer (MixStyle)
├── runners/                        # Training, SWAD optimization, and 6-step/2-step sampling loops
│   ├── BaseRunner.py               # Base runner with dense stochastic weight averaging (SWAD)
│   └── DiffusionBasedModelRunners/
│       └── BBDMRunner.py           # Multi-organ BBDM training and test runner
├── main.py                         # Unified entry point for training and testing
├── precompute_uni.py               # Offline UNI dense token extraction (accelerates training)
├── build_conch_bank.py             # Offline source-only CONCH memory bank builder
├── gen_breast_server.py            # High-throughput server-side 6-step batch sampler
├── gen_cross.py                    # Cross-organ zero-shot inference & polarity alignment script
├── eval_lib.py                     # Core evaluation & Beer-Lambert OD site calibration library
├── fast_official_eval.py           # Fast 4-level evaluator with precomputed GT Cellpose mask cache
└── official_eval.py                # Standalone 4-level automated pathology evaluation script
```

---

## 3. Dataset Organization & Data Availability / 数据集结构与获取政策

### Data Availability Statement (数据可用性说明)
- **Full Raw Dataset (全量原始数据)**: The complete multi-organ paired PI$\to$H&E dataset (**15,222 paired patches** across Liver, Kidney, Brain, and Breast) was acquired under institutional collaboration protocols at Hainan University. Due to institutional data privacy policies, the full raw image archive is not hosted directly in this public repository, but is available from the corresponding author upon reasonable academic request and advisor approval.
- **Reproducibility Assets Included Here (本仓库已公开的复现资产)**:
  - **`data_splits/`**: Exact deterministic file lists (`*_filelist.txt`) and optical polarity metadata (`dataset_manifest.json`) for every split of all 4 organs.

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
└── Breast_all/
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

### 4.3 Training Main Dual-Prior BBDM (主模型训练，190k 步受控)
```bash
# Train 3-Organ Joint Dual-Prior BBDM (Main Model)
CUDA_VISIBLE_DEVICES=0 python main.py -c configs/full_e3.yaml --train --gpu_ids 0

# Train Single-Organ Donor Models
CUDA_VISIBLE_DEVICES=0 python main.py -c configs/base_liver_server.yaml --train --gpu_ids 0
CUDA_VISIBLE_DEVICES=0 python main.py -c configs/base_kidney_server.yaml --train --gpu_ids 0
CUDA_VISIBLE_DEVICES=0 python main.py -c configs/base_brain_server.yaml --train --gpu_ids 0
```

### 4.4 Training Classical Domain Generalization (DG) Baselines (DG 基准复现)
All DG baselines use the identical `Swin-UNet BBDM` backbone and 190,000 steps controlled schedule:
```bash
# DG Class 1: StainAug (Data Diversity & Stain Augmentation)
CUDA_VISIBLE_DEVICES=0 python main.py -c configs/stainaug_190k_server.yaml --train --gpu_ids 0

# DG Class 2: DANN (Gradient Reversal Layer + 3-Organ Domain Discriminator)
CUDA_VISIBLE_DEVICES=0 python main.py -c configs/dann_190k_server.yaml --train --gpu_ids 0

# DG Class 3: MixStyle (Feature Statistics Mixing)
CUDA_VISIBLE_DEVICES=0 python main.py -c configs/mixstyle_190k_server.yaml --train --gpu_ids 0

# DG Class 6: SWAD (Dense Stochastic Weight Averaging on Flat Minima)
CUDA_VISIBLE_DEVICES=0 python main.py -c configs/swad_190k_server.yaml --train --gpu_ids 0
```

### 4.5 High-Throughput Batch Inference & Fast Official Evaluation (批量生图与极速官方评测)
```bash
# 1. Server-side high-throughput 6-step generation (batch_size=32/64, ~4-6 min for 5,636 patches)
python gen_breast_server.py configs/dann_190k_server.yaml results/ws300-e3/DANN-190K/checkpoint/last_model.pth 5636 64 dann_190k

# 2. Fast 4-level official evaluation using precomputed GT Cellpose mask cache (~18 min)
python fast_official_eval.py official_eval/condition gens_screen/dann_190k official_eval/ground_truth official_screen_dann190k_b5636.json 50
```

---

## 5. Benchmark Results / 核心实验基准

### 5.1 Controlled Cross-Organ DG Benchmark on 5,636 Unseen Breast Images (论文表 3 Panel A)
All models trained on 3 source organs (**Liver, Kidney, Brain**; 9,000 pairs) under 190,000 steps controlled schedule, evaluated on **5,636 blind breast patches**:

| Model Configuration / DG Category | Reference | PSNR (dB)↑ | SSIM↑ | LPIPS↓ | FID↓ | Cellpose Dice↑ | H-channel PCC↑ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pix2Pix** (1 step) | CVPR 2017 (Isola) | 15.23 | 0.665 | 0.286 | 75.76 | 0.536 | 0.818 |
| **CycleGAN** (1 step) | ICCV 2017 (Zhu) | 10.18 | -0.001 | 0.505 | 134.02 | 0.024 | -0.654 |
| **Conditional DDIM** (50 steps) | ICLR 2021 (Song) | 12.42 | 0.570 | 0.360 | 78.65 | 0.504 | 0.739 |
| **I2SB Schrödinger Bridge** (20 steps) | ICML 2023 (Chen) | 16.23 | 0.747 | 0.222 | 42.93 | 0.549 | 0.833 |
| **Ablation: Base Backbone** (6 steps) | Swin-UNet BBDM | 16.57 | 0.757 | 0.215 | 38.99 | 0.535 | 0.835 |
| **DG Class 1: StainAug** (6 steps) | IEEE TMI 2019 (Tellez) | 15.93 | 0.756 | 0.222 | 40.99 | 0.541 | 0.831 |
| **DG Class 2: DANN** (6 steps) | JMLR 2016 (Ganin) | 14.35 | 0.741 | 0.276 | 52.40 | 0.535 | 0.774 |
| **DG Class 6: SWAD** (6 steps) | NeurIPS 2021 (Cha) | 15.71 | 0.741 | 0.244 | 44.39 | 0.528 | 0.794 |
| **DualPrior (Ours)** (6 steps) | This work | 18.31 | 0.783 | **0.210** | **38.23** | 0.546 | 0.861 |
| **DualPrior (Ours)** (2 steps) | This work (Fast Sampling) | **18.47** | **0.785** | 0.211 | 42.24 | **0.551** | **0.865** |

### 5.2 24-Cell Donor × Recipient Matrix (PSNR in dB at 190k Steps)
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

If you find this work or benchmark useful for your research, please cite:
```bibtex
@article{wang2026dualprior,
  title={Input Polarity and Stain Statistics: Controlled Attribution and a Dual-Prior Brownian Bridge Diffusion Framework for Cross-Organ Zero-Shot Virtual Staining},
  author={Wang, Dabin and Ma, Li and collaborators},
  journal={IEEE Transactions on Medical Imaging (Under Review)},
  year={2026}
}
```
