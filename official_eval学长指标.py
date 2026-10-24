#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""组内官方评估脚本(来自师姐模板 1.txt), 仅改动:
1) 三个文件夹路径改为命令行参数: official_eval.py <condition> <generated> <ground_truth> [out.json]
2) 末尾增加 JSON 存档
指标实现(PSNR/SSIM/L1/PCC/Cellpose-Dice/FID/KID/LPIPS)与模板逐字一致, 勿改动。"""
import os, sys, json
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from skimage.metrics import structural_similarity as ssim
from skimage.filters import threshold_otsu
from scipy.stats import pearsonr
from cellpose import models
from natsort import natsorted
import math
from tqdm import tqdm


# ==========================================
# 1. 临床指标: 细胞核分割 Dice (Nuclei Segmentation Dice)
# ==========================================
def compute_nuclei_dice(tar_img, gen_img, cell_model):
    tar_rgb = cv2.cvtColor(tar_img, cv2.COLOR_BGR2RGB)
    gen_rgb = cv2.cvtColor(gen_img, cv2.COLOR_BGR2RGB)
    tar_masks, _, _ = cell_model.eval(tar_rgb, diameter=None, channels=[0, 0])
    gen_masks, _, _ = cell_model.eval(gen_rgb, diameter=None, channels=[0, 0])
    tar_bin = (tar_masks > 0).astype(np.float32)
    gen_bin = (gen_masks > 0).astype(np.float32)
    intersection = np.sum(tar_bin * gen_bin)
    union = np.sum(tar_bin) + np.sum(gen_bin)
    if union == 0:
        return 1.0
    dice = 2.0 * intersection / (union + 1e-6)
    return dice


# ==========================================
# 2. 官方标准指标函数
# ==========================================
def psnr(gt_image, syn_image):
    gt = gt_image.astype(np.float32) / 255.0 if gt_image.dtype == np.uint8 else gt_image
    syn = syn_image.astype(np.float32) / 255.0 if syn_image.dtype == np.uint8 else syn_image
    mse_val = np.mean((gt - syn) ** 2)
    if mse_val == 0: return 100.0
    data_range = 1.0
    return 20 * math.log10(data_range / math.sqrt(mse_val))


def mse(gt_image, syn_image):
    gt = gt_image.astype(np.float32) / 255.0 if gt_image.dtype == np.uint8 else gt_image
    syn = syn_image.astype(np.float32) / 255.0 if syn_image.dtype == np.uint8 else syn_image
    return np.mean((gt - syn) ** 2)


def l1(gt_image, syn_image):
    gt = gt_image.astype(np.float32) / 255.0 if gt_image.dtype == np.uint8 else gt_image
    syn = syn_image.astype(np.float32) / 255.0 if syn_image.dtype == np.uint8 else syn_image
    return np.mean(np.abs(gt - syn))


def regional_h_score(image, cell_model):
    dab_signal = 255.0 - image[..., 0].astype(np.float32)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks, _, _ = cell_model.eval(img_rgb, diameter=None, channels=[0, 0])
    total_cells = len(np.unique(masks)) - 1
    if total_cells == 0: return 0.0
    try:
        thresh = threshold_otsu(dab_signal)
        if thresh < 30: thresh = 30
    except:
        thresh = 30
    weak, medium, strong = 0, 0, 0
    for cell_id in range(1, total_cells + 1):
        cell_mask = (masks == cell_id)
        cell_vals = dab_signal[cell_mask]
        if len(cell_vals) == 0: continue
        avg_val = np.mean(cell_vals)
        if avg_val > thresh:
            if avg_val < 100:
                weak += 1
            elif avg_val < 160:
                medium += 1
            else:
                strong += 1
    return (weak * 1 + medium * 2 + strong * 3) / total_cells * 100


def calculate_generic_pcc(gen_img, src_img):
    """PCC: 生成图(HE) vs 输入图(PI), 灰度, 不反转 (与模板一致)"""
    g_gen = cv2.cvtColor(gen_img, cv2.COLOR_BGR2GRAY).flatten().astype(np.float32)
    g_src = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY).flatten().astype(np.float32)
    g_src_inverted = g_src
    if np.var(g_gen) < 1e-5 or np.var(g_src_inverted) < 1e-5:
        return 0.0
    corr, _ = pearsonr(g_gen, g_src_inverted)
    return corr if not np.isnan(corr) else 0.0


# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    source_images_folder = sys.argv[1]   # condition (PI 输入)
    generated_images_folder = sys.argv[2]  # 生成图
    target_images_folder = sys.argv[3]   # ground_truth
    out_json = sys.argv[4] if len(sys.argv) > 4 else 'official_eval_result.json'
    kid_subset = int(sys.argv[5]) if len(sys.argv) > 5 else 50  # KID 子集大小, 需 ≤ 样本数

    metrics = {
        'psnr': [], 'ssim': [], 'mse': [], 'l1': [],
        'fid': None, 'kid': None, 'lpips': [],
        'pcc': [], 'nuclei_dice': [],
        'gen_h_score': [], 'tar_h_score': [], 'h_score_diff': []
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"condition : {source_images_folder}")
    print(f"generated : {generated_images_folder}")
    print(f"ground    : {target_images_folder}")

    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)
    cell_model = models.CellposeModel(gpu=torch.cuda.is_available(), model_type='nuclei')
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    kid = KernelInceptionDistance(subset_size=kid_subset, normalize=True).to(device)

    preprocess_fid = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])
    preprocess_lpips = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])

    print("Reading files with Natural Sort...")
    raw_src = natsorted([f for f in os.listdir(source_images_folder) if f.lower().endswith(('.png', '.jpg'))])
    raw_gen = natsorted([f for f in os.listdir(generated_images_folder) if f.lower().endswith(('.png', '.jpg'))])
    raw_tar = natsorted([f for f in os.listdir(target_images_folder) if f.lower().endswith(('.png', '.jpg'))])

    src_files = {f.lower(): f for f in raw_src}
    gen_files = {f.lower(): f for f in raw_gen}
    tar_files = {f.lower(): f for f in raw_tar}

    common_keys = set(src_files.keys()) & set(gen_files.keys()) & set(tar_files.keys())
    common = natsorted(list(common_keys))
    print(f"Valid samples: {len(common)}")

    for key in tqdm(common):
        src_path = os.path.join(source_images_folder, src_files[key])
        gen_path = os.path.join(generated_images_folder, gen_files[key])
        tar_path = os.path.join(target_images_folder, tar_files[key])

        src_img = cv2.imread(src_path)
        gen_img = cv2.imread(gen_path)
        tar_img = cv2.imread(tar_path)
        if key == common[0]:
            debug_view = np.hstack([gen_img, tar_img])
            cv2.imwrite("DEBUG_SSIM_CHECK.png", debug_view)
            print(f"DEBUG已保存比对图: DEBUG_SSIM_CHECK.png")
            print(f"Gen来自于: {gen_path}")
            print(f"Tar来自于: {tar_path}")

        if src_img is None or gen_img is None or tar_img is None: continue

        gen_np = gen_img.astype(np.float32) / 255.0
        tar_np = tar_img.astype(np.float32) / 255.0
        gen_chw = gen_np.transpose(2, 0, 1)
        tar_chw = tar_np.transpose(2, 0, 1)

        # --- 1. Pixel Metrics ---
        metrics['psnr'].append(psnr(tar_img, gen_img))
        metrics['mse'].append(mse(tar_img, gen_img))
        metrics['l1'].append(l1(tar_img, gen_img))

        # [Strict Official SSIM Alignment]
        metrics['ssim'].append(ssim(
            tar_chw, gen_chw,
            channel_axis=0,
            data_range=1.0,
            win_size=11,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False
        ))

        # --- 2. Structural & Clinical Metrics ---
        metrics['pcc'].append(calculate_generic_pcc(gen_img, src_img))
        n_dice = compute_nuclei_dice(tar_img, gen_img, cell_model)
        metrics['nuclei_dice'].append(n_dice)

        # --- 3. Deep Metrics (FID/LPIPS) ---
        gen_rgb = cv2.cvtColor(gen_img, cv2.COLOR_BGR2RGB)
        tar_rgb = cv2.cvtColor(tar_img, cv2.COLOR_BGR2RGB)

        gen_tensor_fid = preprocess_fid(gen_rgb).unsqueeze(0).to(device)
        tar_tensor_fid = preprocess_fid(tar_rgb).unsqueeze(0).to(device)

        fid.update(gen_tensor_fid, real=False)
        fid.update(tar_tensor_fid, real=True)
        kid.update(gen_tensor_fid, real=False)
        kid.update(tar_tensor_fid, real=True)

        gen_tensor_lpips = preprocess_lpips(gen_rgb).unsqueeze(0).to(device)
        tar_tensor_lpips = preprocess_lpips(tar_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            metrics['lpips'].append(lpips_metric(gen_tensor_lpips, tar_tensor_lpips).item())

    # Output
    print("\n" + "=" * 60)
    print("Evaluation Results (Fully Aligned with Official Standards)")
    print("=" * 60)

    res = {}
    if len(common) > 0:
        try:
            fid_val = fid.compute().item()
            kid_val = kid.compute()[0].item()
        except:
            fid_val, kid_val = -1, -1

        print("--- Image Quality (Official) ---")
        print(f"PSNR:  {np.mean(metrics['psnr']):.4f}")
        print(f"SSIM:  {np.mean(metrics['ssim']):.4f} (Gaussian, win=11, sigma=1.5)")
        print(f"LPIPS: {np.mean(metrics['lpips']):.4f} (AlexNet)")
        print(f"FID:   {fid_val:.4f}")
        print(f"KID:   {kid_val:.5f}")
        print(f"MSE:   {np.mean(metrics['mse']):.4f}")
        print(f"L1:    {np.mean(metrics['l1']):.4f}")

        print("\n--- Structural & Clinical Value ---")
        print(f"Nuclei Dice:  {np.mean(metrics['nuclei_dice']):.4f} (Segmentation Consistency)")
        print(f"PCC:          {np.mean(metrics['pcc']):.4f} (Intensity Correlation)")

        res = {
            'n': len(common),
            'generated_folder': generated_images_folder,
            'psnr': float(np.mean(metrics['psnr'])),
            'ssim': float(np.mean(metrics['ssim'])),
            'lpips': float(np.mean(metrics['lpips'])),
            'fid': float(fid_val),
            'kid': float(kid_val),
            'mse': float(np.mean(metrics['mse'])),
            'l1': float(np.mean(metrics['l1'])),
            'nuclei_dice': float(np.mean(metrics['nuclei_dice'])),
            'pcc': float(np.mean(metrics['pcc'])),
        }
        json.dump(res, open(out_json, 'w'), indent=1)
        print(f"已存: {out_json}")
    else:
        print("No images processed.")
