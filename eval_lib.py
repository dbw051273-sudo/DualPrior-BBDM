#!/usr/bin/env python
"""eval_lib.py — 可导入的评估函数库（6 指标 + 滑窗 Patch FID）"""
import os, sys, torch, numpy as np
from PIL import Image
import torchvision.transforms as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def m01(t):
    return ((t + 1) / 2).clamp(0, 1)

def load_organ_pairs(organ='Brain', n=10, standardize=False):
    """加载任意器官 test 集图对（域内评估用）。organ: liver/Kidney/Brain/Breast"""
    root = 'C:/Users/felix/Desktop/Datasets_processed'
    sub = '' if organ == 'Breast' else '/test'
    dp = f'{root}/{organ}{sub}'
    inp_f = sorted(os.listdir(f'{dp}/input'))[:n]
    tgt_f = sorted(os.listdir(f'{dp}/target'))[:n]
    tf = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    pairs = []
    for fi, ft in zip(inp_f, tgt_f):
        ci = tf(Image.open(f'{dp}/input/{fi}').convert('RGB'))
        ti = tf(Image.open(f'{dp}/target/{ft}').convert('RGB'))
        if standardize:
            mu, sig = ci.mean(), ci.std().clamp_min(0.05)
            ci = ((ci - mu) / sig * 0.22 + 0.75).clamp(0., 1.)
        ci = -((ci - 0.5) * 2)
        ti = (ti - 0.5) * 2
        pairs.append((ci, ti))
    return pairs


def load_breast_pairs(n=10, standardize=False):
    """加载 Breast 测试图对（PI 输入 + H&E 真值）。
    standardize=True 时与 MultiOrganDataset 的 standardize_cond 完全同式(E5)。"""
    dp = 'C:/Users/felix/Desktop/Datasets_processed/Breast'
    inp_f = sorted(os.listdir(f'{dp}/input'))[:n]
    tgt_f = sorted(os.listdir(f'{dp}/target'))[:n]
    tf = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    pairs = []
    for fi, ft in zip(inp_f, tgt_f):
        ci = tf(Image.open(f'{dp}/input/{fi}').convert('RGB'))
        ti = tf(Image.open(f'{dp}/target/{ft}').convert('RGB'))
        if standardize:
            mu, sig = ci.mean(), ci.std().clamp_min(0.05)
            ci = ((ci - mu) / sig * 0.22 + 0.75).clamp(0., 1.)
        ci = -((ci - 0.5) * 2)  # normalize + invert
        ti = (ti - 0.5) * 2
        pairs.append((ci, ti))
    return pairs

def load_model(cfg_path, ckpt_path, sample_step=20, use_ema=False):
    """加载模型并设置推理步数。use_ema=True 时用 checkpoint 里的 EMA 影子权重"""
    from utils import dict2namespace
    import yaml
    with open(cfg_path) as f:
        cfg = dict2namespace(yaml.load(f, Loader=yaml.FullLoader))
    mtype = getattr(cfg.model, 'model_type', 'BBDM')
    if mtype == 'GDDPM':
        from model.BrownianBridge.GaussianDiffusionModel import GaussianDiffusionModel
        m = GaussianDiffusionModel(cfg.model)
    elif mtype == 'I2SB':
        from model.BrownianBridge.I2SBModel import I2SBModel
        m = I2SBModel(cfg.model)
        m.sample_steps = sample_step
        m.register_schedule()
    else:
        from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
        m = BrownianBridgeModel(cfg.model)
        m.sample_step = sample_step
        m.skip_sample = True
        m.sample_type = 'linear'
        m.register_schedule()
    st = torch.load(ckpt_path, map_location='cpu')
    m.load_state_dict(st['model'], strict=False)
    m = m.to(DEVICE)
    if use_ema and st.get('ema'):
        m.load_state_dict(st['ema'], strict=False)
    m.eval()
    return m

def gen_samples(model, pairs, seed=None):
    """生成 H&E 样本。seed 非 None 时固定采样种子，使评估可复现（E0 起）"""
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    out = []
    for ci, ti in pairs:
        with torch.no_grad():
            s = model.sample(ci.unsqueeze(0).to(DEVICE), clip_denoised=True, task_id=0)
        out.append(s[0].cpu())
    return out

# ===== 6 指标 =====
def calc_psnr(gen, gt):
    from skimage.metrics import peak_signal_noise_ratio
    return peak_signal_noise_ratio(m01(gt).permute(1,2,0).numpy(), m01(gen).permute(1,2,0).numpy(), data_range=1.0)

def calc_ssim(gen, gt):
    from skimage.metrics import structural_similarity
    return structural_similarity(m01(gt).permute(1,2,0).numpy(), m01(gen).permute(1,2,0).numpy(), data_range=1.0, channel_axis=2)

def calc_pcc(gen, gt):
    return float(np.corrcoef(m01(gen).flatten().numpy(), m01(gt).flatten().numpy())[0, 1])

def calc_dice(gen, gt):
    from skimage.color import rgb2hed
    from skimage.filters import threshold_otsu
    g = (m01(gen).permute(1,2,0).numpy()*255).astype(np.uint8)
    t = (m01(gt).permute(1,2,0).numpy()*255).astype(np.uint8)
    gh, th = rgb2hed(g)[:,:,0], rgb2hed(t)[:,:,0]
    try:
        gm, tm = gh > threshold_otsu(gh), th > threshold_otsu(th)
    except Exception:
        return 0.0
    inter = (gm & tm).sum()
    denom = gm.sum() + tm.sum()
    return float(2*inter/(denom+1e-8)) if denom > 0 else 0.0

def calc_lpips(lp, gen, gt):
    with torch.no_grad():
        return lp(gen.unsqueeze(0).to(DEVICE), gt.unsqueeze(0).to(DEVICE)).item()

def slide_window_patches(images, window=128, stride=32):
    """把 list[Tensor 3,H,W] 按滑窗切成 Patch list[Tensor 3,window,window]"""
    patches = []
    for img in images:
        _, H, W = img.shape
        if H < window or W < window:
            patches.append(img)
            continue
        for y in range(0, H - window + 1, stride):
            for x in range(0, W - window + 1, stride):
                patches.append(img[:, y:y+window, x:x+window])
    return patches

def make_fid_calculator():
    """返回 compute(gens, gts) 函数，基于 InceptionV3"""
    import torchvision.models as models
    inc = models.inception_v3(weights='DEFAULT', aux_logits=True, transform_input=False)
    inc.fc = torch.nn.Identity()
    inc.eval().to(DEVICE)
    tf_incep = T.Compose([T.Resize((299,299)), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    def extract(imgs):
        fs = []
        for img in imgs:
            with torch.no_grad():
                f = inc(tf_incep(m01(img).unsqueeze(0)).to(DEVICE))
            fs.append(f.cpu().numpy().flatten())
        return np.array(fs)
    def compute(gens, gts):
        gf, tf_ = extract(gens), extract(gts)
        mu1, mu2 = gf.mean(0), tf_.mean(0)
        d = gf.shape[1]
        s1 = np.cov(gf, rowvar=False) + np.eye(d)*1e-6
        s2 = np.cov(tf_, rowvar=False) + np.eye(d)*1e-6
        from scipy.linalg import sqrtm
        cm = sqrtm(s1 @ s2)
        if np.iscomplexobj(cm): cm = cm.real
        return float(((mu1-mu2)**2).sum() + np.trace(s1+s2-2*cm))
    return compute

def eval_model(cfg_path, ckpt_path, pairs, sample_step=20, lp=None, fid_compute=None):
    """加载模型 → 生成 → 6 指标 → 返回 dict"""
    from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
    model = load_model(cfg_path, ckpt_path, sample_step)
    samples = gen_samples(model, pairs)
    del model
    torch.cuda.empty_cache()

    if lp is None:
        import lpips as lpips_mod
        lp = lpips_mod.LPIPS(net='alex').to(DEVICE)
    if fid_compute is None:
        fid_compute = make_fid_calculator()

    # 整图指标
    psnrs, ssims, pccs, dices, lpips_vals = [], [], [], [], []
    for s, (_, t) in zip(samples, pairs):
        psnrs.append(calc_psnr(s, t))
        ssims.append(calc_ssim(s, t))
        pccs.append(calc_pcc(s, t))
        dices.append(calc_dice(s, t))
        lpips_vals.append(calc_lpips(lp, s, t))

    # 滑窗 Patch FID
    gen_patches = slide_window_patches(samples, window=128, stride=32)
    tgt_patches = slide_window_patches([t for _, t in pairs], window=128, stride=32)
    fid = fid_compute(gen_patches, tgt_patches)

    return {
        'PSNR': float(np.mean(psnrs)),
        'SSIM': float(np.mean(ssims)),
        'LPIPS': float(np.mean(lpips_vals)),
        'PCC': float(np.mean(pccs)),
        'Dice': float(np.mean(dices)),
        'FID': fid,
        'n_patches': len(gen_patches),
    }
