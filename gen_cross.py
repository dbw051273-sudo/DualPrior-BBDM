#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""供体×受体矩阵补格：在指定受体器官测试集上生成 + 学长口径评估。
与 official_screen_*（screen_brain.py 系）完全同协议：batch-1 前向、六步采样、
eta=[1,1,0,0,0,0]、种子 1234、输入 1024→256 重采样 + 极性取反。
用法: python gen_cross.py <cfg> <ckpt> <organ> [N] [tag]
  organ ∈ liver(750)/kidney(800)/brain(1000)
  tag 默认 = <ckpt文件名去.pth>_k<N>
产物: gens_screen/<tag>/ + official_screen_<tag>.json"""
import os
import sys
import time
import subprocess

import torch
from PIL import Image
import torchvision.transforms as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import eval_lib as E

MAP = {'liver': ('eval_liver', 750), 'kidney': ('eval_kidney', 800), 'brain': ('eval_brain', 1000)}

if __name__ == '__main__':
    cfg, ckpt, organ = sys.argv[1], sys.argv[2], sys.argv[3]
    evaldir, nmax = MAP[organ]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else nmax
    tag = sys.argv[5] if len(sys.argv) > 5 else \
        os.path.basename(ckpt).replace('.pth', '') + f'_k{n}'
    polarity = sys.argv[6] if len(sys.argv) > 6 else 'neg'   # 肝供体须用 pos（其 PI 存储极性相反）
    PY = sys.executable
    DEVICE = torch.device('cuda:0')

    model = E.load_model(cfg, ckpt, sample_step=6)
    model.eta = [1, 1, 0, 0, 0, 0]
    inp_f = sorted(os.listdir(f'{evaldir}/condition'))[:n]
    out = f'gens_screen/{tag}'
    os.makedirs(out, exist_ok=True)
    tf = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    t0 = time.time()
    for i, fi in enumerate(inp_f):
        ci = tf(Image.open(f'{evaldir}/condition/{fi}').convert('RGB'))
        ci = (ci - 0.5) * 2
        if polarity == 'neg':
            ci = -ci
        with torch.no_grad():
            s = model.sample(ci.unsqueeze(0).to(DEVICE), clip_denoised=True, task_id=0)[0].cpu()
        Image.fromarray((E.m01(s).permute(1, 2, 0).numpy() * 255).astype('uint8')).save(
            f'{out}/{os.path.splitext(fi)[0]}.png')
        if (i + 1) % 100 == 0:
            print(f'[{i + 1}/{len(inp_f)}] {(time.time() - t0) / 60:.0f}min', flush=True)
    print(f'生成完成: {(time.time() - t0) / 60:.1f} 分钟', flush=True)
    out_json = f'official_screen_{tag}.json'
    r = subprocess.run([PY, 'official_eval学长指标.py', f'{evaldir}/condition', out,
                        f'{evaldir}/ground_truth', out_json, '50'])
    print(f'完成 exit={r.returncode}: {out} + {out_json}')
