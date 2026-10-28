#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""服务器端高吞吐极速批量生图脚本 (gen_breast_server.py)
利用服务器大显存 (RTX 5880 48G / RTX A5000 24G) 与多核 CPU:
1) PyTorch DataLoader 多进程预读 (num_workers=4, pin_memory=True)
2) 大批次采样 (默认 batch_size=32, 可设 48)
3) 6 步采样 + 零拷贝推流, 5,636 张全量图仅需 6~8 分钟
4) 生图完毕自动打 tar.gz 压缩包 (~250MB), 笔记本 15 秒下载解压

用法:
  python gen_breast_server.py <cfg> <ckpt> [n=5636] [batch_size=32] [tag]
"""
import os, sys, time, tarfile
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
try:
    from natsort import natsorted
except ImportError:
    import re
    def natsorted(l):
        return sorted(l, key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(s))])

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_lib as E

class BreastCondDataset(Dataset):
    def __init__(self, cond_dir, file_list):
        self.cond_dir = cond_dir
        self.files = file_list
        self.tf = T.Compose([T.Resize((256, 256)), T.ToTensor()])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        img_p = os.path.join(self.cond_dir, f)
        img = Image.open(img_p).convert('RGB')
        t = self.tf(img)
        ci = -((t - 0.5) * 2)  # 严格对齐 BBDM PI 输入规范
        return ci, os.path.splitext(f)[0]

def main():
    if len(sys.argv) < 3:
        print("用法: python gen_breast_server.py <cfg> <ckpt> [n=5636] [batch_size=32] [tag]")
        sys.exit(1)

    cfg = sys.argv[1]
    ckpt = sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 5636
    batch_size = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    tag = sys.argv[5] if len(sys.argv) > 5 else os.path.basename(ckpt).replace('.pth', '') + f'_b{n}'

    # 自动探测乳腺数据集 input 路径
    possible_roots = [
        '/media/simon/f660aae3-bbc5-4f38-976a-d622ee37ae7c/mali/user2/wangdb/Datasets_processed/Breast_all/input',
        '/data/wangdb/data/Datasets_processed/Breast/input',
        '/media/simon/f660aae3-bbc5-4f38-976a-d622ee37ae7c/mali/user2/wangdb/Datasets_processed/Breast/input',
        'official_eval/condition',
        'C:/Users/felix/Desktop/Datasets_processed/Breast/input'
    ]
    cond_dir = sys.argv[6] if len(sys.argv) > 6 else None
    if cond_dir is None:
        for r in possible_roots:
            if os.path.exists(r):
                cnt = len([f for f in os.listdir(r) if f.lower().endswith(('.png', '.jpg'))])
                if cnt >= n:
                    cond_dir = r
                    break
                elif cond_dir is None and cnt > 0:
                    cond_dir = r

    if cond_dir is None:
        raise RuntimeError("未找到乳腺 input 条件图目录！")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("=" * 60)
    print("【Gen-Breast-Server】服务器高吞吐批量生图引擎")
    print("=" * 60)
    print(f"配置文件: {cfg}")
    print(f"权重文件: {ckpt}")
    print(f"条件目录: {cond_dir}")
    print(f"样本数量: {n}")
    print(f"BatchSize: {batch_size}")
    print(f"输出Tag : {tag}")
    print(f"设备    : {device}")

    # 加载模型
    print("正在加载模型 (6步线性采样, eta=[1,1,0,0,0,0])...")
    import yaml
    from utils import dict2namespace
    from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel

    with open(cfg) as f:
        try:
            raw_cfg = yaml.load(f, Loader=yaml.FullLoader)
        except Exception:
            f.seek(0)
            raw_cfg = yaml.load(f, Loader=yaml.UnsafeLoader)

    model_cfg = raw_cfg.model if hasattr(raw_cfg, 'model') else dict2namespace(raw_cfg).model
    model = BrownianBridgeModel(model_cfg)
    st = torch.load(ckpt, map_location='cpu')
    if st.get('swad'):
        print(f">>> [权重加载] 成功加载 SWAD 权重 (已密集平均 {st.get('swad_n', '多')} 步平坦极小值)！")
        model.load_state_dict(st['swad'], strict=False)
    elif st.get('ema'):
        print(">>> [权重加载] 成功加载 EMA 影子权重！")
        model.load_state_dict(st['ema'], strict=False)
    else:
        print(">>> [权重加载] 加载基础 Model 权重！")
        model.load_state_dict(st['model'], strict=False)
    model.sample_step = 6
    model.skip_sample = True
    model.sample_type = 'linear'
    model.register_schedule()
    model.eta = [1, 1, 0, 0, 0, 0]
    model = model.to(device)
    model.eval()

    all_files = natsorted([f for f in os.listdir(cond_dir) if f.lower().endswith(('.png', '.jpg'))])[:n]
    dataset = BreastCondDataset(cond_dir, all_files)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    out_dir = os.path.join('gens_screen', tag)
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)

    t0 = time.time()
    total_done = 0
    with torch.inference_mode():
        for batch_ci, batch_names in loader:
            batch_ci = batch_ci.to(device, non_blocking=True)
            samples = model.sample(batch_ci, clip_denoised=True, task_id=0).cpu()

            for j in range(len(batch_names)):
                name = batch_names[j]
                img_np = (E.m01(samples[j]).permute(1, 2, 0).numpy() * 255).astype('uint8')
                Image.fromarray(img_np).save(os.path.join(out_dir, f"{name}.png"))

            total_done += len(batch_names)
            elapsed = time.time() - t0
            speed = total_done / (elapsed + 1e-6)
            remain = (n - total_done) / (speed + 1e-6)
            if total_done % (batch_size * 4) == 0 or total_done == n:
                print(f"[{total_done}/{n}] ({total_done/n*100:.1f}%) "
                      f"已用时: {elapsed/60:.1f}m, 速度: {speed:.1f}图/s, ETA: {remain/60:.1f}m", flush=True)

    total_time = (time.time() - t0) / 60
    print(f"\n生图全部完成! 总耗时: {total_time:.1f} 分钟 ({speed:.1f} 张/秒)")

    # 打包压缩
    tar_path = f"gens_screen/{tag}.tar.gz"
    print(f"正在打包压缩生成图为 {tar_path} ...")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(out_dir, arcname=tag)
    tar_size_mb = os.path.getsize(tar_path) / (1024 * 1024)
    print(f"打包完成! 压缩包大小: {tar_size_mb:.1f} MB")
    print("=" * 60)
    print(f"拉取命令: scp user@server:{os.path.abspath(tar_path)} gens_screen/")
    print(f"解压命令: tar -xzf gens_screen/{tag}.tar.gz -C gens_screen/")
    print("=" * 60)

if __name__ == '__main__':
    main()
