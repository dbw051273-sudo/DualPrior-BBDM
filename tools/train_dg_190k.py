"""Resolve server paths, validate data pairs and launch a single-GPU run."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def files(directory):
    return sorted(p.name for p in directory.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)


def check_data(cfg, data):
    validation_count = 0
    for organ in cfg['data']['dataset_config']['organs']:
        path = data / Path(organ['path']).name
        organ['path'] = str(path)
        for split in ('train', 'val', 'test'):
            pi, he = (files(path / split / kind) for kind in ('input', 'target'))
            if not pi or pi != he:
                raise ValueError(f'Missing or unmatched PI/H&E pairs: {path / split}')
            print(f'{path.name:6} {split:5} {len(pi):5}', flush=True)
            if split == 'val':
                validation_count += len(pi)
    print(f'Source validation pairs: {validation_count}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', default='run')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    check_data(config, Path(args.data_root).resolve())
    if args.check_only:
        return
    import torch

    if torch.cuda.device_count() != 1:
        raise RuntimeError('Use train.sh to bind one physical GPU')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('This training configuration requires CUDA BF16 support')
    print(f"CUDA device 0: {torch.cuda.get_device_name(0)} ({os.environ.get('CUDA_VISIBLE_DEVICES')})",
          flush=True)
    run = Path(args.out).resolve()
    run.mkdir(parents=True, exist_ok=False)
    config_file = run / 'run.yaml'
    config_file.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    command = [sys.executable, 'main.py', '-c', str(config_file), '--train', '--seed', '1234',
               '--gpu_ids', '0', '--result_path', str(run)]
    subprocess.run(command, cwd=ROOT, check=True)
    checkpoint = run / config['data']['dataset_name'] / config['model']['model_name'] / 'checkpoint/last_model.pth'
    if not checkpoint.is_file():
        raise RuntimeError(f'Missing final checkpoint: {checkpoint}')
    print(f'Checkpoint: {checkpoint}', flush=True)


if __name__ == '__main__':
    main()
