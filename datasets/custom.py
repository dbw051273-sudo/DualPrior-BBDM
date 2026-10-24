import random
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from Register import Registers
from .base import ImagePathDataset
from .utils import get_image_paths_from_dir
from PIL import Image
import cv2
import os


@Registers.datasets.register_with_name('custom_single')
class CustomSingleDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        image_paths = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, stage))
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal

        self.imgs = ImagePathDataset(image_paths, self.image_size, flip=self.flip, to_normal=self.to_normal)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.imgs[i], self.imgs[i]


@Registers.datasets.register_with_name('custom_aligned')
class CustomAlignedDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        dir_target = getattr(dataset_config, 'dir_target', 'B')
        dir_cond = getattr(dataset_config, 'dir_cond', 'A')
        image_paths_ori = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, f'{stage}/{dir_target}'))
        image_paths_cond = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, f'{stage}/{dir_cond}'))
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal
        self.invert_cond = getattr(dataset_config, 'invert_cond', True)

        self.imgs_ori = ImagePathDataset(image_paths_ori, self.image_size, flip=self.flip, to_normal=self.to_normal)
        self.imgs_cond = ImagePathDataset(image_paths_cond, self.image_size, flip=self.flip, to_normal=self.to_normal)

    def __len__(self):
        return len(self.imgs_ori)

    def __getitem__(self, i):
        target_img, target_name = self.imgs_ori[i]
        cond_img, cond_name = self.imgs_cond[i]
        if self.invert_cond:
            cond_img = -1.0 * cond_img
        return (target_img, target_name), (cond_img, cond_name)


@Registers.datasets.register_with_name('multi_organ')
class MultiOrganDataset(Dataset):
    """多器官混合数据集：每个样本携带 organ_label。

    config 示例:
      dataset_type: 'multi_organ'
      organs:
        - path: 'C:/.../liver'
          label: 0
        - path: 'C:/.../Kidney'
          label: 1
        - path: 'C:/.../Brain'
          label: 2
      per_organ_limit: 15          # 每个器官最多取 N 张（压测用）
    """
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        dir_target = getattr(dataset_config, 'dir_target', 'target')
        dir_cond = getattr(dataset_config, 'dir_cond', 'input')
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal
        self.invert_cond = getattr(dataset_config, 'invert_cond', True)
        self.photometric_aug = getattr(dataset_config, 'photometric_aug', False) and stage == 'train'
        # 几何增广: 成对同参数 rot90+水平翻转(病理图无先天朝向), 仅训练期; 与反转可交换
        self.geo_aug = getattr(dataset_config, 'geo_aug', False) and stage == 'train'
        # E5: 逐图 PI 标准化([0,1]域) —— 把 liver 从偏暗带(-0.02)拉进亮器官带(-0.50),
        # Breast 评估输入几乎不变; 与 photometric_aug 互斥(勿同开)
        self.standardize_cond = getattr(dataset_config, 'standardize_cond', False)
        per_organ_limit = getattr(dataset_config, 'per_organ_limit', None)

        self.samples = []  # [(ImagePathDataset_ori_idx, ImagePathDataset_cond_idx, organ_label), ...]
        self.organs = dataset_config.organs

        for organ in self.organs:
            base = organ['path'] if isinstance(organ, dict) else organ.path
            label = organ['label'] if isinstance(organ, dict) else organ.label
            tp = os.path.join(base, f'{stage}/{dir_target}')
            cp = os.path.join(base, f'{stage}/{dir_cond}')
            t_list = sorted(get_image_paths_from_dir(tp))
            c_list = sorted(get_image_paths_from_dir(cp))
            if per_organ_limit:
                t_list = t_list[:per_organ_limit]
                c_list = c_list[:per_organ_limit]
            n = min(len(t_list), len(c_list))
            for j in range(n):
                self.samples.append((t_list[j], c_list[j], label))
        print(f"[MultiOrganDataset/{stage}] {len(self.samples)} samples from {len(self.organs)} organs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        t_path, c_path, label = self.samples[i]
        # 用 ImagePathDataset 的逻辑加载单张图
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.0),
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])
        target_img = Image.open(t_path)
        cond_img = Image.open(c_path)
        if target_img.mode != 'RGB':
            target_img = target_img.convert('RGB')
        if cond_img.mode != 'RGB':
            cond_img = cond_img.convert('RGB')
        target_img = transform(target_img)
        cond_img = transform(cond_img)
        if self.geo_aug:
            k = random.randint(0, 3)
            if k:
                target_img = torch.rot90(target_img, k, (1, 2))
                cond_img = torch.rot90(cond_img, k, (1, 2))  # 与 target 同一 k, 保配准
            if random.random() < 0.5:
                target_img = torch.flip(target_img, (2,))
                cond_img = torch.flip(cond_img, (2,))        # 一起翻
        if self.standardize_cond:
            mu, sig = cond_img.mean(), cond_img.std().clamp_min(0.05)
            cond_img = ((cond_img - mu) / sig * 0.22 + 0.75).clamp(0., 1.)
        if self.to_normal:
            target_img = (target_img - 0.5) * 2
            target_img.clamp_(-1., 1.)
            cond_img = (cond_img - 0.5) * 2
            cond_img.clamp_(-1., 1.)
        if self.invert_cond:
            cond_img = -1.0 * cond_img
        # PI 输入光度增广（仅输入侧，配对合法）：直击跨器官亮度分布偏移（liver 123 ↔ Breast 209）
        if self.photometric_aug and random.random() < 0.5:
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.85, 1.15)
            mid = cond_img.mean()
            cond_img = ((cond_img - mid) * contrast + mid) * brightness
            cond_img = cond_img.clamp(-1.0, 1.0)
        name = Path(t_path).stem
        return (target_img, name), (cond_img, name), label


@Registers.datasets.register_with_name('custom_colorization_LAB')
class CustomColorizationLABDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.image_paths = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, stage))
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal
        self._length = len(self.image_paths)

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = False
        if index >= self._length:
            index = index - self._length
            p = True

        img_path = self.image_paths[index]
        image = None
        try:
            image = cv2.imread(img_path)
            if self.to_lab:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        except BaseException as e:
            print(img_path)

        if p:
            image = cv2.flip(image, 1)
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        image = torch.Tensor(image)
        image = image.permute(2, 0, 1).contiguous()

        if self.to_normal:
            image = (image - 127.5) / 127.5
            image.clamp_(-1., 1.)

        L = image[0:1, :, :]
        ab = image[1:, :, :]
        cond = torch.cat((L, L, L), dim=0)
        return image, cond


@Registers.datasets.register_with_name('custom_colorization_RGB')
class CustomColorizationRGBDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.image_paths = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, stage))
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal
        self._length = len(self.image_paths)

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = False
        if index >= self._length:
            index = index - self._length
            p = True

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=p),
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

        img_path = self.image_paths[index]
        image = None
        try:
            image = Image.open(img_path)
        except BaseException as e:
            print(img_path)

        if not image.mode == 'RGB':
            image = image.convert('RGB')

        cond_image = image.convert('L')
        cond_image = cond_image.convert('RGB')

        image = transform(image)
        cond_image = transform(cond_image)

        if self.to_normal:
            image = (image - 0.5) * 2.
            image.clamp_(-1., 1.)
            cond_image = (cond_image - 0.5) * 2.
            cond_image.clamp_(-1., 1.)

        image_name = Path(img_path).stem
        return (image, image_name), (cond_image, image_name)


@Registers.datasets.register_with_name('custom_inpainting')
class CustomInpaintingDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.image_paths = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, stage))
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal
        self._length = len(self.image_paths)

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = 0.
        if index >= self._length:
            index = index - self._length
            p = 1.

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=p),
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

        img_path = self.image_paths[index]
        image = None
        try:
            image = Image.open(img_path)
        except BaseException as e:
            print(img_path)

        if not image.mode == 'RGB':
            image = image.convert('RGB')

        image = transform(image)

        if self.to_normal:
            image = (image - 0.5) * 2.
            image.clamp_(-1., 1.)

        height, width = self.image_size
        mask_width = random.randint(128, 180)
        mask_height = random.randint(128, 180)
        mask_pos_x = random.randint(0, height - mask_height)
        mask_pos_y = random.randint(0, width - mask_width)
        mask = torch.ones_like(image)
        mask[:, mask_pos_x:mask_pos_x+mask_height, mask_pos_y:mask_pos_y+mask_width] = 0

        cond_image = image * mask

        image_name = Path(img_path).stem
        return (image, image_name), (cond_image, image_name)
