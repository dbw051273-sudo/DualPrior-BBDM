
import os
import torch
import torch.optim.lr_scheduler
from torch import nn
from torch.utils.data import DataLoader
from PIL import Image
from tqdm.autonotebook import tqdm

from Register import Registers
from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
from model.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel
from model.BrownianBridge.GaussianDiffusionModel import GaussianDiffusionModel
from model.BrownianBridge.I2SBModel import I2SBModel
from runners.DiffusionBasedModelRunners.DiffusionBaseRunner import DiffusionBaseRunner
from runners.utils import weights_init, get_optimizer, get_dataset, make_dir, get_image_grid, save_single_image


@Registers.runners.register_with_name('BBDMRunner')
class BBDMRunner(DiffusionBaseRunner):
    def __init__(self, config):
        super().__init__(config)
        self.task_counter = 0
        self.is_latent_model = (config.model.model_type == "LBBDM")

    def initialize_model(self, config):
        if config.model.model_type == "BBDM":
            net = BrownianBridgeModel(config.model).to(config.training.device[0])
        elif config.model.model_type == "LBBDM":
            net = LatentBrownianBridgeModel(config.model).to(config.training.device[0])
        elif config.model.model_type == "GDDPM":
            net = GaussianDiffusionModel(config.model).to(config.training.device[0])
        elif config.model.model_type == "I2SB":
            net = I2SBModel(config.model).to(config.training.device[0])
        else:
            raise NotImplementedError(f"不支持的模型类型: {config.model.model_type}")
        # 特征缓存出处防护: 把当前配置的器官表喂给模型(模型侧只有 model 段, 读不到 data 段)
        try:
            organs = config.data.dataset_config.organs or []
            net._train_organ_map = [
                (int((o if isinstance(o, dict) else vars(o))['label']),
                 str((o if isinstance(o, dict) else vars(o))['path']))
                for o in organs]
        except AttributeError:
            net._train_organ_map = None
        net.apply(weights_init)
        return net

    def load_model_from_checkpoint(self):
        """加载检查点，修复 EMA 设备迁移问题并加载潜空间统计参数"""
        states = None
        model = self.net.module if (self.config.training.use_DDP and hasattr(self.net, 'module')) else self.net

        if self.config.model.only_load_latent_mean_std:
            if hasattr(self.config.model, 'model_load_path') and self.config.model.model_load_path is not None:
                states = torch.load(self.config.model.model_load_path, map_location='cpu')
                if self.config.model.normalize_latent and states is not None:
                    required_keys = ['ori_latent_mean', 'ori_latent_std', 'cond_latent_mean', 'cond_latent_std']
                    if all(key in states for key in required_keys):
                        model.ori_latent_mean = states['ori_latent_mean'].to(self.config.training.device[0])
                        model.ori_latent_std = states['ori_latent_std'].to(self.config.training.device[0])
                        model.cond_latent_mean = states['cond_latent_mean'].to(self.config.training.device[0])
                        model.cond_latent_std = states['cond_latent_std'].to(self.config.training.device[0])
                    else:
                        self.logger("警告：仅加载潜变量模式，但检查点缺少必要参数")
        else:
            # 1. 加载基础模型权重
            states = super().load_model_from_checkpoint()

            # 2. 加载任务 Embedding 权重
            if states is not None and hasattr(model, "task_embed"):
                if "task_embed.weight" in states:
                    model.task_embed.weight = nn.Parameter(
                        states["task_embed.weight"].to(self.config.training.device[0]))
                if "task_embed.bias" in states and model.task_embed.bias is not None:
                    model.task_embed.bias = nn.Parameter(states["task_embed.bias"].to(self.config.training.device[0]))

            # 3.  EMA 设备同步：修复 Resume 训练或测试时的多设备冲突
            if hasattr(self, 'ema') and self.ema is not None:
                device = self.config.training.device[0]
                self.logger(f" 检测到 EMA，正在将 shadow 权重同步至设备 {device}...")
                for name, shadow_param in self.ema.shadow.items():
                    self.ema.shadow[name] = shadow_param.to(device)
                self.logger(" EMA 设备同步完成")

        # 4. 加载/计算潜空间统计参数 (Mean/Std)
        if self.config.model.normalize_latent and not self.config.model.only_load_latent_mean_std:
            if states is not None:
                required_keys = ['ori_latent_mean', 'ori_latent_std', 'cond_latent_mean', 'cond_latent_std']
                if all(key in states for key in required_keys):
                    model.ori_latent_mean = states['ori_latent_mean'].to(self.config.training.device[0])
                    model.ori_latent_std = states['ori_latent_std'].to(self.config.training.device[0])
                    model.cond_latent_mean = states['cond_latent_mean'].to(self.config.training.device[0])
                    model.cond_latent_std = states['cond_latent_std'].to(self.config.training.device[0])
                else:
                    self.logger("警告：检查点缺少潜变量参数，自动重新计算")
                    if self.config.args.train:
                        self.get_latent_mean_std()
            else:
                if self.config.args.train:
                    self.get_latent_mean_std()

    def print_model_summary(self, net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        self.logger(f"Total Parameters: {total_num / 1e6:.2f}M")
        self.logger(f"Trainable Parameters: {trainable_num / 1e6:.2f}M")

    def initialize_optimizer_scheduler(self, net, config):
        optimizer = get_optimizer(config.model.BB.optimizer, net.get_parameters())
        # E1 动力学修复: config.training.lr_milestones/lr_factors 存在时用确定性阶梯 LambdaLR
        # (基线的 ReduceLROnPlateau 被逐 batch 训练 loss 噪声触发, ~3040 步意外减半)
        milestones = getattr(config.training, 'lr_milestones', None)
        factors = getattr(config.training, 'lr_factors', None)
        if milestones and factors:
            m, f = list(milestones), list(factors)
            assert len(f) == len(m) + 1, "lr_factors 需比 lr_milestones 多 1 个(首段因子)"

            def lr_lambda(s):
                for k in range(len(m)):
                    if s < m[k]:
                        return f[k]
                return f[-1]

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode='min',
                # verbose=True,
                threshold_mode='rel', **vars(config.model.BB.lr_scheduler)
            )
        # E4: PatchGAN 判别器独立优化器(恒定 lr, 占位调度器兼容循环的 scheduler.step 分派)
        optimizers, schedulers = [optimizer], [scheduler]
        if getattr(config.model.BB.params, 'use_patch_d', False):
            d_lr = getattr(config.model.BB.params, 'd_lr', 2e-4)
            d_opt = torch.optim.Adam(net.patch_d.parameters(), lr=d_lr, betas=(0.5, 0.999))
            d_sch = torch.optim.lr_scheduler.LambdaLR(d_opt, lambda e: 1.0)
            optimizers.append(d_opt)
            schedulers.append(d_sch)
        return optimizers, schedulers

    @torch.no_grad()
    def get_checkpoint_states(self, stage='epoch_end'):
        model_states, optimizer_scheduler_states = super().get_checkpoint_states()
        model = self.net.module if (self.config.training.use_DDP and hasattr(self.net, 'module')) else self.net

        if hasattr(model, "task_embed"):
            model_states['task_embed.weight'] = model.task_embed.weight

        if self.config.model.normalize_latent:
            model_states['ori_latent_mean'] = model.ori_latent_mean
            model_states['ori_latent_std'] = model.ori_latent_std
            model_states['cond_latent_mean'] = model.cond_latent_mean
            model_states['cond_latent_std'] = model.cond_latent_std

        return model_states, optimizer_scheduler_states

    def get_latent_mean_std(self):
        """计算数据集在潜空间的统计分布"""
        train_dataset, _, _ = get_dataset(self.config.data)
        train_loader = DataLoader(train_dataset, batch_size=self.config.data.train.batch_size,
                                  shuffle=True, num_workers=0, drop_last=True)

        total_ori_mean = total_cond_mean = None
        batch_count = 0
        self.logger("正在计算潜空间统计参数 (Mean)...")
        for batch in tqdm(train_loader, total=len(train_loader), smoothing=0.01):
            batch_count += 1
            (x, _), (x_cond, _) = batch
            x = x.to(self.config.training.device[0])
            x_cond = x_cond.to(self.config.training.device[0])

            x_latent = self.net.encode(x, cond=False, normalize=False)
            x_cond_latent = self.net.encode(x_cond, cond=True, normalize=False)

            mean_ori = x_latent.mean(dim=[0, 2, 3], keepdim=True)
            mean_cond = x_cond_latent.mean(dim=[0, 2, 3], keepdim=True)
            total_ori_mean = mean_ori if total_ori_mean is None else total_ori_mean + mean_ori
            total_cond_mean = mean_cond if total_cond_mean is None else total_cond_mean + mean_cond

        model = self.net.module if (self.config.training.use_DDP and hasattr(self.net, 'module')) else self.net
        model.ori_latent_mean = total_ori_mean / batch_count
        model.cond_latent_mean = total_cond_mean / batch_count

        total_ori_var = total_cond_var = None
        batch_count = 0
        self.logger("正在计算潜空间统计参数 (Std)...")
        for batch in tqdm(train_loader, total=len(train_loader), smoothing=0.01):
            batch_count += 1
            (x, _), (x_cond, _) = batch
            x = x.to(self.config.training.device[0])
            x_cond = x_cond.to(self.config.training.device[0])

            x_latent = self.net.encode(x, cond=False, normalize=False)
            x_cond_latent = self.net.encode(x_cond, cond=True, normalize=False)

            var_ori = ((x_latent - model.ori_latent_mean) ** 2).mean(dim=[0, 2, 3], keepdim=True)
            var_cond = ((x_cond_latent - model.cond_latent_mean) ** 2).mean(dim=[0, 2, 3], keepdim=True)
            total_ori_var = var_ori if total_ori_var is None else total_ori_var + var_ori
            total_cond_var = var_cond if total_cond_var is None else total_cond_var + var_cond

        model.ori_latent_std = torch.sqrt(total_ori_var / batch_count)
        model.cond_latent_std = torch.sqrt(total_cond_var / batch_count)

    def _d_phase_loss(self, net, stage):
        """E4 D 相位 (opt_idx=1): 只用 G 相位缓存, 不做 UNet 前向。
        先 zero_grad 清掉 G 反向沉积在 D 上的脏梯度; val/无缓存时返回可 backward 的常数。"""
        device = self.config.training.device[0]
        cache = getattr(net, '_d_cache', None)
        if stage != 'train' or cache is None or not getattr(net, 'use_patch_d', False):
            return torch.zeros(1, device=device, requires_grad=True)
        self.optimizer[1].zero_grad(set_to_none=True)  # 审计修复②: 清 G 相位脏梯度
        x0_hat_d, x0_real = cache
        d_real = net.patch_d(x0_real)
        d_fake = net.patch_d(x0_hat_d)
        l_d = 0.5 * (torch.nn.functional.relu(1.0 - d_real).mean()
                     + torch.nn.functional.relu(1.0 + d_fake).mean())
        with torch.no_grad():
            acc = ((d_real > 0).float().mean().item() + (d_fake < 0).float().mean().item()) / 2
        if self.is_main_process:
            self.writer.add_scalar('disc/loss_D', float(l_d.detach()), self.global_step)
            self.writer.add_scalar('disc/acc', acc, self.global_step)
        return l_d

    def loss_fn(self, net, batch, epoch, step, opt_idx=0, stage='train', write=True):
        # E4 审计修复①: 双优化器分派 —— D 相位只用缓存, 避免第二次 755M UNet 前向
        if opt_idx == 1:
            return self._d_phase_loss(net, stage)
        # G 相位开始前清 D 缓存(防止用到上一个 batch 的陈旧张量)
        if getattr(net, 'use_patch_d', False):
            net._d_cache = None
        # 支持多器官数据集（batch 含 organ_label）和单器官数据集（从 config 读）
        cn = None  # cond 文件名(特征缓存查表键用)
        if len(batch) == 3:
            (x, _), (x_cond, cn), organ_label = batch
            x = x.to(self.config.training.device[0])
            x_cond = x_cond.to(self.config.training.device[0])
            organ_label = organ_label.to(self.config.training.device[0])
        else:
            (x, _), (x_cond, _) = batch
            x = x.to(self.config.training.device[0])
            x_cond = x_cond.to(self.config.training.device[0])
            organ_label = torch.tensor(
                [getattr(self.config.data.dataset_config, 'organ_label', 0)],
                device=self.config.training.device[0]
            ).expand(x.shape[0])
        # 特征缓存键注入(DDP 时透传到 module): 仅任务0有效, 任务1输入为H&E自身口径不符
        _m = net.module if hasattr(net, 'module') else net

        # 任务配比: 按 global_step 决定(task_counter 会被每 50 步的 val 调用漂移,勿用)
        # task1_end_step>0: 该步之后全部分配给 task 0 (PI→H&E); 缺省/0 = 维持原 50/50 交替
        t1_end = int(getattr(self.config.training, 'task1_end_step', 0) or 0)
        if stage == 'train':
            if t1_end > 0 and step >= t1_end:
                current_task_id = 0
            else:
                current_task_id = (step // 4) % 2
        else:
            current_task_id = 0  # val/test 只监控被评估的任务
        self.task_counter += 1  # 仅保留给日志

        if current_task_id == 0:
            # 任务 A: Source -> Target
            x_input = x
            x_cond_input = x_cond
            task_name = "Source2Target"
            teacher_y = x
            teacher_context = x
            _m._feat_keys = ([(int(l), n) for l, n in zip(organ_label.tolist(), list(cn))]
                             if cn is not None else None)

            loss, additional_info = net(
                x_input,
                x_cond_input,
                current_task_id=current_task_id,
                teacher_y=teacher_y,
                teacher_context=teacher_context,
                organ_label=organ_label,
            )
        else:
            # 任务 B: Target 重建
            x_input = x
            x_cond_input = x
            task_name = "Target_recon"
            _m._feat_keys = None  # 任务1输入是 H&E 自身, 缓存口径不符
            loss, additional_info = net(
                x_input, x_cond_input,
                current_task_id=current_task_id,
                organ_label=organ_label,
            )

        if write and self.is_main_process:
            self.writer.add_scalar(f'loss/total_{stage}', loss, step)
            self.writer.add_scalar(f'loss/{task_name}_{stage}', loss, step)
            monitor_keys = ['recloss']
            # GRL 监控
            if 'disc_acc' in additional_info:
                self.writer.add_scalar(f'disc_acc/{task_name}_{stage}', additional_info['disc_acc'], step)
                self.writer.add_scalar(f'adv_loss/{task_name}_{stage}', additional_info.get('adv_loss', 0), step)
            for key in monitor_keys:
                if key in additional_info:
                    self.writer.add_scalar(f'{key}/{task_name}_{stage}', additional_info[key], step)

        return loss

    @torch.no_grad()
    def sample(self, net, batch, sample_path, stage='train'):
        sample_path = make_dir(os.path.join(sample_path, f'{stage}_sample'))
        source2target_path = make_dir(os.path.join(sample_path, 'Source2Target'))
        target_recon_path = make_dir(os.path.join(sample_path, 'Target_recon'))
        degraded_target_path = make_dir(os.path.join(sample_path, 'degraded_Target'))
        gt_path = make_dir(os.path.join(sample_path, 'ground_truth'))

        if len(batch) == 3:
            (x, x_name), (x_cond, x_cond_name), _ = batch
        else:
            (x, x_name), (x_cond, x_cond_name) = batch
        batch_size = x.shape[0] if x.shape[0] < 4 else 4
        x = x[0:batch_size].to(self.config.training.device[0])
        x_cond_source = x_cond[0:batch_size].to(self.config.training.device[0])
        grid_size = 4
        to_normal = getattr(self.config.data.dataset_config, 'to_normal', False)
        sample_task = getattr(self.config.testing, 'sample_task', 0)
        noise_level = getattr(self.config.testing, 'noise_level', 0.4)
        clip_denoised = getattr(self.config.testing, 'clip_denoised', True)

        # ===== 任务 1: Source → Target 采样 =====
        if sample_task in [0, 2]:
            sample_kwargs = {'clip_denoised': clip_denoised, 'task_id': 0}
            if self.is_latent_model:
                sample_kwargs['x_cond_img'] = x_cond_source
            else:
                sample_kwargs['y'] = x_cond_source

            source2target_sample = net.sample(**sample_kwargs).to('cpu')
            source2target_grid = get_image_grid(source2target_sample, grid_size, to_normal=to_normal)
            Image.fromarray(source2target_grid).save(os.path.join(source2target_path, 'Source2Target.png'))
            if stage != 'test':
                self.writer.add_image(f'{stage}/Source2Target', source2target_grid, self.global_step, dataformats='HWC')

        # ===== 任务 2: Target 重建采样 =====
        if sample_task in [1, 2]:
            degraded_target = x + noise_level * torch.randn_like(x)
            degraded_target = torch.clamp(degraded_target, -1., 1.)
            degraded_target_grid = get_image_grid(degraded_target.to('cpu'), grid_size, to_normal=to_normal)
            Image.fromarray(degraded_target_grid).save(os.path.join(degraded_target_path, 'degraded_Target.png'))

            sample_kwargs = {'clip_denoised': clip_denoised, 'task_id': 1}
            if self.is_latent_model:
                sample_kwargs['x_cond_img'] = degraded_target
            else:
                sample_kwargs['y'] = degraded_target

            target_recon_sample = net.sample(**sample_kwargs).to('cpu')
            target_recon_grid = get_image_grid(target_recon_sample, grid_size, to_normal=to_normal)
            Image.fromarray(target_recon_grid).save(os.path.join(target_recon_path, 'Target_recon.png'))
            if stage != 'test':
                self.writer.add_image(f'{stage}/degraded_Target', degraded_target_grid, self.global_step, dataformats='HWC')
                self.writer.add_image(f'{stage}/Target_recon', target_recon_grid, self.global_step, dataformats='HWC')

        # 保存真值
        gt_target_grid = get_image_grid(x.to('cpu'), grid_size, to_normal=to_normal)
        gt_source_grid = get_image_grid(x_cond_source.to('cpu'), grid_size, to_normal=to_normal)
        Image.fromarray(gt_source_grid).save(os.path.join(gt_path, 'GT_Source.png'))
        Image.fromarray(gt_target_grid).save(os.path.join(gt_path, 'GT_Target.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}/GT_Source', gt_source_grid, self.global_step, dataformats='HWC')
            self.writer.add_image(f'{stage}/GT_Target', gt_target_grid, self.global_step, dataformats='HWC')

    @torch.no_grad()
    def sample_to_eval(self, net, test_loader, sample_path):
        """测试阶段采样：修正文件夹命名逻辑与过时注释"""
        ema = getattr(self, 'ema', None)
        use_ema_test = getattr(self.config.testing, 'use_ema', False)
        active_ema = use_ema_test and (ema is not None)
        
        if active_ema:
            self.ema.apply_shadow(net)
            if self.is_main_process:
                self.logger(" [指标评测] 正在激活 EMA 平滑权重推理...")


        source2target_path = make_dir(os.path.join(sample_path, 'Source2Target'))
        target_recon_path = make_dir(os.path.join(sample_path, 'Target_recon'))
        condition_path = make_dir(os.path.join(sample_path, 'condition'))
        gt_path = make_dir(os.path.join(sample_path, 'ground_truth'))

        pbar = tqdm(test_loader, total=len(test_loader), smoothing=0.01)
        batch_size = self.config.data.test.batch_size
        to_normal = self.config.data.dataset_config.to_normal
        noise_level = getattr(self.config.testing, 'noise_level', 0.4)
        clip_denoised = getattr(self.config.testing, 'clip_denoised', True)

        for test_batch in pbar:
            # x = Target (目标), x_cond = Source (输入条件)
            if len(test_batch) == 3:
                (x, x_name), (x_cond, x_cond_name), _ = test_batch
            else:
                (x, x_name), (x_cond, x_cond_name) = test_batch
            x = x.to(self.config.training.device[0])
            x_cond = x_cond.to(self.config.training.device[0])

            # ===== 任务 1: Source → Target 推理 =====
            source2target_sample = net.sample(x_cond, clip_denoised=clip_denoised, task_id=0).to('cpu')
            for i in range(batch_size):
                # 保存输入 Source
                save_single_image(x_cond[i].cpu(), condition_path, f'{x_cond_name[i]}.png', to_normal=to_normal)
                #  仅在此处保存一次真值 GT，避免 I/O 冗余
                save_single_image(x[i].cpu(), gt_path, f'{x_name[i]}.png', to_normal=to_normal)
                # 保存生成结果
                save_single_image(source2target_sample[i], source2target_path, f'{x_name[i]}.png', to_normal=to_normal)

            # ===== 任务 2: Target 自重建推理 (鲁棒性验证) =====
            degraded_target = x + noise_level * torch.randn_like(x) 
            degraded_target = torch.clamp(degraded_target, -1., 1.)
            target_recon_sample = net.sample(degraded_target, clip_denoised=clip_denoised, task_id=1).to('cpu')

            for i in range(batch_size):
                # 保存降质后的条件输入
                save_single_image(degraded_target[i].cpu(), target_recon_path, f'{x_name[i]}_degraded.png', to_normal=to_normal)
                # 保存重建结果
                save_single_image(target_recon_sample[i], target_recon_path, f'{x_name[i]}_recon.png', to_normal=to_normal)

        if active_ema:
            self.ema.restore(net)
            if self.is_main_process:
                self.logger("✅ [指标评测] 推理完成，模型状态已恢复")