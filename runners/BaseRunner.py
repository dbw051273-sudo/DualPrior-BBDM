import argparse
import datetime
import pdb
import time

import yaml
import os
import traceback

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

from abc import ABC, abstractmethod
from tqdm.autonotebook import tqdm

# from evaluation.FID import calc_FID
# from evaluation.LPIPS import calc_LPIPS
from runners.base.EMA import EMA
from runners.base.SWAD import SWAD
from runners.utils import make_save_dirs, make_dir, get_dataset, remove_file


class BaseRunner(ABC):
    def __init__(self, config):
        self.net = None  # Neural Network
        self.optimizer = None  # optimizer
        self.scheduler = None  # scheduler
        self.config = config  # config from configuration file
        self.use_amp = getattr(self.config.training, 'use_amp', False)
        self.is_main_process = True if (not self.config.training.use_DDP) or self.config.training.local_rank == 0 else False
        # set training params
        self.global_epoch = 0  # global epoch
        if config.args.sample_at_start:
            self.global_step = -1  # global step
        else:
            self.global_step = 0

        self.GAN_buffer = {}  # GAN buffer for Generative Adversarial Network
        self.topk_checkpoints = {}  # Top K checkpoints

        # set log and save destination
        self.config.result = argparse.Namespace()
        self.config.result.result_path, \
        self.config.result.image_path, \
        self.config.result.ckpt_path, \
        self.config.result.log_path, \
        self.config.result.sample_path, \
        self.config.result.sample_to_eval_path = make_save_dirs(self.config.args,
                                                                prefix=self.config.data.dataset_name,
                                                                suffix=self.config.model.model_name)

        self.logger("save training results to " + self.config.result.result_path)

        self.save_config()  # save configuration file
        self.writer = SummaryWriter(self.config.result.log_path)  # initialize SummaryWriter

        # initialize model
        self.net, self.optimizer, self.scheduler = self.initialize_model_optimizer_scheduler(self.config)

        self.print_model_summary(self.net)

        # initialize EMA
        self.use_ema = False if not self.config.model.__contains__('EMA') else self.config.model.EMA.use_ema
        if self.use_ema:
            self.ema = EMA(self.config.model.EMA.ema_decay)
            self.update_ema_interval = self.config.model.EMA.update_ema_interval
            self.start_ema_step = self.config.model.EMA.start_ema_step
            self.apply_ema_during_training = getattr(self.config.model.EMA, 'apply_during_training', True)
            self.ema.register(self.net)

        # initialize SWAD (Stochastic Weight Averaging Densely, NeurIPS 2021)
        self.use_swad = getattr(self.config.training, 'use_swad', False)
        if self.use_swad:
            swad_cfg = getattr(self.config.training, 'swad', {})
            start_step = getattr(swad_cfg, 'start_step', 30000)
            update_interval = getattr(swad_cfg, 'update_interval', 100)
            loss_threshold = getattr(swad_cfg, 'loss_threshold', 0.045)
            shadow_device = getattr(swad_cfg, 'shadow_device', 'cpu')
            self.swad = SWAD(start_step=start_step,
                             update_interval=update_interval,
                             loss_threshold=loss_threshold,
                             shadow_device=shadow_device)
            self.logger(f"===> [SWAD 启用] start_step={start_step}, interval={update_interval}, loss_thresh={loss_threshold}, device={shadow_device}")

        # load model from checkpoint
        self.load_model_from_checkpoint()

        # initialize DDP
        if self.config.training.use_DDP:
            self.net = DDP(self.net, device_ids=[self.config.training.local_rank], output_device=self.config.training.local_rank)
        else:
            self.net = self.net.to(self.config.training.device[0])
        # self.ema.reset_device(self.net)

    # print msg
    def logger(self, msg, **kwargs):
        if self.is_main_process:
            print(msg, **kwargs)

    # save configuration file
    def save_config(self):
        if self.is_main_process:
            save_path = os.path.join(self.config.result.ckpt_path, 'config.yaml')
            save_config = self.config
            with open(save_path, 'w') as f:
                yaml.dump(save_config, f)

    def initialize_model_optimizer_scheduler(self, config, is_test=False):
        """
        get model, optimizer, scheduler
        :param args: args
        :param config: config
        :param is_test: is_test
        :return: net: Neural Network, nn.Module;
                 optimizer: a list of optimizers;
                 scheduler: a list of schedulers or None;
        """
        net = self.initialize_model(config)
        optimizer, scheduler = None, None
        if not is_test:
            optimizer, scheduler = self.initialize_optimizer_scheduler(net, config)
        return net, optimizer, scheduler

    # load model, EMA, optimizer, scheduler from checkpoint
    def load_model_from_checkpoint(self):
        model_states = None
        if self.config.model.__contains__('model_load_path') and self.config.model.model_load_path is not None:
            self.logger(f"load model {self.config.model.model_name} from {self.config.model.model_load_path}")
            model_states = torch.load(self.config.model.model_load_path, map_location='cpu')
            # self.global_epoch = model_states.get('epoch',108)

            self.global_epoch = model_states['epoch']
            self.global_step = model_states['step']

            # load model
            self.net.load_state_dict(model_states['model'],strict=False)

            # load ema
            if self.use_ema:
                self.ema.shadow = model_states['ema']
                self.ema.reset_device(self.net)

            # load swad
            if getattr(self, 'use_swad', False) and 'swad' in model_states and model_states['swad'] is not None:
                self.swad.shadow = model_states['swad']
                self.swad.n_averaged = model_states.get('swad_n', 0)
                self.swad.actual_start_step = model_states.get('swad_start_step', None)
                self.swad.active = self.swad.n_averaged > 0
                self.logger(f"[SWAD 恢复断点] 成功恢复已平均 {self.swad.n_averaged} 个权重的 SWAD 状态")

            # load optimizer and scheduler
            if self.config.args.train:
                if self.config.model.__contains__('optim_sche_load_path') and self.config.model.optim_sche_load_path is not None:
                    optimizer_scheduler_states = torch.load(self.config.model.optim_sche_load_path, map_location='cpu')
                    for i in range(len(self.optimizer)):
                        self.optimizer[i].load_state_dict(optimizer_scheduler_states['optimizer'][i])

                    if self.scheduler is not None:
                        for i in range(len(self.optimizer)):
                            self.scheduler[i].load_state_dict(optimizer_scheduler_states['scheduler'][i])
        return model_states

    def get_checkpoint_states(self, stage='epoch_end'):
        optimizer_state = []
        for i in range(len(self.optimizer)):
            optimizer_state.append(self.optimizer[i].state_dict())

        scheduler_state = []
        for i in range(len(self.scheduler)):
            scheduler_state.append(self.scheduler[i].state_dict())

        optimizer_scheduler_states = {
            'optimizer': optimizer_state,
            'scheduler': scheduler_state
        }

        model_states = {
            'step': self.global_step,
        }

        if self.config.training.use_DDP:
            model_states['model'] = self.net.module.state_dict()
        else:
            model_states['model'] = self.net.state_dict()

        if stage == 'exception':
            model_states['epoch'] = self.global_epoch
        else:
            model_states['epoch'] = self.global_epoch + 1

        if self.use_ema:
            model_states['ema'] = self.ema.shadow

        if getattr(self, 'use_swad', False) and self.swad.n_averaged > 0:
            model_states['swad'] = self.swad.shadow
            model_states['swad_n'] = self.swad.n_averaged
            model_states['swad_start_step'] = self.swad.actual_start_step

        return model_states, optimizer_scheduler_states

    # SWAD part (Stochastic Weight Averaging Densely, NeurIPS 2021)
    def step_swad(self, current_loss=0.0):
        if not getattr(self, 'use_swad', False):
            return
        model = self.net.module if (self.config.training.use_DDP and hasattr(self.net, 'module')) else self.net
        updated = self.swad.update(model, step=self.global_step, loss=current_loss)
        if updated and self.is_main_process and (self.swad.n_averaged == 1 or self.swad.n_averaged % 100 == 0):
            self.logger(f"[SWAD 密集平均] Step {self.global_step} | 累计平均 {self.swad.n_averaged} 个平坦低谷模型权重 | Loss: {current_loss:.4f}")

    # EMA part
    def step_ema(self):
        with_decay = False if self.global_step < self.start_ema_step else True
        if self.config.training.use_DDP:
            self.ema.update(self.net.module, with_decay=with_decay)
        else:
            self.ema.update(self.net, with_decay=with_decay)

    def apply_ema(self):
        # 训练期换装 EMA 需瞬时 +影子大小显存(755M≈3GB), 16GB 卡默认关(config EMA.apply_during_training)
        if self.use_ema and getattr(self, 'apply_ema_during_training', True):
            if self.config.training.use_DDP:
                self.ema.apply_shadow(self.net.module)
            else:
                self.ema.apply_shadow(self.net)

    # def restore_ema(self):
    #     if self.use_ema:
    #         if self.config.training.use_DDP:
    #             self.ema.restore(self.net.module)
    #         else:
    #             self.ema.restore(self.net)
    def restore_ema(self):
        if self.use_ema:
            try:
                if self.config.training.use_DDP:
                    self.ema.restore(self.net.module)
                else:
                    self.ema.restore(self.net)
            except AssertionError:
                pass

    # Evaluation and sample part
    @torch.no_grad()
    def validation_step(self, val_batch, epoch, step):
        self.apply_ema()
        self.net.eval()
        loss = self.loss_fn(net=self.net,
                            batch=val_batch,
                            epoch=epoch,
                            step=step,
                            opt_idx=0,
                            stage='val_step')
        if len(self.optimizer) > 1:
            loss = self.loss_fn(net=self.net,
                                batch=val_batch,
                                epoch=epoch,
                                step=step,
                                opt_idx=1,
                                stage='val_step')
        self.restore_ema()

    @torch.no_grad()
    def validation_epoch(self, val_loader, epoch):
        self.apply_ema()
        self.net.eval()

        pbar = tqdm(val_loader, total=len(val_loader), smoothing=0.01, disable=not self.is_main_process)
        step = 0
        loss_sum = 0.
        dloss_sum = 0.
        for val_batch in pbar:
            loss = self.loss_fn(net=self.net,
                                batch=val_batch,
                                epoch=epoch,
                                step=step,
                                opt_idx=0,
                                stage='val',
                                write=False)
            loss_sum += loss
            if len(self.optimizer) > 1:
                loss = self.loss_fn(net=self.net,
                                    batch=val_batch,
                                    epoch=epoch,
                                    step=step,
                                    opt_idx=1,
                                    stage='val',
                                    write=False)
                dloss_sum += loss
            step += 1
        average_loss = loss_sum / step
        if self.is_main_process:
            self.writer.add_scalar(f'val_epoch/loss', average_loss, epoch)
            if len(self.optimizer) > 1:
                average_dloss = dloss_sum / step
                self.writer.add_scalar(f'val_dloss_epoch/loss', average_dloss, epoch)
        self.restore_ema()
        return average_loss

    @torch.no_grad()
    def sample_step(self, train_batch, val_batch):
        self.apply_ema()
        self.net.eval()
        sample_path = make_dir(os.path.join(self.config.result.image_path, str(self.global_step)))
        if self.config.training.use_DDP:
            self.sample(self.net.module, train_batch, sample_path, stage='train')
            self.sample(self.net.module, val_batch, sample_path, stage='val')
        else:
            self.sample(self.net, train_batch, sample_path, stage='train')
            self.sample(self.net, val_batch, sample_path, stage='val')
        self.restore_ema()

    # abstract methods
    @abstractmethod
    def print_model_summary(self, net):
        pass

    @abstractmethod
    def initialize_model(self, config):
        """
        initialize model
        :param config: config
        :return: nn.Module
        """
        pass

    @abstractmethod
    def initialize_optimizer_scheduler(self, net, config):
        """
        initialize optimizer and scheduler
        :param net: nn.Module
        :param config: config
        :return: a list of optimizers; a list of schedulers
        """
        pass

    @abstractmethod
    def loss_fn(self, net, batch, epoch, step, opt_idx=0, stage='train', write=True):
        """
        loss function
        :param net: nn.Module
        :param batch: batch
        :param epoch: global epoch
        :param step: global step
        :param opt_idx: optimizer index, default is 0; set it to 1 for GAN discriminator
        :param stage: train, val, test
        :param write: write loss information to SummaryWriter
        :return: a scalar of loss
        """
        pass

    @abstractmethod
    def sample(self, net, batch, sample_path, stage='train'):
        """
        sample a single batch
        :param net: nn.Module
        :param batch: batch
        :param sample_path: path to save samples
        :param stage: train, val, test
        :return:
        """
        pass

    @abstractmethod
    def sample_to_eval(self, net, test_loader, sample_path):
        """
        sample among the test dataset to calculate evaluation metrics
        :param net: nn.Module
        :param test_loader: test dataloader
        :param sample_path: path to save samples
        :return:
        """
        pass

    def on_save_checkpoint(self, net, train_loader, val_loader, epoch, step):
        """
        additional operations whilst saving checkpoint
        :param net: nn.Module
        :param train_loader: train data loader
        :param val_loader: val data loader
        :param epoch: epoch
        :param step: step
        :return:
        """
        pass

    def train(self):
        self.logger(self.__class__.__name__)

        train_dataset, val_dataset, test_dataset = get_dataset(self.config.data)
        train_sampler = None
        val_sampler = None
        test_sampler = None
        if self.config.training.use_DDP:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
            train_loader = DataLoader(train_dataset,
                                      batch_size=self.config.data.train.batch_size,
                                      num_workers=8,
                                      drop_last=True,
                                      sampler=train_sampler)
            val_loader = DataLoader(val_dataset,
                                    batch_size=self.config.data.val.batch_size,
                                    num_workers=8,
                                    drop_last=True,
                                    sampler=val_sampler)
            test_loader = DataLoader(test_dataset,
                                     batch_size=self.config.data.test.batch_size,
                                     num_workers=8,
                                     drop_last=True,
                                     sampler=test_sampler)
        else:
            _nw = getattr(self.config.training, 'num_workers', 8)
            train_loader = DataLoader(train_dataset,
                                      batch_size=self.config.data.train.batch_size,
                                      shuffle=self.config.data.train.shuffle,
                                      num_workers=_nw,
                                      drop_last=True)
            val_loader = DataLoader(val_dataset,
                                    batch_size=self.config.data.val.batch_size,
                                    shuffle=self.config.data.val.shuffle,
                                    num_workers=_nw,
                                    drop_last=True)
            test_loader = DataLoader(test_dataset,
                                     batch_size=self.config.data.test.batch_size,
                                     shuffle=False,
                                     num_workers=_nw,
                                     drop_last=True)

        epoch_length = len(train_loader)
        start_epoch = self.global_epoch
        self.logger(f"start training {self.config.model.model_name} on {self.config.data.dataset_name}, {len(train_loader)} iters per epoch")

        try:
            accumulate_grad_batches = self.config.training.accumulate_grad_batches
            for epoch in range(start_epoch, self.config.training.n_epochs):
                if self.global_step >= self.config.training.n_steps:  # >=: 恰好停在 n_steps 时原 > 判断永假, 会空转 epoch 反复写盘
                    break

                if self.config.training.use_DDP:
                    train_sampler.set_epoch(epoch)
                    val_sampler.set_epoch(epoch)

                pbar = tqdm(train_loader, total=len(train_loader), smoothing=0.01, disable=not self.is_main_process)
                self.global_epoch = epoch
                start_time = time.time()
                for train_batch in pbar:
                    if self.global_step >= self.config.training.n_steps:
                        break
                    self.global_step += 1
                    self.net.train()

                    losses = []
                    for i in range(len(self.optimizer)):
                        if self.use_amp:
                            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                                loss = self.loss_fn(net=self.net,
                                                    batch=train_batch,
                                                    epoch=epoch,
                                                    step=self.global_step,
                                                    opt_idx=i,
                                                    stage='train')
                        else:
                            loss = self.loss_fn(net=self.net,
                                                batch=train_batch,
                                                epoch=epoch,
                                                step=self.global_step,
                                                opt_idx=i,
                                                stage='train')

                        loss.backward()
                        if self.global_step % accumulate_grad_batches == 0:
                            _clip_norm = getattr(self.config.training, 'grad_clip_norm', None)
                            real_grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(),
                                                                            max_norm=float(_clip_norm) if _clip_norm is not None else float('inf'))

                            # ✅ 打印真实梯度 (建议只在主进程打印，防止刷屏)
                            if self.is_main_process:
                                # 格式化打印，保留4位小数
                                print(
                                    f"[真实梯度监测] Step {self.global_step} | Real Grad Norm: {real_grad_norm.item():.4f}")

                            # ⚠️ [高能预警] 唯一的保险措施 (建议保留注释，如果你想彻底裸奔就删掉)
                            # 如果真的出现 NaN (梯度炸没了)，这步 update 会直接毁掉模型。
                            # 如果你想观察什么时候炸，可以不加下面这个判断，直接让它炸。
                            if torch.isnan(real_grad_norm) or torch.isinf(real_grad_norm):
                                print(f"[危险] 检测到梯度爆炸 (NaN/Inf)！模型参数即将被污染！")

                            self.optimizer[i].step()
                            self.optimizer[i].zero_grad()
                            if self.scheduler is not None:
                                sch = self.scheduler[i]
                                if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                    sch.step(loss)
                                else:
                                    sch.step()  # LambdaLR 等: 喂 loss 会把 last_epoch 设成浮点,毁掉调度
                                if self.is_main_process:
                                    try:
                                        self.writer.add_scalar(f'train/lr_{i}', sch.get_last_lr()[0], self.global_step)
                                    except Exception:
                                        pass
                        if self.config.training.use_DDP:
                            dist.reduce(loss, dst=0, op=dist.ReduceOp.SUM)
                        losses.append(loss.detach().mean())

                    if self.use_ema and self.global_step % (self.update_ema_interval*accumulate_grad_batches) == 0:
                        self.step_ema()

                    if getattr(self, 'use_swad', False) and self.global_step % (self.swad.update_interval * accumulate_grad_batches) == 0:
                        loss_val = losses[0].item() if (len(losses) > 0 and hasattr(losses[0], 'item')) else 0.0
                        self.step_swad(current_loss=loss_val)

                    if len(self.optimizer) > 1:
                        pbar.set_description(
                            (
                                f'Epoch: [{epoch + 1} / {self.config.training.n_epochs}] '
                                f'iter: {self.global_step} loss-1: {losses[0]:.4f} loss-2: {losses[1]:.4f}'
                            )
                        )
                    else:
                        pbar.set_description(
                            (
                                f'Epoch: [{epoch + 1} / {self.config.training.n_epochs}] '
                                f'iter: {self.global_step} loss: {losses[0]:.4f}'
                            )
                        )

                    with torch.no_grad():
                        if self.global_step % 50 == 0:
                            val_batch = next(iter(val_loader))
                            self.validation_step(val_batch=val_batch, epoch=epoch, step=self.global_step)

                        _sample_freq = getattr(self.config.training, 'sample_freq', None)
                        if _sample_freq is not None:
                            _should_sample = self.global_step % int(_sample_freq) == 0
                        else:
                            _should_sample = self.global_step % int(self.config.training.sample_interval * epoch_length) == 0

                        if _should_sample:
                            if self.is_main_process:
                                val_batch = next(iter(val_loader))
                                self.sample_step(val_batch=val_batch, train_batch=train_batch)
                                torch.cuda.empty_cache()

                    # ===== 步级 checkpoint（save_freq，含 EMA 与 SWAD 权重）=====
                    _save_freq = getattr(self.config.training, 'save_freq', None)
                    if _save_freq is not None and self.is_main_process and self.global_step % int(_save_freq) == 0:
                        with torch.no_grad():
                            _ckpt = os.path.join(self.config.result.ckpt_path, f'model_step_{self.global_step}.pth')
                            model_states, _ = self.get_checkpoint_states(stage='epoch_end')
                            _ckpt_data = {'model': model_states['model'],
                                          'ema': model_states.get('ema'),
                                          'step': self.global_step}
                            if 'swad' in model_states:
                                _ckpt_data['swad'] = model_states['swad']
                                _ckpt_data['swad_n'] = model_states['swad_n']
                                _ckpt_data['swad_start_step'] = model_states['swad_start_step']
                            torch.save(_ckpt_data, _ckpt)
                            self.logger(f"saved step checkpoint: model_step_{self.global_step}.pth")

                end_time = time.time()
                elapsed_rounded = int(round((end_time-start_time)))
                self.logger("training time: " + str(datetime.timedelta(seconds=elapsed_rounded)))

                # validation
                _skip_val = getattr(self.config.training, 'skip_final_validation', False)
                if not _skip_val and ((epoch + 1) % self.config.training.validation_interval == 0 or (
                        epoch + 1) == self.config.training.n_epochs):
                    # if self.is_main_process == 0:
                    with torch.no_grad():
                        self.logger("validating epoch...")
                        average_loss = self.validation_epoch(val_loader, epoch)
                        torch.cuda.empty_cache()
                        self.logger("validating epoch success")

                # save checkpoint
                if (epoch + 1) % self.config.training.save_interval == 0 or \
                        (epoch + 1) == self.config.training.n_epochs or \
                        self.global_step > self.config.training.n_steps:
                    if self.is_main_process:
                        with torch.no_grad():
                            self.logger("saving latest checkpoint...")
                            self.on_save_checkpoint(self.net, train_loader, val_loader, epoch, self.global_step)
                            model_states, optimizer_scheduler_states = self.get_checkpoint_states(stage='epoch_end')

                            # save latest checkpoint
                            temp = 0
                            while temp < epoch + 1:
                                remove_file(os.path.join(self.config.result.ckpt_path, f'latest_model_{temp}.pth'))
                                remove_file(
                                    os.path.join(self.config.result.ckpt_path, f'latest_optim_sche_{temp}.pth'))
                                temp += 1
                            torch.save(model_states,
                                       os.path.join(self.config.result.ckpt_path,
                                                    f'latest_model_{epoch + 1}.pth'))
                            torch.save(optimizer_scheduler_states,
                                       os.path.join(self.config.result.ckpt_path,
                                                    f'latest_optim_sche_{epoch + 1}.pth'))
                            torch.save(model_states,
                                       os.path.join(self.config.result.ckpt_path,
                                                    f'last_model.pth'))
                            torch.save(optimizer_scheduler_states,
                                       os.path.join(self.config.result.ckpt_path,
                                                    f'last_optim_sche.pth'))

                            # save top_k checkpoints
                            model_ckpt_name = f'top_model_epoch_{epoch + 1}.pth'
                            optim_sche_ckpt_name = f'top_optim_sche_epoch_{epoch + 1}.pth'

                            if self.config.args.save_top:
                                print("save top model start...")
                                top_key = 'top'
                                if top_key not in self.topk_checkpoints:
                                    print('top key not in topk_checkpoints')
                                    self.topk_checkpoints[top_key] = {"loss": average_loss,
                                                                      'model_ckpt_name': model_ckpt_name,
                                                                      'optim_sche_ckpt_name': optim_sche_ckpt_name}

                                    print(f"saving top checkpoint: average_loss={average_loss} epoch={epoch + 1}")
                                    torch.save(model_states,
                                               os.path.join(self.config.result.ckpt_path, model_ckpt_name))
                                    torch.save(optimizer_scheduler_states,
                                               os.path.join(self.config.result.ckpt_path, optim_sche_ckpt_name))
                                else:
                                    if average_loss < self.topk_checkpoints[top_key]["loss"]:
                                        print("remove " + self.topk_checkpoints[top_key]["model_ckpt_name"])
                                        remove_file(os.path.join(self.config.result.ckpt_path,
                                                                 self.topk_checkpoints[top_key]['model_ckpt_name']))
                                        remove_file(os.path.join(self.config.result.ckpt_path,
                                                                 self.topk_checkpoints[top_key]['optim_sche_ckpt_name']))

                                        print(
                                            f"saving top checkpoint: average_loss={average_loss} epoch={epoch + 1}")

                                        self.topk_checkpoints[top_key] = {"loss": average_loss,
                                                                          'model_ckpt_name': model_ckpt_name,
                                                                          'optim_sche_ckpt_name': optim_sche_ckpt_name}

                                        torch.save(model_states,
                                                   os.path.join(self.config.result.ckpt_path, model_ckpt_name))
                                        torch.save(optimizer_scheduler_states,
                                                   os.path.join(self.config.result.ckpt_path, optim_sche_ckpt_name))
                if self.config.training.use_DDP:
                    dist.barrier()
        except BaseException as e:
            if self.is_main_process:
                print("exception save model start....")
                print(self.__class__.__name__)
                model_states, optimizer_scheduler_states = self.get_checkpoint_states(stage='exception')
                torch.save(model_states,
                           os.path.join(self.config.result.ckpt_path, f'last_model.pth'))
                torch.save(optimizer_scheduler_states,
                           os.path.join(self.config.result.ckpt_path, f'last_optim_sche.pth'))

                print("exception save model success!")

            print('str(Exception):\t', str(Exception))
            print('str(e):\t\t', str(e))
            print('repr(e):\t', repr(e))
            print('traceback.print_exc():')
            traceback.print_exc()
            print('traceback.format_exc():\n%s' % traceback.format_exc())

    @torch.no_grad()
    def test(self):
        train_dataset, val_dataset, test_dataset = get_dataset(self.config.data)
        if test_dataset is None:
            test_dataset = val_dataset
        # test_dataset = val_dataset
        if self.config.training.use_DDP:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
            test_loader = DataLoader(test_dataset,
                                     batch_size=self.config.data.test.batch_size,
                                     shuffle=False,
                                     num_workers=1,
                                     drop_last=True,
                                     sampler=test_sampler)
        else:
            test_loader = DataLoader(test_dataset,
                                     batch_size=self.config.data.test.batch_size,
                                     shuffle=False,
                                     num_workers=1,
                                     drop_last=True)

        if self.use_ema:
            self.apply_ema()

        self.net.eval()
        if self.config.args.sample_to_eval:
            sample_path = self.config.result.sample_to_eval_path
            if self.config.training.use_DDP:
                self.sample_to_eval(self.net.module, test_loader, sample_path)
            else:
                self.sample_to_eval(self.net, test_loader, sample_path)
        else:
            test_iter = iter(test_loader)
            for i in tqdm(range(1), initial=0, dynamic_ncols=True, smoothing=0.01):
                test_batch = next(test_iter)
                sample_path = os.path.join(self.config.result.sample_path, str(i))
                if self.config.training.use_DDP:
                    self.sample(self.net.module, test_batch, sample_path, stage='test')
                else:
                    self.sample(self.net, test_batch, sample_path, stage='test')
