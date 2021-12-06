import datetime 
import os 
import time 
from collections import defaultdict, deque
import functools

from copy import deepcopy

from torch.cuda import synchronize
from utils.ema import is_parallel
from thop import profile 

import numpy as np 
import torch
from torch import is_distributed
from torch.nn.modules.loss import L1Loss 
from torch.nn.parallel import DistributedDataParallel as DDP 
from torch.utils.tensorboard import SummaryWriter
from loguru import logger 
import shutil 

from ..utils import get_world_size, get_rank, get_local_rank, setup_logger, all_reduce_norm, synchronize
from ..data import DataPrefetcher
from ..utils import ModelEMA

class AverageMeter:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """
    def __init__(self, window_size=50):
        self._deque = deque(maxlen=window_size)
        self._total = 0. 
        self._count = 0

    def update(self, value):
        self._deque.append(value)
        self._count += 1
        self._total += value
    
    @property
    def median(self):
        d = np.array(list(self._deque))
        return np.median(d)
    
    @property
    def avg(self):
        # if deque is empty, nan will be returned.
        d = np.array(list(self._deque))
        return d.mean()
    
    @property
    def global_avg(self):
        return self._total / max(self._count, 1e-5)
    
    @property
    def latest(self):
        return self._deque[-1] if len(self._deque) > 0 else None 
    
    @property
    def total(self):
        return self._total
    
    @property
    def reset(self):
        self._deque.clear()
        self._total=0.0
        self._count=0
    
    @property
    def clear(self):
        self._deque.clear()

class MeterBuffer(defaultdict):
    '''
    compute and store the average and current value.
    '''
    def __init__(self, window_size=20):
        factory = functools.partial(AverageMeter, window_size=window_size)
        super().__init__(factory)
    
    def reset(self):
        for v in self.values:
            v.reset()
    
    def get_filtered_meter(self, filter_key='time'):
        return {k:v for k, v in self.items() if filter_key in k}
    
    def update(self, values=None, **kwargs):
        if values is None:
            values = {}
        values.update(kwargs)
        for k, v in values.items():
            if isinstance(v, torch.Tensor):
                v = v.detach()
            self[k].update(v)

    def clear_meters(self):
        for v in self.values():
            v.clear()

def get_model_info(model, tsize):
    stride = 64 
    img = torch.zeros((1, 3, stride, stride), device=next(model.parameters()).device)
    flops, parameters = profile(deepcopy(model), inputs=(img,), verbose=False)
    parameters /= 1e6 
    flops /= 1e9 
    flops *= 2 * tsize[0] * tsize[1] / stride ** 2 # GFLOPS
    info = "Parameters: {:.2f}M, GFLOPS: {:.2f}".format(parameters, flops)
    return info

def get_total_and_free_memory_in_Mb(cuda_device):
    devices_info_str = os.popen("nvidia-smi --query-gpu=memory.total,memory.used --format=csv,nounits,noheader")
    devices_info = devices_info_str.read().strip().split('\n')
    total, used = devices_info[int(cuda_device)].split(',')
    return int(total), int(used)

def occupy_mem(cuda_device, mem_ratio=0.9):
    """
    pre-allocate gpu memory for training to avoid memory Fragmentation.
    """
    total, used = get_total_and_free_memory_in_Mb(cuda_device)
    max_mem = int(total * mem_ratio)
    block_mem = max_mem - used
    x = torch.cuda.FloatTensor(256,1024,block_mem)
    del x 
    time.sleep(5)


def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (MB).
    """
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / (1024 * 1024)

class Trainer:
    def __init__(self, exp, args):
        '''
        init function only defines some basic attributes, other attributes like model, optimizer are built in before_train methods.
        '''
        self.exp = exp 
        self.args = args 

        # training related attributes
        self.max_epoch = exp.max_epoch 
        self.amp_training = args.fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.device = 'cuda:{}'.format(self.local_rank)
        self.use_model_ema = exp.ema 
        
        # data/dataloader related attributes
        self.data_type = torch.float16 if args.fp16 else torch.float32
        self.input_size = exp.input_size
        self.best_ap = 0

        # metric record 
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.filename = os.path.join(exp.output_dir, args.experiment_name)

        if self.rank==0:
            os.makedirs(self.filename, exist_ok=True)
        
        setup_logger(self.filename, distributed_rank=self.rank, filename='train_log.txt', mode='a')

    def train(self):
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception:
            raise 
        finally:
            self.after_train()
        
    def before_train(self):
        logger.info(f"args: {self.args}")
        logger.info(f"exp values:\n{self.exp}")

        # model related initialization
        torch.cuda.set_device(self.local_rank)
        model = self.exp.get_model()
        logger.info(f'Model summary: {get_model_info(model, self.exp.test_size)}')
        model.to(self.device)

        # solver related initialization
        self.optimizer = self.exp.get_optimizer(self.args.batch_size)

        # value of epoch will be set in `resume_train`
        model = self.resume_train(model)

        # data related initialization
        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs
        self.train_loader = self.exp.get_data_loader(batch_size=self.args.batch_size, is_distributed=self.is_distributed, \
                                                     no_aug=self.no_aug, cache_img=self.cache_img)

        logger.info("Init prefetcher, this might take one minute or less...")
        self.prefetcher = DataPrefetcher(self.train_loader)
        # max_iter means iters per epoch 
        self.max_iter = len(self.train_loader)

        self.lr_scheduler = self.exp.get_lr_scheduler(self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter)
        if self.args.occupy:
            occupy_mem(self.local_rank)
        
        if self.is_distributed:
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False)
        
        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998)
            self.ema_model.updates = self.max_iter * self.start_epoch
        
        self.model = model 
        self.model.train()

        self.evaluator = self.exp.get_evaluator(batch_size=self.args.batch_size, is_distributed=self.is_distributed)
        # Tensorboard logger 
        if self.rank==0:
            self.tblogger = SummaryWriter(self.filename)
        
        logger.info("Training start...")
        logger.info(f"\n{model}") 

    def after_train(self):
        logger.info(
            "Training of experiment is done and the best AP is {:.2f}".format(self.best_ap * 100)
        )

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()
    
    def train_in_iter(self):
        for self.iterr in range(self.max_iter):
            self.before_iter()
            self.train_one_iter()
            self.after_iter()

    def after_epoch(self):
        self.save_ckpt(ckpt_name="latest")
        if (self.epoch + 1) % self.exp.eval_interval == 0:
            all_reduce_norm(self.model)
            self.evaluate_and_save_model()

    def before_iter(self):
        pass 
    
    def train_one_iter(self):
        iter_start_time = time.time()
        inps, targets = self.prefetcher.next()
        inps = inps.to(self.data_type)
        targets = targets.to(self.data_type)
        targets.require_grad = False 
        inps, targets = self.exp.preprocess(inps, targets, self.input_size)
        data_end_time = time.time()
    
        with torch.cuda.amp.autocast(enabled = self.amp_training):
            outputs = self.model(inps, targets)
        loss = outputs['total_loss']
        
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.use_model_ema:
            self.ema_model.update(self.model)
        
        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr 
        
        iter_end_time = time.time()
        self.meter.update(iter_time = iter_end_time - iter_start_time, data_time = data_end_time - iter_start_time, lr=lr, **outputs)

    def after_iter(self):
        """
        `after_iter` contains two parts of logic:
            * log information
            * reset setting of resize
        """
        # log needed information
        if (self.iter + 1) & self.exp.print_interval == 0:
            # TODO check ETA logic
            left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter['iter_time'].global_avg * left_iters 
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                ["{}: {:.1f}".format(k, v.latest) for k, v in loss_meter.items()]
            )

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )

            logger.info(
                "{}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}".format(
                    progress_str,
                    gpu_mem_usage(),
                    time_str,
                    loss_str,
                    self.meter["lr"].latest,
                )
                + (", size: {:d}, {}".format(self.input_size[0], eta_str))
            )
            self.meter.clear_meters()
        # random resize
        if (self.progress_in_iter + 1) % 10 == 0:
            self.input_size = self.exp.random_resize(self.train_loader, self.epoch, self.rank, self.is_distributed)

    def before_epoch(self):
        logger.info("---> start train epoch{}".format(self.epoch + 1))
        if self.epoch + 1 == self.max_epoch - self.exp.no_aug_epochs or self.no_aug:
            logger.info("--->No mosaic aug now!")
            self.train_loader.close_mosaic()
            logger.info("--->Add additional L1 loss now!")
            if self.is_distributed:
                self.model.module.head.use_l1 = True 
            else:
                self.model.head.use_l1 = True 
            self.exp.eval_interval = 1
            if not self.no_aug:
                self.save_ckpt(ckpt_name="last_mosaic_epoch")

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter 
    
    def resume_train(self, model):
        if self.args.resume:
            logger.info("resume training")
            if self.args.ckpt is None:
                ckpt_file = os.path.join(self.filename, "latest" + "_ckpt.pth")
            else:
                ckpt_file = self.args.ckpt 
            
            ckpt = torch.load(ckpt_file, map_location=self.device)
            # resume the model/optimizer state dict 
            model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            # resume the training states variables
            start_epoch = self.args.start_epoch-1 if self.args.start_epoch is not None else ckpt['start_epoch']
            self.start_epoch = start_epoch
            logger.info("loaded checkpoint '{}' (epoch {})".format(
                    self.args.resume, self.start_epoch
            ))
        else:
            if self.args.ckpt is not None:
                logger.info("loading checkpoint for finetuning")
                ckpt_file = self.args.ckpt 
                ckpt = torch.load(ckpt_file, map_location=self.device)['model']
                model = load_ckpt(model, ckpt)
            self.start_epoch = 0
        return model
    
    def evaluate_and_save_model(self):
        if self.use_model_ema:
            eval_model = self.ema_model.ema 
        else:
            eval_model = self.model 
            if is_parallel(eval_model):
                eval_model = eval_model.module 
        ap50_95, ap50, summary = self.exp.eval(eval_model, self.evaluator, self.is_distributed)
        self.model.train()
        if self.rank==0:
            self.tblogger.add_scalar('val/COCOAP50', ap50, self.epoch + 1)
            self.tblogger.add_scalar('val/COCOAP50_95', ap50_95, self.epoch + 1)
            logger.info("\n" + summary)
        synchronize()
        self.save_ckpt('last_epoch', ap50_95 > self.best_ap)
        self.best_ap = max(self.best_ap, ap50_95)
    
    def save_ckpt(self, ckpt_name, update_best_ckpt=False):
        if self.rank==0:
            save_model = self.ema_model.ema if self.use_model_ema else self.model
            logger.info("Save weights to {}".format(self.file_name))
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": save_model.state_dict(),
                "optimizer": self.optimizer.state_dict()
            }
            save_checkpoint(ckpt_state, update_best_ckpt, self.filename, ckpt_name)

def load_ckpt(model, ckpt):
    model_state_dict = model.state_dict()
    load_dict = {}
    for key_model, v in model_state_dict.items():
        if key_model not in ckpt:
            logger.warning("{} is not in the ckpt. Please double check and see if this is desired.".format(key_model))
            continue
        v_ckpt = ckpt[key_model]
        if v.shape != v_ckpt.shape:
            logger.warning("Shape of {} in checkpoint is {}, while shape of {} in model is {}.".format(key_model, v_ckpt.shape, key_model, v.shape))
            continue
        load_dict[key_model] = v_ckpt
    model.load_state_dict(load_dict, strict=False)
    return model

def save_checkpoint(state, is_best, save_dir, model_name=''):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, model_name + '_ckpt.pth')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save_dir, 'best_ckpt.pth')
        shutil.copyfile(filename, best_filename)