import torch 
import torch.backends.cudnn as cudnn

import argparse
import random 
import warnings 
from loguru import logger 

from ..core import Trainer, launch
from ..utils import configure_nccl, configure_omp

def make_parser():
    parser = argparse.ArgumentParser('YOLOX train parser')
    parser.add_argument('-expn', '--experiment-name', type=str, default=None)
    parser.add_argument('-n', '--name', type=str, default=None, help='model name')

    # distributed
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--dist-url', default=None, type=str, help='url used to set up distributed training')
    parser.add_argument('-b', '--batch-size', default=64, type=int, help='batch size')
    parser.add_argument('-d', '--devices', default=None, type=int, help='device for training')
    parser.add_argument('-f', '--exp_file', default=None, type=str, help='Please input your experiment description file')
    parser.add_argument('--resume', default=False, action='store_true', help='resume training')
    parser.add_argument('-c', '--ckpt', default=None, type=str, help='checkpoint file')
    parser.add_argument('-e', '--start_epoch', default=None, type=int, help='resume the start epoch of training')
    parser.add_argument('--num-machines', default=1, type=int, help='num of node for training')
    parser.add_argument('--machine-rank', default=0, type=int, help='node rank for multi-node training')
    parser.add_argument('--fp16', dest='fp16', default=False, action='store_true', help='adopting mix precision training')
    parser.add_argument('--cache', dest='cache', default=None, action='store_true', help='caching images to RAM for fast training')
    parser.add_argument('-o', '--occupy', dest='occupy', default=False, action='store_true', help='occupy GPU memory first for training')
    parser.add_argument('opts', help='modify config options using the command-line', default=None, nargs=argparse.REMAINDER)
    return parser 

@logger.catch
def main(exp, args):
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True 
        warnings.warn("You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints.")
    
    # set environment variables for distributed training
    configure_nccl()
    configure_omp()
    cudnn.benchmark = True

    trainer = Trainer(exp, args)
    trainer.train()
