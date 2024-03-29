import torch 
import torch.backends.cudnn as cudnn

import argparse
import random 
import warnings 
from loguru import logger 

from yolox.core import Trainer, launch
from yolox.utils import configure_nccl, configure_omp, get_num_devices
from yolox.exp import get_exp

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
        torch.manual_seed(exp.seed) # 设置 pytorch 的随机种子为固定值，保证每次网络运行的时候相同的输入会得到相同的输出
        cudnn.deterministic = True # 设为 True, 每次返回的卷积算法都将是确定的，即默认算法。
        warnings.warn("You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints.")
    
    # set environment variables for distributed training
    configure_nccl() # 配置 NCCL 的环境变量
    configure_omp() # 配置 OMP 的环境变量
    cudnn.benchmark = True # 为整个网络的每个卷积层搜索最适合的实现算法，实现网络加速。参考 https://zhuanlan.zhihu.com/p/73711222

    trainer = Trainer(exp, args)
    trainer.train()

if __name__ == '__main__':
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name) # for experiment recording
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name
    
    num_gpu = get_num_devices() if args.devices is None else args.devices
    
    assert num_gpu <= get_num_devices()

    dist_url = 'auto' if args.dist_url is None else args.dist_url

    launch(main, num_gpu, args.num_machines, args.machine_rank, backend=args.dist_backend, dist_url=dist_url, args=(exp, args))