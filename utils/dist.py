"""
This file contains primitives for multi-gpu communication.
This is useful when doing distributed training.
"""
import functools
import os
import pickle
import time
from contextlib import contextmanager
from loguru import logger


import numpy as np

import torch
from torch import distributed as dist


__all__ = [
    "get_num_devices",
    "wait_for_the_master",
    "is_main_process",
    "synchronize",
    "get_world_size",
    "get_rank",
    "get_local_rank",
    "get_local_size",
    "time_synchronized",
    "gather",
    "all_gather",
]

_LOCAL_PROCESS_GROUP = None
def get_local_rank() -> int:
    """
    Returns:
        The rank of the current process within the local (per-machine) process group.
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    assert _LOCAL_PROCESS_GROUP is not None
    return dist.get_rank(group=_LOCAL_PROCESS_GROUP)

def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0 
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0

def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()