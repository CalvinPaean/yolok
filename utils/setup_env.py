import os 
import subprocess
from loguru import logger
import cv2 
from torch import distributed as dist
import functools 

__all__ = [
    "is_main_process", 
    "configure_nccl", 
    "get_rank", 
    "_get_global_gloo_group", 
    "get_local_rank", 
    "configure_module", 
    "configure_omp", 
    "synchronize",
    "get_world_size"
]

def configure_nccl():
    '''
    Configure multi-machine environment variables of NCCL.
    '''
    os.environ['NCCL_LAUNCH_MODEL'] = 'PARALLEL'
    os.environ['NCCL_IB_HCA'] = subprocess.getoutput(
        "pushd /sys/class/infiniband/ > /dev/null; for i in mlx5_*; "
        "do cat $i/ports/1/gid_attrs/types/* 2>/dev/null "
        "| grep v >/dev/null && echo $i ; done; popd > /dev/null"
    )
    os.environ['NCCL_IB_GID_INDEX'] = '3'
    os.environ['NCCL_IB_TC'] = '106'

def configure_omp(num_threads=1):
    '''
    If OMP_NUM_THREADS is not configured and world_size is greater than 1,
    Configure OMP_NUM_THREADS environment variables of NCCL to `num_thread`.

    Args:
        num_threads (int): value of `OMP_NUM_THREADS` to set.
    '''
    if 'OMP_NUM_THREADS' not in os.environ and get_world_size()>1:
        os.environ['OMP_NUM_THREADS'] = str(num_threads)
        if is_main_process():
            logger.info("\n***************************************************************\n"
                "We set `OMP_NUM_THREADS` for each process to {} to speed up.\n"
                "please further tune the variable for optimal performance.\n"
                "***************************************************************".format(
                    os.environ["OMP_NUM_THREADS"]
                ))


def configure_module(ulimit_value=8192):
    '''
    Configure pytorch module environment. setting of ulimit and cv2 will be set.

    Args:
        ulimit_value(int): default open file number on linux. Default value: 8192.
    '''
    try:
        import resource 
        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (ulimit_value, rlimit[1]))
    except Exception:
        # Exception might be raised in Windows OS or rlimit reaches max limit number.
        # However, set rlimit value might not be necessary.
        pass
    # cv2 
    # multiprocess may be harmful on performance of torch dataloader
    os.environ['OPENCV_OPENCL_RUNTIME'] = 'disabled'
    try:
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        # cv2 version mismatch might raise exception
        pass

