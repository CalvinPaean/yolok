
import itertools
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data.sampler import BatchSampler as torchBatchSampler
from torch.utils.data.sampler import Sampler

class YoloBatchSampler(torchBatchSampler):
    """
    This batch sampler will generate mini-batches of (mosaic, index) tuples from another sampler.
    It works just like the :class:`torch.utils.data.sampler.BatchSampler`,
    but it will turn on/off the mosaic aug.
    """
    def __init__(self, *args, mosaic=True, **kwargs):
        super(YoloBatchSampler, self).__init__(*args, **kwargs)
        self.mosaic = mosaic
    def __iter__(self):
        for batch in super().__iter__():
            yield [(self.mosaic, idx) for idx in batch]