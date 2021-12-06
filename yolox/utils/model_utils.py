import torch
from copy import deepcopy
from thop import profile


__all__ = [
    "fuse_conv_and_bn",
    "fuse_model",
    "get_model_info",
    "replace_module",
]


def get_model_info(model, tsize):
    stride = 64 
    img = torch.zeros((1, 3, stride, stride), device=next(model.parameters()).device)
    flops, parameters = profile(deepcopy(model), inputs=(img,), verbose=False)
    parameters /= 1e6 
    flops /= 1e9 
    flops *= 2 * tsize[0] * tsize[1] / stride ** 2 # GFLOPS
    info = "Parameters: {:.2f}M, GFLOPS: {:.2f}".format(parameters, flops)
    return info