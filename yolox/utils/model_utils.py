import torch
import torch.nn as nn 
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

def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = ( #in_c, out_c, k, s, g=1, bias=False
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            groups=conv.groups,
            bias=True        
        ).requires_grad_(False).to(conv.weight.device)
    )
    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1) # w
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var))) # \frac{\gamma}{\sqrt{\epsilon + running_var}}
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape)) # w \times \frac{\gamma}{\sqrt{\epsilon + running_var}}
    # prepare spatial bias
    b_conv = (torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias) # b
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps)) # \beta - \frac{\gamma \times running_mean}{\sqrt{\epsilon + running_var}}
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn) #  \frac{\gamma \times running_mean}{\sqrt{\epsilon + running_var}} \times (b - running_mean) + \beta
    return fusedconv


def fuse_model(model):
    from yolox.models.blocks import Conv 
    for m in model.modules():
        if type(m) is Conv and hasattr(m, 'bn'):
            m.conv = fuse_conv_and_bn(m.conv, m.bn) # update conv 
            delattr(m, 'bn') # remove batch norm
            m.forward = m.fuseforward # update forward 
    return model 