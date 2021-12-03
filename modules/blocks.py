
import torch
import torch.nn as nn 
import torch.nn.functional as F 

class SiLU(nn.Module):
    '''
    y = x * sigmoid(x)
    '''
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x) 

class Mish(nn.Module):
    '''
    y = x * tanh(ln(1+e^x))
    '''
    def __init__(self) -> None:
        super(Mish, self).__init__()
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

def get_act(act='silu', inplace=True):
    if act == 'silu':
        module = nn.SiLU(inplace=inplace)
    elif act == 'relu':
        module = nn.ReLU(inplace=inplace)
    elif act == 'leaky_relu':
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif act == 'mish':
        module = Mish()
    else:
        raise AttributeError(f"Unsupported activation type: {act}")
    return module

class Conv(nn.Module):
    '''
    Conv2d + BN + Activation
    '''
    def __init__(self, in_c, out_c, k, s, g=1, bias=False, act='silu') -> None:
        super(Conv, self).__init__()
        padding = (k - 1) // 2
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, groups=g, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = get_act(act, inplace=True) if isinstance(act, str) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class DepthwiseConv(nn.Module):
    '''
    Depthwise conv + Conv
    '''
    def __init__(self, in_c, out_c, k, s=1, act='silu') -> None:
        self.dwconv = Conv(in_c, in_c, k=k, s=s, g=in_c, act=act)
        self.pwconv = Conv(in_c, out_c, k=1, s=1, g=1, act=act)

    def forward(self, x):
        return self.pwconv(self.dwconv(x))

class Bottleneck(nn.Module):
    '''
    Standard bottleneck: shortcut connection between 1*1 conv + 3*3 conv
    '''
    def __init__(self, in_c, out_c, shortcut=True, exp=0.5, depthwise=False, act='silu') -> None:
        super(Bottleneck, self).__init__()
        hid_c = int(out_c * exp) # hidden channels = output_channels * expansion_rate
        conv = DepthwiseConv if depthwise else Conv 
        self.conv1 = Conv(in_c, hid_c, 1, s=1, act=act)
        self.conv2 = conv(hid_c, out_c, 3, s=1, act=act)
        self.shortcut = shortcut and in_c == out_c
    
    def forward(self, x):
        out = (self.conv2(self.conv1(x)) + x) if self.shortcut else self.conv2(self.conv1(x))
        return out

class ResBottleneck(nn.Module):
    '''
    Residual layer whose input has in_c channels
    '''
    def __init__(self, in_c) -> None:
        super(ResBottleneck, self).__init__()
        hid_c = in_c // 2
        self.conv1 = Conv(in_c, hid_c, k=1, s=1, act='leaky_relu')
        self.conv2 = Conv(hid_c, in_c, k=3, s=1, act='leaky_relu')
    def forward(self, x):
        return self.conv2(self.conv1(x)) + x

class SPPBottleneck(nn.Module):
    '''
    Spatial pyramid pooling (SPP) layer from https://arxiv.org/abs/1406.4729
    '''
    def __init__(self, in_c, out_c, k=[5, 9, 13], act='silu') -> None:
        super(SPPBottleneck, self).__init__()
        hid_c = in_c // 2
        self.conv1 = Conv(in_c, hid_c, k=1, s=1, act=act)
        self.conv2 = Conv(hid_c*(1+len(k)), out_c, k=1, s=1, act=act)
        self.modules = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x//2) for x in k])

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [m(x) for m in self.modules], dim=1))

class CSPBottleneck(nn.Module):
    '''
    CSP bottleneck with 3 convolutions
    '''
    def __init__(self, in_c, out_c, n=1, shortcut=True, exp=0.5, depthwise=False, act='silu') -> None:
        # n: the number of bottlenecks
        # exp: expansion rate
        super(CSPBottleneck, self).__init__()
        hid_c = int(out_c * exp)
        self.conv1 = Conv(in_c, hid_c, k=1, s=1, act=act)
        self.conv2 = Conv(in_c, hid_c, k=1, s=1, act=act)
        self.conv3 = Conv(2 * hid_c, out_c, k=1, s=1, act=act)
        self.modules = nn.Sequential(*(Bottleneck(hid_c, hid_c, shortcut, 1.0, depthwise, act=act) for _ in range(n)))

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.modules(x1)
        x2 = self.conv2(x)        
        return self.conv3(torch.cat((x1, x2), dim=1))

class GhostConv(nn.Module):
    '''
    Ghost Convolution from https://github.com/huawei-noah/ghostnet
    '''
    def __init__(self, in_c, out_c, k=1, s=1, g=1, act='silu') -> None:
        super(GhostConv, self).__init__()
        hid_c = out_c // 2
        self.conv1 = Conv(in_c, hid_c, k=k, s=s, g=g, act=act)
        self.conv2 = Conv(hid_c, hid_c, k=5, s=1, g=hid_c, act=act)

    def forward(self, x):
        y = self.conv1(x)
        return torch.cat([y, self.conv2(y)], dim=1)

class GhostBottleneck(nn.Module):
    '''
    Ghost bottleneck from https://github.com/huawei-noah/ghostnet
    '''
    def __init__(self, in_c, out_c, k=3, s=1) -> None:
        super(GhostBottleneck, self).__init__()
        hid_c = out_c // 2
        self.conv = nn.Sequential(
            GhostConv(in_c, hid_c, 1, 1), # Ghost module 1: pointwise
            DepthwiseConv(hid_c, hid_c, k, s, act=None) if s == 2 else nn.Identity(), # depthwise operation when stride is 2
            GhostConv(hid_c, out_c, 1, 1, act=None)) # Ghost module 2: linear operation
        self.shortcut = nn.Sequential(
            DepthwiseConv(in_c, in_c, k, s, act=None),
            Conv(in_c, out_c, 1, 1, act=None)) if s == 2 else nn.Identity()
    def forward(self, x):
        return self.conv(x) + self.shortcut(x)

class Expand(nn.Module):
    '''
    Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    '''
    def __init__(self, gain=2) -> None:
        super(Expand, self).__init__()
        self.gain = gain 
    def forward(self, x):
        n, c, h, w = x.shape # x(1,64,80,80)
        x = x.view(n, self.gain, self.gain, c // self.gain ** 2, h, w) # x(1, 2, 2, 16, 80, 80)
        x = x.permute(0, 3, 4, 1, 5, 2) # x(1, 16, 80, 2, 80, 2)
        return x.reshape(n, c // self.gain ** 2, self.gain * h, self.gain * w) # x(1, 16, 160, 160)
        
class Contract(nn.Module):
    '''
    Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    '''
    def __init__(self, gain=2) -> None:
        super(Contract, self).__init__()
        self.gain = gain 
    def forward(self, x):
        n, c, h, w = x.shape # x(1, 64, 80, 80)
        x = x.view(n, c, h // self.gain, self.gain, w // self.gain, self.gain) # x(1, 64, 40, 2, 40, 2)
        x = x.permute(0, 3, 5, 1, 2, 4) # x(1, 2, 2, 64, 40, 40)
        return  x.reshape(n, c * self.gain ** 2, h // self.gain, w // self.gain) # x(1, 256, 40, 40)

class Focus(nn.Module):
    '''
    (b,c,w,h) -> (b,4c,w/2,h/2)
    '''
    def __init__(self, in_c, out_c, k=1, s=1, g=1, act='silu') -> None:
        super(Focus, self).__init__()
        self.conv = Conv(in_c*4, out_c, k=k, s=s, g=g, act=act)
    def forward(self, x):
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], dim=1))
