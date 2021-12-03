import torch.nn as nn 
from .blocks import * 

class Darknet53(nn.Module):
    '''
    Darknet21 or Darknet53
    '''
    BlocksConfig = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4]}

    def __init__(self, bk_depth, in_c = 3, out_c = 32, outputs = ['block3', 'block4', 'block5']) -> None:
        """
        Args:
            bk_depth (int): depth of darknet backbone used in the model, typically use [21, 53] for the program.
            in_c (int): num of input channels, 3 for RGB image.
            out_c (int): num of output channels of darknet block1, which decides the channels of darknet block2 to block5.
            outputs (List[str]): names of the layers desired to output
        """
        super(Darknet, self).__init__()
        assert outputs, "Please provide output features of Darknet"
        self.outputs = outputs
        self.block1 = nn.Sequential(
            Conv(in_c, out_c, k=3, s=1, act='leaky_relu'),
            *self.build_group_layer(out_c, n=1, s=2)
        )
        in_c = out_c * 2 # 64 
        n_blocks = Darknet.BlocksConfig[bk_depth] # number of resnet bottlenecks
        '''
        create darknet with `out_c` channels and `n` bottlenecks.
        '''
        self.block2 = nn.Sequential(
            *self.build_group_layer(in_c, n_blocks[0], s=2)
        )
        in_c *= 2 # 128
        self.block3 = nn.Sequential(
            *self.build_group_layer(in_c, n_blocks[1], s=2)
        )
        in_c *= 2 # 256
        self.block4 = nn.Sequential(
            *self.build_group_layer(in_c, n_blocks[2], s=2)
        )
        in_c *= 2 # 512
        self.block5 = nn.Sequential(
            *self.build_group_layer(in_c, n_blocks[3], s=2),
            *self.build_spp_layer([in_c, in_c * 2], in_c * 2)
        )
    
    def build_spp_layer(self, ch_list, in_c):
        modules = nn.Sequential(
            *[
                Conv(in_c, ch_list[0], 1, s=1, act='leaky_relu'),
                Conv(ch_list[0], ch_list[1], 3, s=1, act='leaky_relu'),
                SPPBottleneck(in_c=ch_list[1], out_c=ch_list[0], act='leaky_relu'),
                Conv(ch_list[0], ch_list[1], 3, s=1, act='leaky_relu'),
                Conv(ch_list[1], ch_list[0], 1, s=1, act='leaky_relu')
            ]
        )
        return modules

    def build_group_layer(self, in_c, n, s = 1):
        '''
        Started with conv layer, then followed by `n` `ResBottleneck`.
        The number of input channels is `in_c`, and stride is `s`.
        '''
        return [
            Conv(in_c, in_c * 2, k=3, s=s, act='leaky_relu'),
            *[(ResBottleneck(in_c * 2)) for _ in range(n)]
        ]

    def forward(self, x):
        results = {}
        x = self.block1(x)
        results["block1"] = x
        x = self.block2(x)
        results["block2"] = x 
        x = self.block3(x)
        results["block3"] = x 
        x = self.block4(x)
        results["block4"] = x 
        x = self.block5(x)
        results["block5"] = x 
        return {k: v for k, v in results.items() if k in self.outputs}


class CSPDarknet53(nn.Module):
    def __init__(self, depth, width, depthwise = False, outputs = ['block3', 'block4', 'block5'], act='silu') -> None:
        super(CSPDarknet, self).__init__()
        assert outputs, "Please provide output features of Darknet"
        self.outputs = outputs 
        conv = DepthwiseConv if depthwise else Conv 

        channels_base = int(width * 64) # 64
        depth_base = max(round(depth * 3), 1) # 3

        # block1 (3, 64, 3, `silu`)
        self.block1 = Focus(3, channels_base, k=3, act=act)

        # block2
        self.block2 = nn.Sequential(
            conv(channels_base, channels_base * 2, 3, 2, act=act), # (64, 128, 3, 2, `silu`)
            CSPBottleneck(channels_base*2, channels_base*2, n=depth_base, depthwise=depthwise, act=act) # (128, 128, 3, depthwise, `silu`)
        )

        # block3
        self.block3 = nn.Sequential(
            conv(channels_base * 2, channels_base * 4, 3, 2, act=act), # (128, 256, 3, 2, `silu`)
            CSPBottleneck(channels_base*4, channels_base*4, n=depth_base*3, depthwise=depthwise, act=act) # (256, 256, 9, depthwise, `silu`)
        )

        # block4
        self.block4 = nn.Sequential(
            conv(channels_base * 4, channels_base * 8, 3, 2, act=act), # (256, 512, 3, 2, `silu`)
            CSPBottleneck(channels_base*8, channels_base*8, n=depth_base*3, depthwise=depthwise, act=act) # (512, 512, 9, depthwise, `silu`)
        )

        # block5
        self.block5 = nn.Sequential(
            conv(channels_base*8, channels_base*16, 3, 2, act=act), # (512, 1024, 3, 2, `silu`)
            SPPBottleneck(channels_base*16, channels_base*16, act=act), # (1024, 1024, `silu`)
            CSPBottleneck(channels_base*16, channels_base*16, n=depth_base, shortcut=False, depthwise=depthwise, act=act) # (1024, 1024, 3, depthwise, `silu`)
        )
    
    def forward(self, x):
        results = {}
        x = self.block1(x)
        results["block1"] = x
        x = self.block2(x)
        results["block2"] = x 
        x = self.block3(x)
        results["block3"] = x 
        x = self.block4(x)
        results["block4"] = x 
        x = self.block5(x)
        results["block5"] = x 
        return {k: v for k, v in results.items() if k in self.outputs}
