
import torch 
import torch.nn as nn 

from .darknet_backbone import CSPDarknet53
from .blocks import * 

class PAFPN(nn.Module):
    '''
    YOLOv3 model. Darknet53 is the default backbone of this model.
    Read `yolok/data/cspdarknet53.jpeg` for details about network structure.
    '''
    def __init__(self, depth=1.0, width=1.0, inputs = ['block3','block4','block5'], in_c=[256,512,1024], depthwise=False, act='silu'):
        super(PAFPN, self).__init__()
        self.backbone = CSPDarknet53(depth, width, depthwise=depthwise, act=act)
        self.inputs = inputs 
        self.in_c = in_c 
        conv = DepthwiseConv if depthwise else Conv 

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.shortcut_conv = Conv(int(in_c[2] * width), int(in_c[1] * width), k=1, s=1, act=act)
        self.neck_block4_up = CSPBottleneck(
            int(2 * in_c[1] * width),
            int(in_c[1] * width), 
            round( 3 * depth),
            False,
            depthwise=depthwise,
            act=act
        ) # cat

        self.reduce_conv = Conv(int(in_c[1] * width), int(in_c[0] * width), k=1, s=1, act=act)
        self.neck_block3_up = CSPBottleneck(
            int(2 * in_c[0] * width),
            int(in_c[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act
        )

        # bottom-up convolution
        self.bu_conv2 = conv(int(in_c[0] * width), int(in_c[0] * width), k=3, s=2, act=act)
        self.neck_block3_down = CSPBottleneck(
            int(2 * in_c[0] * width),
            int(in_c[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act
        )

        # bottom-up convolution
        self.bu_conv1 = conv(int(in_c[1] * width), int(in_c[1] * width), k=3, s=2, act=act)
        self.neck_block4_down = CSPBottleneck(
            int(2 * in_c[1] * width),
            int(in_c[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act
        )

    def forward(self, x):
        '''
        Args:
            x: input images.
        Returns:
            Tuple[Tensor]: FPN Features.
        '''
        # backbone 
        output_feats = self.backbone(x)
        feats = [output_feats[ft] for ft in self.inputs]
        [x2, x1, x0] = feats # 'block3', 'block4', 'block5'

        fpn_out0 = self.shortcut_conv(x0) # 1024 -> 512 / 32
        f_out0 = self.upsample(fpn_out0) # 512 / 16
        f_out0 = torch.cat([f_out0, x1], dim=1) # 512 -> 1024 / 16
        f_out0 = self.neck_block4_up(f_out0) # 1024 -> 512 / 16

        fpn_out1 = self.reduce_conv(f_out0) # 512 -> 256 / 16
        f_out1 = self.upsample(fpn_out1) # 256 / 8
        f_out1 = torch.cat([f_out1, x2], dim=1) # 256 -> 512 / 8
        pan_out2 = self.neck_block3_up(f_out1) # 512 -> 256 / 8

        p_out1 = self.bu_conv2(pan_out2) # 256 -> 256 / 16 
        p_out1 = torch.cat([p_out1, fpn_out1], dim=1) # 256 -> 512 / 16 
        pan_out1 = self.neck_block3_down(p_out1) # 512 -> 512 / 16 

        p_out0 = self.bu_conv1(pan_out1) # 512 -> 512 / 32
        p_out0 = torch.cat([p_out0, fpn_out0], dim=1) # 512 -> 1024 / 32
        pan_out0 = self.neck_block4_down(p_out0) # 1024 -> 1024 / 32

        return (pan_out2, pan_out1, pan_out0)
        