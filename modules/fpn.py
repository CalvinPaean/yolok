import torch 
import torch.nn as nn 

from .darknet_backbone import Darknet53
from .blocks import * 

class FPN(nn.Module):
    '''
    YOLOFPN Module. Darknet53 is the default backbone of this model.
    '''
    def __init__(self, bk_depth=53, inputs=['block3', 'block4', 'block5']) -> None:
        super(FPN, self).__init__()
        self.backbone = Darknet53(bk_depth)
        self.inputs = inputs 

        # output 1
        self.out1_cbl = self.build_cbl(512, 256, 1)
        self.out1 = self.build_cbl_seq([256, 512], 512 + 256)

        # output 2
        self.out2_cbl = self.build_cbl(256, 128, 1)
        self.out2 = self.build_cbl_seq([128, 256], 256 + 128)

        # upsample
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
    def build_cbl_seq(self, in_c_list, in_c):
        modules = nn.Sequential(*[
                self.build_cbl(in_c, in_c_list[0], 1),
                self.build_cbl(in_c_list[0], in_c_list[1], 3),
                self.build_cbl(in_c_list[1], in_c_list[0], 1),
                self.build_cbl(in_c_list[0], in_c_list[1], 3),
                self.build_cbl(in_c_list[1], in_c_list[0], 1)])
        return modules
    
    def build_cbl(self, in_c, out_c, k):
        return Conv(in_c, out_c, k, s=1, act='leaky_relu')

    def load_pretrained(self, filename):
        with open(filename, 'r') as f:
            state_dict = torch.load(f, map_location='cpu')
        print(f'Loading pretrained weights from {filename}...')
        self.backbone.load_state_dict(state_dict)

    def forward(self, x):
        '''
        Args:
            x (Tensor): input image.
        Returns:
            Tuple[Tensor]: FPN output features.
        '''
        # backbone
        output_feats = self.backbone(x)
        x2, x1, x0 = [output_feats[ft] for ft in self.inputs] # 'block3', 'block4', 'block5'

        # YOLO branch 1
        x1_in = self.out1_cbl(x0)
        x1_in = self.upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], dim=1)
        out_block4 = self.out1(x1_in)

        # YOLO branch 2
        x2_in = self.out2_cbl(out_block4)
        x2_in = self.upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], dim=1)
        out_block3 = self.out2(x2_in)
        
        return (out_block3, out_block4, x0)