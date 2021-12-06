import torch 
import torch.nn as nn 

from .head import YOLOXHead
from .pafpn import PAFPN

class YOLOX(nn.Module):
    '''
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    '''

    def __init__(self, backbone=None, head=None):
        super(YOLOX, self).__init__()
        if backbone is None:
            backbone = PAFPN()
        if head is None:
            head = YOLOXHead(80)
        
        self.backbone = backbone 
        self.head = head 

    def forward(self, input, targets=None):
        # FPN output content features of [block3, block4, block5]
        fpn_results = self.backbone(input)
        if self.training:
            assert targets is not None 
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(fpn_results, targets, input)
            results = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg
            }
        else:
            results = self.head(fpn_results)
        return results
