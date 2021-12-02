
import torch 
import torch.nn as nn 

class IOULoss(nn.Module):
    def __init__(self, reduction='none', loss_type='iou'):
        super(IOULoss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type
    
    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4) # center x, center y, width, height
        target = target.view(-1, 4) # center x, center y, width, height
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.max(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )
        area_pred = torch.prod(pred[:, 2:], 1)
        area_targ = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_ovr = torch.prod(br-tl, 1) * en 
        area_uni = area_pred + area_targ - area_ovr
        iou = (area_ovr) / (area_uni.clamp(1e-16)) 

        if self.loss_type == 'iou':
            loss = 1 - iou ** 2
        elif self.loss_type == 'giou':
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_uni) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss
