
import torch 
import torch.nn as nn 
import math
import numpy as np 

class IOULoss(nn.Module):
    def __init__(self, reduction='none', loss_type='iou'):
        super(IOULoss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type
    
    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]
        eps = 1e-7
        pred = pred.view(-1, 4) # center x, center y, width, height
        target = target.view(-1, 4) # center x, center y, width, height
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )
        area_pred = torch.prod(pred[:, 2:], 1)
        area_targ = torch.prod(target[:, 2:], 1)

        # en = (tl < br).type(tl.type()).prod(dim=1)
        en = (tl < br).all()
        area_ovr = torch.prod(br-tl, 1) * en 
        area_uni = area_pred + area_targ - area_ovr
        iou = (area_ovr) / (area_uni.clamp(eps)) 
        loss = torch.zeros_like(iou).to(iou.device)
        if self.loss_type == 'iou':
            loss = 1 - iou ** 2
        elif self.loss_type == 'giou':
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            c_area = torch.prod(c_br - c_tl, 1)
            giou = iou - (c_area - area_uni) / c_area.clamp(eps)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)
        elif self.loss_type == 'ciou' or self.loss_type == 'diou':            
            # smallest enclosing box width
            c_width = torch.max(pred[:, 0]+pred[:, 2]/2, target[:, 0]+target[:, 2]/2) - torch.min(pred[:, 0]-pred[:, 2]/2, target[:, 0]-target[:, 2]/2) 
            # smallest enclosing box height
            c_height = torch.max(pred[:, 1]+pred[:, 3]/2, target[:, 1]+target[:, 3]/2) - torch.min(pred[:, 1]-pred[:, 3]/2, target[:, 1]-target[:, 3]/2) 
            c_diagnal = (c_width ** 2 + c_height ** 2).clamp(eps) # diagnal length of smallest enclosing box
            rho2 = ((target[:, 0] - pred[:, 0])**2 + (target[:, 1] - pred[:, 1])**2) # distance between pred box center and target box center
            if self.loss_type == 'ciou':
                pw, ph = (pred[:, 2]).clamp(eps), (pred[:, 3]).clamp(eps) # predicted box width, height
                gw, gh = (target[:, 2]).clamp(eps), (target[:, 3]).clamp(eps) # target box width, height                
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(gw / gh) - torch.atan(pw / ph), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou  + v).clamp(eps)
                loss = 1 - iou + (rho2 / c_diagnal + v * alpha)
            else:
                loss = 1 - iou + rho2 / c_diagnal
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

'''
if __name__ == '__main__':
    pred = np.array([[50, 50, 100, 100]])
    target = np.array([[55, 60, 100, 110]])
    pred = torch.from_numpy(pred)
    target = torch.from_numpy(target)
    iouloss = IOULoss()
    iouloss.forward(pred, target)
'''