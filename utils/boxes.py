import torch 
import torch.nn as nn 
import numpy as np

def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    assert bboxes_a.shape[1] == 4 and bboxes_b.shape[1] == 4, "Any bounding box must have 4 coordinates."

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod((bboxes_a[:, 2:] - bboxes_a[:, :2]), dim=1)
        area_b = torch.prod((bboxes_b[:, 2:] - bboxes_b[:, :2]), dim=1)
    else:
        tl = torch.max(bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2,
                       bboxes_b[:, None, :2] - bboxes_b[:, 2:] / 2)
        br = torch.min(bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2,
                       bboxes_b[:, None, :2] + bboxes_b[:, 2:] / 2)
        area_a = torch.prod(bboxes_a[:, 2:], dim=1)
        area_b = torch.prod(bboxes_b[:, 2:], dim=1)
    en = (tl < br).all()
    area_ovr = torch.prod(br-tl, dim=2) * en 
    return area_ovr / (area_a[:, None] + area_b - area_ovr)

'''
if __name__ == '__main__':
    a = torch.tensor([[3,0,10,10]])
    b = torch.tensor([[2,2,8,12]])
    tl = torch.max(a[:, None, :2], b[:, :2])
    br = torch.min(a[:, None, 2:], b[:, 2:])
    
    print(a[:, None, :2])
    print(b[:, :2])
'''