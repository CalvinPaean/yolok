import torch 
import torchvision
import torch.nn as nn 
import numpy as np


__all__ = [
    "filter_box",
    "postprocess",
    "bboxes_iou",
    "matrix_iou",
    "adjust_box_anns",
    "xyxy2xywh",
    "xyxy2cxcywh",
]

def filter_box(output, scale_range):
    """
    output: (N, 5+class) shape
    """
    min_scale, max_scale = scale_range
    w = output[:, 2] - output[:, 0]
    h = output[:, 3] - output[:, 1]
    keep = (w * h > min_scale * min_scale) & (w * h < max_scale * max_scale)
    return output[keep]

def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    return bbox

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

def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i + 1e-12)

def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, img_pred in enumerate(prediction):
        # if non are remained => process next image 
        if not img_pred.size(0):
            continue
        # get score and class with highest confidence
        cls_conf, cls_pred = torch.max(img_pred[:, 5: 5 + num_classes], 1, keepdim=True)
        conf_mask = (img_pred[:, 4] * cls_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((img_pred[:, :5], cls_conf, cls_pred.float()), dim=1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue
        if class_agnostic:
            nms_out_index = torchvision.ops.nms(detections[:, :4], detections[:, 4] * detections[:, 5], nms_thre)
        else:
            nms_out_index = torchvision.ops.nms(detections[:, :4], detections[:, 4] * detections[:, 5], detections[:, 6], nms_thre)
        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))
    return output

def xyxy2cxcywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes

def xyxy2xywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes

'''
if __name__ == '__main__':
    a = torch.tensor([[3,0,10,10]])
    b = torch.tensor([[2,2,8,12]])
    tl = torch.max(a[:, None, :2], b[:, :2])
    br = torch.min(a[:, None, 2:], b[:, 2:])
    
    print(a[:, None, :2])
    print(b[:, :2])
'''