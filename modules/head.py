import torch
from torch import dtype 
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.modules import padding
from torch.nn.modules.loss import L1Loss 

from loguru import logger 

from ..utils import bboxes_iou

from .losses import IOULoss
from .blocks import Conv, DepthwiseConv 

class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width=1.0, s_list=[8,16,32], in_c_list=[256,512,1024], act='silu', depthwise=False):
        super(YOLOXHead,self).__init__()
        self.num_anchor = 1
        self.num_classes = num_classes
        self.decode_in_reference = True

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        conv = DepthwiseConv if depthwise else Conv 

        for i in range(len(in_c_list)):
            self.stems.append(Conv(in_c = int(in_c_list[i] * width), out_c=int(256 * width), k=1, s=1, act=act))
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        conv(in_c = int(256 * width), out_c = int(256 * width), k=3, s=1, act=act),
                        conv(in_c = int(256 * width), out_c = int(256 * width), k=3, s=1, act=act)
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        conv(in_c = int(256 * width), out_c = int(256 * width), k=3, s=1, act=act),
                        conv(in_c = int(256 * width), out_c = int(256 * width), k=3, s=1, act=act)
                    ]
                )
            )
            self.cls_preds.append(nn.Conv2d(int(256 * width), self.num_anchor * self.num_classes, kernel_size=1, stride=1, padding=0))
            self.reg_preds.append(nn.Conv2d(int(256 * width), self.num_anchor * 4, kernel_size=1, stride=1, padding=0))
            self.obj_preds.append(nn.Conv2d(int(256 * width), self.num_anchor * 1, kernel_size=1, stride=1, padding=0))
        self.use_l1_loss = False 
        self.l1_loss = nn.L1Loss(reduction='none')
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.iou_loss = IOULoss(reduction='none')
        self.s_list = s_list # strides list
        self.grids = [torch.zeros(1)] * len(in_c_list)

    def forward(self, inputs, labels=None, imgs=None):
        outputs = []
        org_preds = []
        x_offsets = [] 
        y_offsets = [] 
        expanded_strides = []

        for i, (cls_conv, reg_conv, cur_stride, input) in zip(self.cls_convs, self.reg_convs, self.s_list, inputs):
            input = self.stems[i](input)
            cls_input = input 
            reg_input = input 

            cls_feature = cls_conv(cls_input)
            cls_output = self.cls_preds[i](cls_feature)

            reg_feature = reg_conv(reg_input)
            reg_output = self.reg_preds[i](reg_feature)
            obj_output = self.obj_preds[i](reg_feature)

            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], dim=1)
                output, grid = self.get_output_and_grid(output, i, cur_stride, inputs[0].type())
                x_offsets.append(grid[:, :, 0])
                y_offsets.append(grid[:, :, 1])
                expanded_strides.append(torch.zeros(1, grid.shape[1]).fill_(cur_stride).type_as(inputs[0]))
                if self.use_l1_loss:
                    batch_size = reg_output.shape[0]
                    height, width = reg_output.shape[-2:]
                    reg_output = reg_output.view(batch_size, self.num_anchor, 4, height, width).permute(0, 1, 3, 4, 2).contiguous()
                    reg_output = reg_output.reshape(batch_size, -1, 4)
                    org_preds.append(reg_output.clone())
            else:
                output = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], dim=1)
            outputs.append(output)
        if self.training:
            return self.get_losses(imgs, x_offsets, y_offsets, expanded_strides, labels, torch.cat(outputs, 1), org_preds, dtype=inputs[0].dtype)
        else:
            self.height_width =  [input.shape[-2:] for input in inputs]
            # [batch, num_anchor_all, 80 + 4 + 1]
            outputs = torch.cat([output.flatten(start_dim=2) for output in outputs], dim=2).permute(0, 2, 1)
            return self.decode_outputs(outputs, dtype=inputs[0].dtype) if self.decode_in_reference else outputs

    def get_losses(self, imgs, x_offsets, y_offsets, expanded_strides, labels, outputs, org_preds, dtype):
        bbox_preds = outputs[:,:,:4] # [batch, num_anchor_all, 4]
        obj_preds = outputs[:,:,4].unsqueeze(-1) # [batch, num_anchor_all, 1]
        cls_preds = outputs[:,:,5:] # [batch, num_anchor_all, num_classes]

        # compute targets
        num_labels = (labels.sum(dim=2) > 0).sum(dim=1) # number of objects. labels: (batchsize, num_gt, 1 cls + 4 bbox_coordinates)
        num_anchor_all = outputs.shape[1]
        x_offsets = torch.cat(x_offsets, 1) # [1, num_anchor_all]
        y_offsets = torch.cat(y_offsets, 1) # [1, num_anchor_all]
        expanded_strides = torch.cat(expanded_strides, 1) 
        if self.use_l1_loss:
            org_preds = torch.cat(org_preds, 1)
        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = [] 

        num_fg = 0.
        num_gts = 0.

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(num_labels[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((num_anchor_all, 1))
                fg_mask = outputs.new_zeros(num_anchor_all).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]
                try:
                    (gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg_img) = self.get_assignments(
                            batch_idx, num_gt, num_anchor_all, gt_bboxes_per_image, gt_classes, bboxes_preds_per_image, expanded_strides, 
                            x_offsets, y_offsets, cls_preds, bbox_preds, obj_preds, labels, imgs)

                except RuntimeError:
                    logger.error("OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size.")
                    torch.cuda.empty_cache()
                    (gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg_img) = self.get_assignments(
                            batch_idx, num_gt, num_anchor_all, gt_bboxes_per_image, gt_classes, bboxes_preds_per_image, expanded_strides, 
                            x_offsets, y_offsets, cls_preds, bbox_preds, obj_preds, labels, imgs, 'cpu'
                    )
                torch.cuda.empty_cache()
                num_fg += num_fg_img
                cls_target = F.one_hot(gt_matched_classes.to(torch.int64), self.num_classes) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                if self.use_l1_loss:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_offsets = x_offsets[0][fg_mask],
                        y_offsets = y_offsets[0][fg_mask]
                    )
            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1_loss:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        if self.use_l1_loss:
            l1_targets = torch.cat(l1_targets, 0)
        
        num_fg = max(num_fg, 1)
        loss_iou = (self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)).sum() / num_fg 
        loss_obj = (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)).sum() / num_fg 
        loss_cls = (self.bcewithlog_loss(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets)).sum() / num_fg 
        if self.use_l1_loss:
            loss_l1 = self.l1_loss(org_preds.view(-1, 4)[fg_masks], l1_targets).sum() / num_fg 
        else:
            loss_l1 = 0. 
        
        reg_weight = 5.
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1 
        return (loss, reg_weight * loss_iou, loss_obj, loss_cls, loss_l1, num_fg/max(num_gts, 1))

        

    def decode_outputs(self, outputs, dtype):
        grids = [] 
        strides = [] 
        for (height, width), stride in zip(self.height_width, self.s_list):
            yv, xv = torch.meshgrid([torch.arange(height), torch.arange(width)])
            grid = torch.stack((xv, yv), 2).view(1,-1,2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))
        
        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)
        outputs[..., :2] = (outputs[..., :2] + grids) * strides 
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides 
        return outputs

    @torch.no_grad()
    def get_assignments(self,batch_idx, num_gt, num_anchor_all, gt_bboxes_per_image, gt_classes, bboxes_preds_per_image, expanded_strides, 
                            x_offsets, y_offsets, cls_preds, bbox_preds, obj_preds, labels, imgs, mode='cpu'):
        pass


    def get_output_and_grid(self, output, i, stride, dtype):
        grid = self.grids[i]
        batch_size = output.shape[0] 
        num_channels = 4 + 1 + self.num_classes 
        height, width = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(height), torch.arange(width)])
            grid = torch.stack((xv, yv), 2).view(1, 1, height, width, 2).type(dtype)
            self.grids[i] = grid 
        output = output.view(batch_size, self.num_anchor, num_channels, height, width)
        output = output.permute(0, 1, 3, 4, 2).reshape(batch_size, self.num_anchor * height * width, -1)
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride 
        return output, grid

