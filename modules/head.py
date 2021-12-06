import torch
import torch.nn as nn 
import torch.nn.functional as F
import math 

from loguru import logger 

from ..utils import bboxes_iou

from .losses import IOULoss
from .blocks import Conv, DepthwiseConv 

class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width=1.0, s_list=[8, 16, 32], in_c_list=[256, 512, 1024], act='silu', depthwise=False):
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

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True) 

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

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
        x_offsets = torch.cat(x_offsets, dim=1) # [1, num_anchor_all]
        y_offsets = torch.cat(y_offsets, dim=1) # [1, num_anchor_all]
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

    @torch.no_grad()
    def get_assignments(self,batch_idx, num_gt, num_anchor_all, gt_bboxes_per_image, gt_classes, bboxes_preds_per_image, expanded_strides, 
                            x_offsets, y_offsets, cls_preds, bbox_preds, obj_preds, labels, imgs, mode='cpu'):
        if mode=='cpu':
            print(f'----------CPU Mode for this batch----------')
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_offsets = x_offsets.cpu()
            y_offsets = y_offsets.cpu()

        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(gt_bboxes_per_image, expanded_strides, x_offsets, y_offsets, num_anchor_all, num_gt)
        
        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_bboxes_anchor = bboxes_preds_per_image.shape[0]

        if mode=='cpu':
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()
        
        pairwise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)

        gt_cls_per_image = (F.one_hot(gt_classes.to(torch.int64), self.num_classes).float().unsqueeze(1).repeat(1, num_in_bboxes_anchor, 1))
        pairwise_ious_loss = -torch.log(pairwise_ious + 1e-8)

        if mode == 'cpu':
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()
        
        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_() * \
                          obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_())
            pairwise_cls_loss = F.binary_cross_entropy(cls_preds_.sqrt_(), gt_cls_per_image, reduction='none').sum(-1)
        del cls_preds_

        cost = (pairwise_cls_loss + 3.0 * pairwise_ious_loss + 100000.0 * (~is_in_boxes_and_center))
        
        (num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds) = self.dynamic_k_match(cost, pairwise_ious, gt_classes, num_gt, fg_mask)
        del pairwise_cls_loss, cost, pairwise_ious, pairwise_ious_loss
        if mode=='cpu':
            gt_matched_classes = gt_matched_classes.cuda() 
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()
        return (gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg)

    def dynamic_k_match(self, cost, pairwise_ious, gt_classes, num_gt, fg_mask):
        # Dynamic K
        match_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        ious_in_boxes_matrix = pairwise_ious
        num_candidates_k = min(10, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, num_candidates_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        dynamic_ks = dynamic_ks.tolist()
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx], largest=False)
            match_matrix[gt_idx][pos_idx] = 1
        del topk_ious, dynamic_ks, pos_idx

        anchor_match_gt = match_matrix.sum(0)
        if (anchor_match_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_match_gt > 1], dim=0)
            match_matrix[:, anchor_match_gt > 1] *= 0
            match_matrix[cost_argmin, anchor_match_gt > 1] = 1
        fg_mask_inboxes = match_matrix.sum(0) > 0
        num_fg = fg_mask_inboxes.sum().item() 

        fg_mask[fg_mask.clone()] = fg_mask_inboxes
        matched_gt_inds = match_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (match_matrix * pairwise_ious).sum(0)[fg_mask_inboxes]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds

    def get_in_boxes_info(self, gt_bboxes_per_image, expanded_strides, x_offsets, y_offsets, num_anchor_all, num_gt):
        expanded_strides_per_img = expanded_strides[0]
        x_offsets_per_img = x_offsets[0] * expanded_strides_per_img
        y_offsets_per_img = y_offsets[0] * expanded_strides_per_img
        x_centers_per_img = ((x_offsets_per_img + expanded_strides_per_img / 2).unsqueeze(0).repeat(num_gt, 1)) # [num_anchor] -> [num_gt, num_anchor]
        y_centers_per_img = ((y_offsets_per_img + expanded_strides_per_img / 2).unsqueeze(0).repeat(num_gt, 1))

        gt_bboxes_per_image_l = ((gt_bboxes_per_image[:, 0] - gt_bboxes_per_image[:, 2] / 2).unsqueeze(1).repeat(1, num_anchor_all))
        gt_bboxes_per_image_r = ((gt_bboxes_per_image[:, 0] + gt_bboxes_per_image[:, 2] / 2).unsqueeze(1).repeat(1, num_anchor_all))
        gt_bboxes_per_image_t = ((gt_bboxes_per_image[:, 1] - gt_bboxes_per_image[:, 3] / 2).unsqueeze(1).repeat(1, num_anchor_all))
        gt_bboxes_per_image_b = ((gt_bboxes_per_image[:, 1] + gt_bboxes_per_image[:, 3] / 2).unsqueeze(1).repeat(1, num_anchor_all))

        b_l = x_centers_per_img - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_img
        b_t = y_centers_per_img - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_img 
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], dim=2)

        is_in_bboxes = bbox_deltas.min(dim=-1).values > 0
        is_in_bboxes_all = is_in_bboxes.sum(dim=0) > 0        
        # in fixed center

        center_radius = 2.5

        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(1, num_anchor_all) - center_radius * expanded_strides_per_img.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(1, num_anchor_all) + center_radius * expanded_strides_per_img.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(1, num_anchor_all) - center_radius * expanded_strides_per_img.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(1, num_anchor_all) + center_radius * expanded_strides_per_img.unsqueeze(0)

        c_l = x_centers_per_img - gt_bboxes_per_image_l 
        c_r = gt_bboxes_per_image_r - x_centers_per_img
        c_t = x_centers_per_img - gt_bboxes_per_image_b
        c_b = gt_bboxes_per_image_b - x_centers_per_img 

        center_deltas = torch.stack([c_l, c_r, c_t, c_b], dim=2)
        is_in_centers = center_deltas.min(dim=-1).values > 0 
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers 
        is_in_bboxes_anchor = is_in_bboxes_all | is_in_centers_all

        is_in_bboxes_and_center = (is_in_bboxes[:, is_in_bboxes_anchor] & is_in_centers[:, is_in_bboxes_anchor])
        return is_in_bboxes_anchor, is_in_bboxes_and_center
