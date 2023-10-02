# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
from cmath import pi
import logging

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list
from connectomics.model.loss import discriminative_loss
from fvcore.nn import sigmoid_focal_loss_jit

def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
        # weight_mask: torch.Tensor,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    # loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none", weight=weight_mask)
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.mean(1).sum() / num_masks

sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule

# def sigmoid_focal_loss(inputs:torch.Tensor, targets:torch.Tensor, num_masks:float, alpha: float = 0.25, gamma: float = 2):
#     """
#     Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
#     Args:
#         inputs: A float tensor of arbitrary shape.
#                 The predictions for each example.
#         targets: A float tensor with the same shape as inputs. Stores the binary
#                  classification label for each element in inputs
#                 (0 for the negative class and 1 for the positive class).
#         alpha: (optional) Weighting factor in range (0,1) to balance
#                 positive vs negative examples. Default = -1 (no weighting).
#         gamma: Exponent of the modulating factor (1 - p_t) to
#                balance easy vs hard examples.
#     Returns:
#         Loss tensor
#     """
#     prob = inputs.sigmoid()
#     ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
#     p_t = prob * targets + (1 - prob) * (1 - targets)
#     loss = ce_loss * ((1 - p_t) ** gamma)

#     if alpha >= 0:
#         alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
#         loss = alpha_t * loss

#     return loss.mean(1).sum() / num_masks

def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    @torch.no_grad()
    def weight_binary_ratio(self, label):
        # import pdb
        # pdb.set_trace()
        min_ratio = 5e-2
        # label[:,...] = (label[:,...] != 0).to(torch.float64)  # foreground
        label = label.flatten(1)
        ww = torch.sum(label, dim=1) / label.shape[1]
        ww = torch.clip(ww, min=min_ratio, max=1-min_ratio)
        weight_factor = torch.max(ww, 1-ww)/torch.min(ww, 1-ww)

        # Case 1 -- Affinity Map
        # In that case, ww is large (i.e., ww > 1 - ww), which means the high weight
        # factor should be applied to background pixels.

        # Case 2 -- Contour Map
        # In that case, ww is small (i.e., ww < 1 - ww), which means the high weight
        # factor should be applied to foreground pixels.

        if (ww > 1-ww).any() == True:
            # switch when foreground is the dominate class
            label[ww > 1-ww] = 1 - label[ww > 1-ww]
        weight = weight_factor[:,None]*label + (1-label)
        return weight.to(torch.float32)

    def loss_masks_mask_(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        # src_masks = F.interpolate(
        #     src_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False
        # )
        # src_masks = src_masks[:, 0].flatten(1)
        # print('src_masks',src_masks.shape)
        # print('target_masks',target_masks.shape)
        # target_masks = F.interpolate(
        #     target_masks[:, None], size=src_masks.shape[-2:], mode="nearest")
        # print(torch.unique(target_masks))
        weight_mask = self.weight_binary_ratio(target_masks)

        # [40, 512, 512]
        # [40, 262144]    512*512=262144
        src_masks = src_masks.flatten(1)
        target_masks = target_masks.flatten(1)
        # print('src_masks',src_masks.shape)
        # print('target_masks',target_masks.shape)
        # target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_ce_loss_jit(src_masks, target_masks, num_masks, weight_mask),
            "loss_dice": dice_loss_jit(src_masks, target_masks, num_masks),
        }
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # import pdb; pdb.set_trace()
        # import imageio as io
        # import numpy as np
        # io.volsave('src_masks_logits.tif', src_masks.detach().cpu().numpy())
        # io.volsave('src_masks.tif', (src_masks.detach().cpu().sigmoid()>0.5).numpy().astype(np.uint8)*255)
        # io.volsave('target_masks.tif', target_masks.cpu().numpy())

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]   # [98, 1, 128, 128]
        target_masks = target_masks[:, None]   # [98, 1, 512, 512]
        # print('target_masks',target_masks.shape)
        # print('src_masks',src_masks.shape)
        # target_masks = F.interpolate(target_masks, size=src_masks.shape[-2:], mode="nearest")
        # weight_masks = self.weight_binary_ratio(target_masks)
        # weight_masks = weight_masks.view(-1,1,target_masks.shape[-2],target_masks.shape[-1])
        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)
            
            # point_weights = point_sample(
            #     weight_masks,
            #     point_coords,
            #     align_corners=False,
            # ).squeeze(1)
            

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        # del weight_masks
        return losses


    def loss_embedding(self, emb, targets, alpha=1, beta=1, gama=0.001):
        import imageio as io
        import numpy as np
        gt = []
        for target in targets:
            # nearest | linear | bilinear | bicubic | trilinear
            down_gt = F.interpolate(target['masks'][:,None].to(torch.float), size=[emb.shape[2], emb.shape[3]], mode="nearest")
            down_gt = torch.cat([torch.zeros((1,1,down_gt.shape[-2],down_gt.shape[-1])).to(down_gt), down_gt])
            down_gt = torch.argmax((down_gt[:,0]>0).to(torch.int16), axis = 0)
            gt.append(down_gt)
        targets = torch.stack(gt)

        # import pdb; pdb.set_trace()
        # io.imsave('gt_after.png',targets[0].cpu().numpy().astype(np.int16))
        
        # from sklearn.decomposition import PCA
        # import numpy as np
        # import imageio as io
        # pca = PCA(n_components=3)
        # x_emb = np.transpose(emb[0].detach().cpu().numpy(), [1, 2, 0])
        # h,w,c = x_emb.shape
        # x_emb = x_emb.reshape(-1, c)
        # new_emb = pca.fit_transform(x_emb)
        # new_emb = new_emb.reshape(h, w, 3)
        # io.imsave('pixel_emb.png',new_emb)
        # import pdb; pdb.set_trace()

        emb_loss = discriminative_loss(emb, targets, alpha=alpha, beta=beta, gama=gama)
        return {'loss_emb': emb_loss}

    def loss_reid_query(self, outputs, targets, indices, num_masks):
        # import pdb; pdb.set_trace()
        qd_items = outputs['pred_qd_query']
        contras_loss = 0
        aux_loss = 0
        if len(qd_items) == 0:
            losses = {'loss_reid_query': 0,
                   'loss_reid_query_aux':  0 }
            return losses
        for qd_item in qd_items:
            pred = qd_item['contrast'].permute(1,0) # [1, 271]
            pred = pred/2.0
            label = qd_item['label'].unsqueeze(0) # [1, 271]
            # contrastive loss
            pos_inds = (label == 1) # [1, 271]
            neg_inds = (label == 0) # [1, 271]
            pred_pos = pred * pos_inds.float() # [1, 271]
            pred_neg = pred * neg_inds.float() # [1, 271]
            # use -inf to mask out unwanted elements.
            pred_pos[neg_inds] = pred_pos[neg_inds] + float('inf')
            pred_neg[pos_inds] = pred_neg[pos_inds] + float('-inf')

            _pos_expand = torch.repeat_interleave(pred_pos, pred.shape[1], dim=1) # [1, 73441]
            _neg_expand = pred_neg.repeat(1, pred.shape[1]) # [1, 73441]
            # [bz,N], N is all pos and negative samples on reference frame, label indicate it's pos or negative
            x = torch.nn.functional.pad((_neg_expand - _pos_expand), (0, 1), "constant", 0) # [1, 73441]
            contras_loss += torch.logsumexp(x, dim=1)

            aux_pred = qd_item['aux_consin'].permute(1,0)
            aux_label = qd_item['aux_label'].unsqueeze(0)

            aux_loss += (torch.abs(aux_pred - aux_label)**2).mean()

        losses = {'loss_reid_query': contras_loss.sum()/len(qd_items),
                   'loss_reid_query_aux':  aux_loss/len(qd_items) }

        return losses
    
    def loss_reid_mask(self, outputs, targets, indices, num_masks):
        # import pdb; pdb.set_trace()
        qd_items = outputs['pred_qd_mask']
        contras_loss = 0
        if len(qd_items) == 0:
            losses = {'loss_reid_mask': 0}
            return losses
        for qd_item in qd_items:
            pred = qd_item['contrast'].permute(1,0) # [1, 271]
            pred = pred/0.5
            label = qd_item['label'].unsqueeze(0) # [1, 271]
            # contrastive loss
            pos_inds = (label == 1) # [1, 271]
            neg_inds = (label == 0) # [1, 271]
            pred_pos = pred * pos_inds.float() # [1, 271]
            pred_neg = pred * neg_inds.float() # [1, 271]
            # use -inf to mask out unwanted elements.
            pred_pos[neg_inds] = pred_pos[neg_inds] + float('inf')
            pred_neg[pos_inds] = pred_neg[pos_inds] + float('-inf')

            _pos_expand = torch.repeat_interleave(pred_pos, pred.shape[1], dim=1) # [1, 73441]
            _neg_expand = pred_neg.repeat(1, pred.shape[1]) # [1, 73441]
            # [bz,N], N is all pos and negative samples on reference frame, label indicate it's pos or negative
            x = torch.nn.functional.pad((_neg_expand - _pos_expand), (0, 1), "constant", 0) # [1, 73441]
            contras_loss += torch.logsumexp(x, dim=1)

        losses = {'loss_reid_mask': contras_loss.sum()/len(qd_items)}
        return losses
    
    def loss_refpoints(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'reference_points' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs['reference_points'][idx]
        target_points = torch.cat([t['center_points'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_points = target_points.flatten(1)
        # import pdb; pdb.set_trace()
        loss_refpoints = F.l1_loss(src_points, target_points, reduction='none')

        losses = {}
        losses['loss_refpoints'] = loss_refpoints.sum() / num_masks
        return losses
    
    def loss_sem(self, outputs, targets):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'sem_mask' in outputs
        logits_pred = outputs['sem_mask']
        semantic_targets = torch.stack([target['fg_masks'] for target in targets])
        # resize target to reduce memory
        out_stride=8
        semantic_targets = semantic_targets[:, None, out_stride // 2::out_stride, out_stride // 2::out_stride]
        one_hot = semantic_targets.float()
        num_pos = (one_hot > 0).sum().float().clamp(min=1.0)
        # import pdb; pdb.set_trace()
        
        loss_sem = sigmoid_focal_loss_jit(
                logits_pred, one_hot, 
                alpha=0.25,
                gamma=2.0,
                reduction="sum",
            )/num_pos
        
        losses = {}
        losses['loss_sem'] = loss_sem
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'masks': self.loss_masks,
            'reid_query': self.loss_reid_query, 
            'reid_mask': self.loss_reid_mask, 
            # 'reid_region': self.loss_reid_region,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets, pixel_embedding=None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        # indices = self.matcher(outputs_without_aux, targets)
        indices = outputs['indices_list'][-1]

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        
        losses = {}
        for loss in self.losses:
            # print(loss)
            if loss == 'embedding':
                losses.update(self.loss_embedding(pixel_embedding, targets))
            elif loss == 'refpoints':
                losses.update(self.loss_refpoints(outputs, targets, indices, num_masks))
            elif loss == 'sem':
                losses.update(self.loss_sem(outputs, targets))
            else:
                losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))


        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                # indices = self.matcher(aux_outputs, targets)
                indices = outputs['indices_list'][i]
                for loss in self.losses:
                    if 'reid' in loss:
                        continue
                    if loss == 'sem':
                        continue
                    if not (loss == 'embedding' or loss == 'refpoints'):
                        l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                        l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                        losses.update(l_dict)
                    if i != 0:
                        l_dict = self.loss_refpoints(outputs["aux_reference_points"][i-1], targets, indices, num_masks)
                        l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                        losses.update(l_dict)

        return losses

    # def __repr__(self):
    #     head = "Criterion " + self.__class__.__name__
    #     body = [
    #         "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
    #         "losses: {}".format(self.losses),
    #         "weight_dict: {}".format(self.weight_dict),
    #         "num_classes: {}".format(self.num_classes),
    #         "eos_coef: {}".format(self.eos_coef),
    #         "num_points: {}".format(self.num_points),
    #         "oversample_ratio: {}".format(self.oversample_ratio),
    #         "importance_sample_ratio: {}".format(self.importance_sample_ratio),
    #     ]
    #     _repr_indent = 4
    #     lines = [head] + [" " * _repr_indent + line for line in body]
    #     return "\n".join(lines)
