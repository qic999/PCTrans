# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast

from detectron2.projects.point_rend.point_features import point_sample


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
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
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
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
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


class Point_HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_mask: float = 1, cost_dice: float = 1, num_points: int = 0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

        assert cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

        self.num_points = num_points

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        # import pdb; pdb.set_trace()
        bs, num_queries = outputs["pred_masks"].shape[:2]
        indices = []

        # Iterate through batch size
        for b in range(bs):
            out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["masks"].to(out_mask)

            out_mask = out_mask[:, None]
            tgt_mask = tgt_mask[:, None]
            # import pdb
            # pdb.set_trace()
            # tgt_mask = F.interpolate(tgt_mask, size=out_mask.shape[-2:], mode="nearest")
            
            # weight_mask = self.weight_binary_ratio(tgt_mask)
            # weight_mask = weight_mask.view(-1,1,tgt_mask.shape[-2],tgt_mask.shape[-1])
            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)

            # get gt labels
            tgt_mask = point_sample(
                tgt_mask,
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            out_mask = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)
            
            # weight_mask = point_sample(
            #     weight_mask,
            #     point_coords.repeat(weight_mask.shape[0], 1, 1),
            #     align_corners=False,
            # ).squeeze(1)
            

            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                # weight_mask = weight_mask.float()
                # Compute the focal loss between masks
                cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)

                # Compute the dice loss betwen masks
                cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)
            
            # Final cost matrix
            C = (
                self.cost_mask * cost_mask
                + self.cost_dice * cost_dice
            )
            # C = torch.clip(C, min=-1e6, max=1e6)
            C = C.reshape(num_queries, -1).cpu()

            tmp = linear_sum_assignment(C)
            # tmp = linear_sum_assignment_with_inf(C)
            indices.append(tmp)
            del out_mask, tgt_mask
            torch.cuda.empty_cache()
            
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]
        
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
    
    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)

import numpy as np
def linear_sum_assignment_with_inf(cost_matrix):
    cost_matrix = np.asarray(cost_matrix)
    min_inf = np.isneginf(cost_matrix).any()
    max_inf = np.isposinf(cost_matrix).any()
    if min_inf and max_inf:
        raise ValueError("matrix contains both inf and -inf")

    if min_inf or max_inf:
        values = cost_matrix[~np.isinf(cost_matrix)]
        m = values.min()
        M = values.max()
        n = min(cost_matrix.shape)
        # strictly positive constant even when added
        # to elements of the cost matrix
        positive = n * (M - m + np.abs(M) + np.abs(m) + 1)
        if max_inf:
            place_holder = (M + (n - 1) * (M - m)) + positive
        if min_inf:
            place_holder = (m + (n - 1) * (m - M)) - positive

        cost_matrix[np.isinf(cost_matrix)] = place_holder
    return linear_sum_assignment(cost_matrix)

def batch_sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    prob = inputs.sigmoid()
    focal_pos = ((1 - prob) ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    focal_neg = (prob ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )
    if alpha >= 0:
        focal_pos = focal_pos * alpha
        focal_neg = focal_neg * (1 - alpha)

    loss = torch.einsum("nc,mc->nm", focal_pos, targets) + torch.einsum(
        "nc,mc->nm", focal_neg, (1 - targets)
    )

    return loss / hw

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_mask: float = 1, cost_dice: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_masks"].shape[:2]

        # Work out the mask padding size
        masks = [v["masks"] for v in targets]
        h_max = max([m.shape[1] for m in masks])
        w_max = max([m.shape[2] for m in masks])

        indices = []

        # Iterate through batch size
        for b in range(bs):
            out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]

            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["masks"].to(out_mask)

            # Downsample gt masks to save memory
            tgt_mask = F.interpolate(tgt_mask[:, None], size=out_mask.shape[-2:], mode="nearest")

            weight_mask = self.weight_binary_ratio(tgt_mask)
            
            # print('weight_mask',weight_mask.shape) # [64, 1, 128, 128]
            # Flatten spatial dimension
            out_mask = out_mask.flatten(1)  # [batch_size * num_queries, H*W]
            tgt_mask = tgt_mask[:, 0].flatten(1)  # [num_total_targets, H*W]
            # print('weight_mask',weight_mask.shape) # [64, 16384]

            # Compute the focal loss between masks
            # cost_mask = batch_sigmoid_focal_loss(out_mask, tgt_mask)
            cost_mask = batch_sigmoid_ce_loss(out_mask, tgt_mask, weight_mask)

            # Compute the dice loss betwen masks
            cost_dice = batch_dice_loss(out_mask, tgt_mask)

            # Final cost matrix
            C = (
                self.cost_mask * cost_mask
                + self.cost_dice * cost_dice
            )
            C = C.reshape(num_queries, -1).cpu()

            indices.append(linear_sum_assignment(C))
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

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

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)