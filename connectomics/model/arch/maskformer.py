# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from connectomics.model.loss.maskformer_criterion import SetCriterion
from connectomics.model.loss.matcher import HungarianMatcher, Point_HungarianMatcher
import numpy as np
from torch.cuda.amp import autocast


@META_ARCH_REGISTRY.register()
class MaskFormer(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        test_threshold,
        dataset_name
    ):

        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image
        self.test_threshold = test_threshold
        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference
        self.dataset_name = dataset_name


    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        reid_weight_query = cfg.MODEL.MASK_FORMER.REID_WEIGHT_QUERY
        reid_weight_mask = cfg.MODEL.MASK_FORMER.REID_WEIGHT_MASK
        refpoints_weight = cfg.MODEL.MASK_FORMER.REF_POINTS_WEIGHT
        sem_weight = cfg.MODEL.MASK_FORMER.SEM_WEIGHT
        emb_weight = cfg.MODEL.MASK_FORMER.EMB_WEIGHT

        # building criterion
        matcher = Point_HungarianMatcher(
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_mask": mask_weight, "loss_dice": dice_weight}
        
        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        
        # emb_weight = 2.0
        weight_dict["loss_emb"] = emb_weight

        weight_dict["loss_reid_query"] = reid_weight_query
        weight_dict["loss_reid_query_aux"] = reid_weight_query*1.5
        
        weight_dict["loss_reid_mask"] = reid_weight_mask


        weight_dict.update({"loss_refpoints": refpoints_weight})
        for i in range(dec_layers - 1):
            if i!=0:
                weight_dict.update({f"loss_refpoints_{i}": refpoints_weight})
        
        if cfg.MODEL.MASK_FORMER.SEMANTIC_LOSS_ON:
            weight_dict.update({f"loss_sem": sem_weight})
            losses = ["masks", 'refpoints', 'reid_query', 'reid_mask', 'sem', 'embedding']
        else:
            losses = ["masks", 'refpoints', 'reid_query', 'reid_mask', 'embedding']


        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        dataset_name = cfg.DATASET.DATA_TYPE
        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            'test_threshold': cfg.TEST.THRESHOLD,
            'dataset_name':dataset_name
        }

    # @property
    # def device(self):
    #     return self.pixel_std.device

    def forward(self, volume, targets=None, train=False):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of self.training
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        features = self.backbone(volume)
        
        # if self.training:
        if train:
            outputs, mask_features = self.sem_seg_head(features, targets, criterion=self.criterion)
            # mask classification target

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets, mask_features)
            # losses = self.criterion(outputs, targets)
            # import pdb; pdb.set_trace()
            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses

        else:
            outputs, _ = self.sem_seg_head(features)
            mask_pred_results = outputs["pred_masks"]
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(volume.shape[-2], volume.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs
            
            processed_results = []
            processed_boundary_results = []
            image_size = volume.shape[-2:]
            height = image_size[0]
            width = image_size[1]
            # import pdb; pdb.set_trace()
            for mask_pred_result in mask_pred_results:  
                # processed_results.append({})
                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )

                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    pre = r
                    if pre.shape[0] == 2:
                        pre = r[1][None,:]
                    processed_results.append(pre)

                # instance segmentation inference
                if self.instance_on:
                    instance_r, boundary_r = retry_if_cuda_oom(self.instance_inference)(mask_pred_result)
                    # processed_results[-1]["instances"] = instance_r
                    processed_results.append(instance_r)
                    if not boundary_r == None:
                        processed_boundary_results.append(boundary_r)

            output = torch.cat(processed_results)
            boundary_output = torch.cat(processed_boundary_results) if not len(processed_boundary_results)== 0 else None

            return output, boundary_output

            # return SBD, diffFG

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def instance_inference(self, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]
        post_process1 = post_process2=False
        if self.dataset_name == 'CVPPP':
            post_process1 = True
        if self.dataset_name == 'BBBC':
            post_process2 = True
        if post_process1:
            mask_pred = mask_pred.sigmoid().float()
            threshold = 0.69 # cvppp
            pred_masks = (mask_pred > threshold).float()

            areas = torch.tensor([pred_masks[num].sum() for num in range(pred_masks.shape[0])])
            valid_idxs = (areas>40)
            pred_masks = pred_masks[valid_idxs]

            import imageio as io
            # io.volsave('query_seg.tif', pred_masks.cpu().numpy())
            # import pdb; pdb.set_trace()

            pred_masks = mask_post(pred_masks, thres1=0.5, thres2=0.6, bd_flag=True)

            areas = torch.tensor([pred_masks[num].sum() for num in range(pred_masks.shape[0])])
            pre_scores = (areas / areas.max()).to(pred_masks)
            # pre_scores = (areas.max() / areas).to(pred_masks)
            nms_iou_threshold = 0.72
            pred_masks = mask_nms(pred_masks, pre_scores, thres=nms_iou_threshold)
            
            # io.volsave('query_seg_filter.tif', pred_masks.cpu().numpy())
            
            areas = torch.tensor([pred_masks[num].sum() for num in range(pred_masks.shape[0])])
            sorted_idxs = torch.argsort(areas).tolist()
            # sorted_idxs = torch.argsort(areas).tolist()[::-1]
            # print('num_inst', len(sorted_idxs))
            pred_masks = pred_masks[sorted_idxs]
            mask_scores = torch.cat([torch.zeros((1,pred_masks.shape[-2],pred_masks.shape[-1])).to(pred_masks), pred_masks])
            prd_result = torch.argmax(mask_scores, axis = 0).to(torch.int16)
            io.imsave('prd_result.png',prd_result.cpu().numpy())
            import pdb; pdb.set_trace()

        elif post_process2:
            mask_pred = mask_pred.sigmoid().float()
            threshold = 0.05 # cvppp
            pred_masks = (mask_pred > threshold).float()
            
            areas = torch.tensor([pred_masks[num].sum() for num in range(pred_masks.shape[0])])
            valid_idxs = (areas>40)
            pred_masks = pred_masks[valid_idxs]

            import imageio as io
            # io.volsave('query_seg.tif', pred_masks.cpu().numpy())

            pred_masks = mask_post(pred_masks, thres1=0.15, thres2=0.25)
            
            # io.volsave('query_seg_filter1.tif', pred_masks.cpu().numpy())

            # areas = torch.tensor([pred_masks[num].sum() for num in range(pred_masks.shape[0])])
            # pre_scores = (areas / areas.max()).to(pred_masks)
            # # pre_scores = (areas.max() / areas).to(pred_masks)
            # nms_iou_threshold = 0.5
            # pred_masks = mask_nms(pred_masks, pre_scores, thres=nms_iou_threshold)
            
            # io.volsave('query_seg_filter2.tif', pred_masks.cpu().numpy())
            
            areas = torch.tensor([pred_masks[num].sum() for num in range(pred_masks.shape[0])])
            sorted_idxs = torch.argsort(areas).tolist()
            # sorted_idxs = torch.argsort(areas).tolist()[::-1]
            pred_masks = pred_masks[sorted_idxs]

            # areas = torch.tensor([pred_masks[num].sum() for num in range(pred_masks.shape[0])])
            # print('areas',areas)

            mask_scores = torch.cat([torch.zeros((1,pred_masks.shape[-2],pred_masks.shape[-1])).to(pred_masks), pred_masks])
            prd_result = torch.argmax(mask_scores, axis = 0).to(torch.int16)
            # import pdb; pdb.set_trace()

        boundary_mask = None

        return prd_result[None,:].to(torch.int16), boundary_mask


def comput_mmi(area_a,area_b,intersect):
    EPS=torch.tensor(0.00001)
    if area_a==0 or area_b==0:
        area_a+=EPS
        area_b+=EPS
    return torch.max(intersect/area_a,intersect/area_b)


def mask_nms(masks, scores, thres=0.3):
    keep=[]
    order=torch.argsort(scores).tolist()[::-1]
    nums=masks.shape[0]
    suppressed=np.zeros((nums), dtype=np.int)

    # 循环遍历
    for i in range(nums):
        idx=order[i]
        if suppressed[idx]==1:
            continue
        keep.append(idx)
        mask_a = masks[idx]
        area_a = mask_a.sum()
        
        for j in range(i,nums):
            idx_j=order[j]
            if suppressed[idx_j]==1:
                continue
            mask_b = masks[idx_j]
            area_b = mask_b.sum()

            # 获取两个文本的相交面积
            merge_mask = mask_a*mask_b
            area_intersect = merge_mask.sum()

            #计算MMI
            mmi=comput_mmi(area_a,area_b,area_intersect)
            # print("area_a:{},area_b:{},inte:{},mmi:{}".format(area_a,area_b,area_intersect,mmi))

            if mmi >= thres:
                suppressed[idx_j] = 1

    return masks[keep]

def dice_for(inputs):
    inputs = inputs.flatten(1)
    
    numerator = inputs @ inputs.transpose(-2, -1)
    inputs = inputs.sum(-1)
    inputs_1 = inputs.view(inputs.size(0), 1)
    inputs_2 = inputs.view(1, inputs.size(0))
    denominator = inputs_1 + inputs_2
    res = (2 * numerator + 1) / (denominator + 1)
    return res

def mask_post(inst_masks, thres1=0.63, thres2=0.5, bd_flag=False):
    dice_mask = dice_for(inst_masks)
    query_num = dice_mask.shape[0]
    clutering_list = []
    valid_clutering_list= []
    for i in range(query_num):
        if i in clutering_list:
            continue
        dice_dice_i = dice_mask[i]
        cluter = torch.where(dice_dice_i>thres1)[0]
        cluter = cluter.tolist()
        clutering_list += cluter

        valid_clutering_list.append(cluter)
    # import pdb; pdb.set_trace()
    query_masks = []
    for i in range(len(valid_clutering_list)):
        query_ids = valid_clutering_list[i]
        query_mask_to_merge = inst_masks[query_ids]

        query_mask_merged = torch.mean(query_mask_to_merge, dim=0)
        if bd_flag:
            query_mask_merged = (query_mask_merged>thres2).float()

        query_masks.append(query_mask_merged)

    
    final_query_masks = torch.stack(query_masks)
    return final_query_masks
