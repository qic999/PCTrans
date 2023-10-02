from __future__ import print_function, division
from typing import Optional
import warnings

import os
import time
import math
import GPUtil
import numpy as np
from yacs.config import CfgNode
import h5py
import waterz
import torch
from torch.cuda.amp import autocast, GradScaler
import gc
from .base import TrainerBase
from .solver import *
from ..model import *
from ..utils.monitor import build_monitor
from ..data.augmentation import build_train_augmentor, TestAugmentor
from ..data.dataset import build_dataloader, get_dataset
from ..data.dataset.build import _get_file_list
from ..data.utils import build_blending_matrix, writeh5, relabel
from ..data.utils import get_padsize, array_unpad
from lib.evaluate.CVPPP_evaluate import BestDice, AbsDiffFGLabels, SymmetricBestDice, SymmetricBestDice_max
from connectomics.inference.evaluation.metrics_bbbc import agg_jc_index, pixel_f1, remap_label, get_fast_pq
import random

class Trainer(TrainerBase):
    r"""Trainer class for supervised learning.

    Args:
        cfg (yacs.config.CfgNode): YACS configuration options.
        device (torch.device): model running device. GPUs are recommended for model training and inference.
        mode (str): running mode of the trainer (``'train'`` or ``'test'``). Default: ``'train'``
        rank (int, optional): node rank for distributed training. Default: `None`
        checkpoint (str, optional): the checkpoint file to be loaded. Default: `None`
    """

    def __init__(self,
                 cfg: CfgNode,
                 device: torch.device,
                 mode: str = 'train',
                 rank: Optional[int] = None,
                 checkpoint: Optional[str] = None):
        self.init_basics(cfg, device, mode, rank)

        self.model = build_model(self.cfg, self.device, rank)
        if self.mode == 'train':
            self.optimizer = build_optimizer(self.cfg, self.model)
            self.lr_scheduler = build_lr_scheduler(self.cfg, self.optimizer)
            self.scaler = GradScaler() if cfg.MODEL.MIXED_PRECESION else None
            self.start_iter = self.cfg.MODEL.PRE_MODEL_ITER
            self.update_checkpoint(checkpoint)

            # stochastic weight averaging
            if self.cfg.SOLVER.SWA.ENABLED:
                self.swa_model, self.swa_scheduler = build_swa_model(
                    self.cfg, self.model, self.optimizer)
            if not (self.cfg.DATASET.DATA_TYPE == 'CVPPP' or self.cfg.DATASET.DATA_TYPE == 'BBBC' or self.cfg.DATASET.DATA_TYPE == 'monuseg' or self.cfg.DATASET.DATA_TYPE == 'cellpose'):
                self.augmentor = build_train_augmentor(self.cfg)
            else:
                self.augmentor = None
            if not self.cfg.MODEL.ARCHITECTURE == 'MaskFormer':
                self.criterion = Criterion.build_from_cfg(self.cfg, self.device)
            if self.is_main_process:
                self.monitor = build_monitor(self.cfg)
                self.monitor.load_info(self.cfg, self.model)

            self.total_iter_nums = self.cfg.SOLVER.ITERATION_TOTAL - self.start_iter
            self.total_time = 0
        else:
            self.update_checkpoint(checkpoint)
            self.model_name = checkpoint[-14:-8]
            # self.model_name = checkpoint[-13:-8]
            # build test-time augmentor and update output filename
            # if self.cfg.MODEL.ARCHITECTURE == 'MaskFormer' or self.cfg.MODEL.ARCHITECTURE == 'unet_residual_3d':
            if self.cfg.MODEL.ARCHITECTURE == 'MaskFormer':
                self.augmentor = TestAugmentor.build_from_cfg(cfg, activation=False)
            else:
                self.augmentor = TestAugmentor.build_from_cfg(cfg, activation=True)
            if not self.cfg.DATASET.DO_CHUNK_TITLE and not self.inference_singly:
                self.test_filename = self.cfg.INFERENCE.OUTPUT_NAME
                self.test_filename = self.augmentor.update_name(self.test_filename)
            if self.cfg.DATASET.DATA_TYPE == 'CVPPP':
                self.val_loader = build_dataloader(self.cfg, None, mode='val', rank=rank)
                # self.val_loader = build_dataloader(self.cfg, None, mode='test', rank=rank)
            if self.cfg.DATASET.DATA_TYPE == 'BBBC':
                self.val_loader = build_dataloader(self.cfg, None, mode='test', rank=rank)
            if self.cfg.DATASET.DATA_TYPE == 'cellpose':
                self.val_loader = build_dataloader(self.cfg, None, mode='test', rank=rank)
            if self.cfg.DATASET.DATA_TYPE == 'monuseg':
                self.val_loader = build_dataloader(self.cfg, None, mode='test', rank=rank)
                
        self.dataset, self.dataloader = None, None
        if (self.mode == 'train') or (self.mode != 'train' and (self.cfg.DATASET.DATA_TYPE == 'CVPPP' or self.cfg.DATASET.DATA_TYPE == 'BBBC' or self.cfg.DATASET.DATA_TYPE == 'monuseg' or self.cfg.DATASET.DATA_TYPE == 'cellpose')):
            if not self.cfg.DATASET.DO_CHUNK_TITLE and not self.inference_singly:
                self.dataloader = build_dataloader(
                    self.cfg, self.augmentor, self.mode, rank=rank)
                if self.mode == 'train':
                    self.train_loader = iter(self.dataloader)
                else:
                    self.dataloader = iter(self.dataloader)
                if self.mode == 'train' and cfg.DATASET.VAL_IMAGE_NAME is not None:
                    self.val_loader = build_dataloader(
                        self.cfg, None, mode='val', rank=rank)

    def init_basics(self, *args):
        # This function is used for classes that inherit Trainer but only 
        # need to initialize basic attributes in TrainerBase.
        super().__init__(*args)

    def train(self):
        r"""Training function of the trainer class.
        """
        self.model.train()

        for i in range(self.total_iter_nums):
            iter_total = self.start_iter + i
            self.start_time = time.perf_counter()
            self.optimizer.zero_grad()

            # load data
            try:
                sample = next(self.train_loader)
            except StopIteration:
                self.train_loader = iter(self.dataloader)
            
            volume = sample.out_input
            target, weight = sample.out_target_l, sample.out_weight_l  # target -> list, len -> len(target_opt)
            self.data_time = time.perf_counter() - self.start_time
            # prediction
            volume = volume.to(self.device, non_blocking=True) # volume -> tensor
            # import imageio as io
            # io.imsave('gt_example.png',target[0][0].cpu().numpy().astype(np.uint16))
            # io.imsave('image_example.png',volume[0].cpu().numpy().transpose((1,2,0)))
            # import pdb; pdb.set_trace()
            if self.cfg.DATASET.DATA_TYPE == 'Lucchi':
                volume = volume.repeat(1,3,1,1)
            with autocast(enabled=self.cfg.MODEL.MIXED_PRECESION):
                if self.cfg.MODEL.ARCHITECTURE == 'MaskFormer':
                    # breakpoint()
                    if self.cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
                        target = self.prepare_targets(target) # target -> list, len -> batch_size
                    elif self.cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
                        if self.cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES > 2:
                            target = self.prepare_intance_targets(target, True) # target -> list, len -> batch_size
                        else:
                            target = self.prepare_intance_targets(target, False)

                    losses_vis = self.model(volume, target, True)
                    loss = sum(losses_vis.values())
                else:
                    pred = self.model(volume)
                    loss, losses_vis = self.criterion(pred, target, weight)
                    del pred

            self._train_misc(loss, volume, target, weight, iter_total, losses_vis)

        self.maybe_save_swa_model()

    def _train_misc(self, loss, volume, target, weight, iter_total, losses_vis):
        self.backward_pass(loss, iter_total)  # backward pass

        # logging and update record
        if hasattr(self, 'monitor'):
            do_vis = self.monitor.update(iter_total, loss, losses_vis,
                                         self.optimizer.param_groups[0]['lr'])
            # if do_vis:
            #     self.monitor.visualize(
            #         volume, target, pred, weight, iter_total)
            #     if torch.cuda.is_available():
            #         GPUtil.showUtilization(all=True)

        # Save model
        if (iter_total+1) % self.cfg.SOLVER.ITERATION_SAVE == 0 and (iter_total+1) >= self.cfg.SOLVER.START_SAVE:
            self.save_checkpoint(iter_total)

        # if (iter_total+1) % self.cfg.SOLVER.ITERATION_VAL == 0 and (iter_total+1) >= self.cfg.SOLVER.START_SAVE:
        #     self.validate(iter_total)

        # update learning rate
        self.maybe_update_swa_model(iter_total)
        self.scheduler_step(iter_total, loss)

        if self.is_main_process:
            self.iter_time = time.perf_counter() - self.start_time
            self.total_time += self.iter_time
            # avg_iter_time = self.total_time / (iter_total+1-self.start_iter)
            avg_iter_time = self.total_time / (iter_total+1)
            est_time_left = avg_iter_time * \
                (self.total_iter_nums+self.start_iter-iter_total-1) / 3600.0
            info = [
                '[Iteration %05d]' % iter_total, 'Data time: %.4fs,' % self.data_time,
                'Iter time: %.4fs,' % self.iter_time, 'Avg iter time: %.4fs,' % avg_iter_time,
                'Time Left %.2fh.' % est_time_left]
            print(' '.join(info))

        # Release some GPU memory and ensure same GPU usage in the consecutive iterations according to
        # https://discuss.pytorch.org/t/gpu-memory-consumption-increases-while-training/2770
        del volume, target, weight, loss, losses_vis
        torch.cuda.empty_cache()

    def prepare_targets(self, targets):
        ignore_label = self.cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        new_targets = []
        batch_size = targets[0].shape[0]
        for num in range(batch_size):
            gt_mask = targets[0][num, ...]
            # gt_mask = (targets[0][num, ...] * 255).to(torch.uint8) # torch.float32 -> torch.uint8
            # gt_mask = gt_mask - 1 # 0 (ignore) becomes 255. others are shifted by 1
            classes = torch.unique(gt_mask)
            # remove ignored region
            classes = classes[classes != ignore_label]
            gt_classes = classes.to(self.device, dtype=torch.int64) 

            masks = []
            for class_id in classes:
                masks.append(gt_mask == class_id)
            gt_masks = torch.cat([x for x in masks])
            # gt_masks = torch.stack([x for x in masks])
            gt_masks = gt_masks.to(self.device)
            new_targets.append(
                {
                    "labels": gt_classes,
                    "masks": gt_masks,
                }
            )
        return new_targets

    def prepare_intance_targets(self, targets, instance_discrimitative=False):
        ignore_label = self.cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        new_targets = []
        batch_size = targets[0].shape[0]
        for num in range(batch_size):
            # import pdb; pdb.set_trace()
            gt_mask = targets[0][num, ...]
            # gt_mask = (targets[0][num, ...] * 255).to(torch.uint8) # torch.float32 -> torch.uint8
            # gt_mask = gt_mask - 1 # 0 (ignore) becomes 255. others are shifted by 1
            # gt_mask = gt_mask + 1 # 考虑boundary的分割
            classes = torch.unique(gt_mask)
            # import pdb
            # pdb.set_trace()
            # remove ignored region
            classes = classes[classes != ignore_label]

            masks = []
            center_points=[]
            points_num = self.cfg.MODEL.MASK_FORMER.POSITION_POINTS_NUM
            for class_id in classes:
                # 中心点
                inst_mask = (gt_mask == class_id)
                hw_img = inst_mask.shape[-1]

                # inst_mask.nonzero().flip(dims=[1]) 与 torch.stack(torch.where(inst_mask == 1)[::-1], dim=1) 等价，前者更快
                pos_xy = inst_mask.nonzero().flip(dims=[1]) # torch.where(inst_mask == 1) 类似于inst_mask.nonzero().flip(dims=[1])
                # import pdb; pdb.set_trace()
                if points_num == 1:
                    center_xy = torch.mean(pos_xy.to(torch.float), dim=0) / hw_img
                    center_points.append(center_xy)
                else:
                    if len(pos_xy) < points_num:
                        # print('points_num', len(pos_xy))
                        continue
                    else:
                        sample_point_xy = pos_xy[random.sample(range(len(pos_xy)), points_num-1)] / hw_img
                        center_xy = torch.mean(pos_xy.to(torch.float), dim=0) / hw_img
                        points_xy = torch.cat([sample_point_xy, center_xy[None,:]])
                        center_points.append(points_xy)
                
                masks.append(inst_mask)

            # gt_masks = torch.cat([x for x in masks])
            if len(masks) == 0:
                gt_masks = torch.zeros(size=gt_mask.shape)[None,:]
                if points_num==1:
                    gt_center_points = torch.zeros((1,2))
                elif points_num>1:
                    gt_center_points = torch.zeros((1,points_num,2))
            else:
                gt_masks = torch.stack([x for x in masks])
                gt_center_points = torch.stack(center_points)
            # print('gt_center_points',gt_center_points.shape)
            # print('gt_masks',gt_masks.shape)
            # import pdb; pdb.set_trace()
            if not instance_discrimitative:
                classes = (classes > 0).to(torch.int64)
            gt_masks = gt_masks.to(self.device)
            gt_classes = classes.to(self.device, dtype=torch.int64) 
            gt_center_points = gt_center_points.to(self.device)
            
            # import pdb; pdb.set_trace()
            # H, W = gt_mask.shape
            # orig_h = torch.as_tensor(H).to(gt_center_points)
            # orig_w = torch.as_tensor(W).to(gt_center_points)
            # scale_f = torch.stack([orig_w, orig_h], dim=0)
            # gt_center_points = gt_center_points * scale_f
            # import pdb; pdb.set_trace()
            # plot_mask(gt_mask.to("cpu"), point_coords=gt_center_points.to("cpu"))
            
            if self.cfg.MODEL.MASK_FORMER.SEMANTIC_LOSS_ON:
                fg_masks = (gt_mask>0).to(gt_masks)
                new_targets.append(
                    {
                        "labels": gt_classes,
                        "masks": gt_masks,
                        'center_points': gt_center_points,
                        'fg_masks': fg_masks
                    }
                )
            else:
                new_targets.append(
                    {
                        "labels": gt_classes,
                        "masks": gt_masks,
                        'center_points': gt_center_points,
                    }
                )
        return new_targets

    def validate(self, iter_total):
        r"""Validation function of the trainer class.
        """
        if not hasattr(self, 'val_loader'):
            return

        self.model.eval()
        with torch.no_grad():
            val_loss = 0.0
            SBD_all = 0.0
            diffFG_all = 0.0
            for i, sample in enumerate(self.val_loader):
                volume = sample.out_input
                target, weight = sample.out_target_l, sample.out_weight_l

                # prediction
                volume = volume.to(self.device, non_blocking=True)
                with autocast(enabled=self.cfg.MODEL.MIXED_PRECESION):
                    if self.cfg.MODEL.ARCHITECTURE == 'MaskFormer':
                        # import pdb
                        # pdb.set_trace()
                        if self.cfg.DATASET.DATA_TYPE == 'CVPPP':
                            seg_outputs, _ = self.model(volume)
                            seg_outputs = seg_outputs.cpu().numpy().astype(np.uint16)
                            gt_ins = target[0].cpu().numpy().astype(np.uint16)
                            for num in range(seg_outputs.shape[0]):
                                SBD = SymmetricBestDice(seg_outputs[num], gt_ins[num])
                                diffFG = AbsDiffFGLabels(seg_outputs[num], gt_ins[num])
                                diffFG_all += abs(diffFG)
                                SBD_all += SBD

                        else:
                            if self.cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
                                target = self.prepare_targets(target) # target -> list, len -> batch_size
                            elif self.cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
                                if self.cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES > 2:
                                    target = self.prepare_intance_targets(target, True) # target -> list, len -> batch_size
                                else:
                                    target = self.prepare_intance_targets(target, False)
                            losses_vis = self.model(volume, target, True)
                            loss = sum(losses_vis.values())
                            val_loss += loss
                    else:
                        pred = self.model(volume)
                        loss, _ = self.criterion(pred, target, weight)
                        val_loss += loss.data
                        del pred
        if self.cfg.DATASET.DATA_TYPE == 'CVPPP':
            # import pdb
            # pdb.set_trace()
            SBD_all = SBD_all/20.0
            diffFG_all = diffFG_all/20.0
            if hasattr(self, 'monitor'):
                self.monitor.logger.log_tb.add_scalar(
                    'SBD', SBD_all, iter_total)
                self.monitor.logger.log_tb.add_scalar(
                    'diffFG', diffFG_all, iter_total)
            del SBD, diffFG, SBD_all, diffFG_all
        else:
            val_loss = val_loss/len(self.val_loader)
            if hasattr(self, 'monitor'):
                self.monitor.logger.log_tb.add_scalar(
                    'Validation_Loss', val_loss, iter_total)

            if not hasattr(self, 'best_val_loss'):
                self.best_val_loss = val_loss

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(iter_total, is_best=True)
            del loss, val_loss
        # Release some GPU memory and ensure same GPU usage in the consecutive iterations according to
        # https://discuss.pytorch.org/t/gpu-memory-consumption-increases-while-training/2770
        # del pred, loss, val_loss
        # model.train() only called at the beginning of Trainer.train().
        self.model.train()

    def eval_cvppp(self):
        self.model.eval()
        start = time.perf_counter()
        with torch.no_grad():
            SBD_all = 0.0
            diffFG_all = 0.0
            final_seg_outputs = np.zeros((20,530,500)).astype(np.uint16)
            num_sum=0
            # import pdb; pdb.set_trace()
            for i, sample in enumerate(self.val_loader):
                print('progress: %d/%d batches, total time %.2fs' %
                    (i+1, len(self.val_loader), time.perf_counter()-start))
                volume = sample.out_input
                target, weight = sample.out_target_l, sample.out_weight_l
                # prediction
                volume = volume.to(self.device, non_blocking=True)
                fg = weight[0][0].to(self.device, non_blocking=True)
                # import pdb; pdb.set_trace()
                with autocast(enabled=self.cfg.MODEL.MIXED_PRECESION):
                    seg_outputs, _ = self.model(volume)
                    # import pdb; pdb.set_trace()
                    seg_outputs = seg_outputs * fg
                    input_data = volume[0].cpu().numpy().transpose((1,2,0))
                    seg_outputs = seg_outputs.cpu().numpy().astype(np.uint16)
                    gt_ins = target[0].cpu().numpy().astype(np.uint16)
                    # for num in range(seg_outputs.shape[0]):
                    #     seg_outputs[num,...] = merge_func(seg_outputs[num])
                    num1 = seg_outputs.shape[0]
                    final_seg_outputs[num_sum:num1+num_sum,:,:]=seg_outputs
                    num_sum+=num1
                    # io.volsave('seg_gt.tif',gt_ins)
                    # print('seg_outputs',seg_outputs.shape)
                    # print('gt_ins',gt_ins.shape)
                    
                    for num in range(seg_outputs.shape[0]):
                        print('num',num)
                        SBD = SymmetricBestDice(seg_outputs[num], gt_ins[num])
                        diffFG = AbsDiffFGLabels(seg_outputs[num], gt_ins[num])
                        print('SBD',SBD)
                        print('diffFG',diffFG)
                        diffFG_all += abs(diffFG)
                        SBD_all += SBD
        import imageio as io
        # io.imsave('seg_outputs.png',input_data)
        io.volsave('seg_outputs.tif',final_seg_outputs)
        SBD_all = SBD_all/20.0
        diffFG_all = diffFG_all/20.0
        output_txt = self.cfg.INFERENCE.OUTPUT_PATH
        with open(output_txt+"logging.txt", "a") as f:
            f.writelines(self.model_name)
            f.writelines("\n")
            f.writelines(" ".join([str(SBD_all), str(diffFG_all)]))
            f.writelines("\n")

    def test_cvppp(self):
        self.model.eval()
        start = time.perf_counter()
        final_outputs = torch.zeros(33, 530, 500)
        last_num = 0
        with torch.no_grad():
            for i, sample in enumerate(self.val_loader):
                print('progress: %d/%d batches, total time %.2fs' %
                    (i+1, len(self.val_loader), time.perf_counter()-start))
                volume = sample.out_input
                fg = sample.pos
                fg = torch.stack(fg)
                # prediction
                volume = volume.to(self.device, non_blocking=True)
                fg = fg.to(self.device, non_blocking=True)
                num = volume.shape[0]
                with autocast(enabled=self.cfg.MODEL.MIXED_PRECESION):
                    seg_outputs, _ = self.model(volume)
                    # import pdb; pdb.set_trace()
                    seg_outputs = seg_outputs * fg
                    final_outputs[last_num:last_num+num] = seg_outputs.cpu()
                last_num += volume.shape[0]
        final_outputs = final_outputs.cpu().numpy()
        for num in range(final_outputs.shape[0]):
            final_outputs[num,...] = merge_func(final_outputs[num])
        import imageio as io 
        io.volsave('out_final.tif', final_outputs)
        
        from shutil import copyfile 
        seg = final_outputs.astype(np.uint8)
        out_seg_path = 'submission.h5'
        example_path = '/data/ZCWANG007/code/submission_example.h5'
        copyfile(example_path, out_seg_path)
        fi = ['plant003','plant004','plant009','plant014','plant019','plant023','plant025','plant028','plant034',
            'plant041','plant056','plant066','plant074','plant075','plant081','plant087','plant093','plant095',
            'plant097','plant103','plant111','plant112','plant117','plant122','plant125','plant131','plant136',
            'plant140','plant150','plant155','plant157','plant158','plant160']
        f_out = h5py.File(out_seg_path, 'r+')
        for k, fn in enumerate(fi):
            data = f_out['A1']
            img = data[fn]['label'][:]
            del data[fn]['label']
            data[fn]['label'] = seg[k]
        f_out.close()

    def test_bbbc(self):
        self.model.eval()
        start = time.perf_counter()
        with torch.no_grad():
            # final_seg = torch.zeros(50, 584, 760)
            # final_gt = torch.zeros(50, 584, 760)
            final_seg = torch.zeros(50, 520, 696)
            final_gt = torch.zeros(50, 520, 696)
            last_num = 0
            aji_score = []
            dice_score = []
            f1_score = []
            pq_score = []
            # import pdb; pdb.set_trace()
            for i, sample in enumerate(self.val_loader):
                print('progress: %d/%d batches, total time %.2fs' %
                    (i+1, len(self.val_loader), time.perf_counter()-start))
                volume = sample.out_input
                gt_ins = sample.pos
                gt_ins = torch.cat(gt_ins)
                # prediction
                volume = volume.to(self.device, non_blocking=True)
                # import imageio as io
                # io.imsave('input_image.png', volume[0].permute(1,2,0).cpu().numpy())
                # import pdb; pdb.set_trace()

                with autocast(enabled=self.cfg.MODEL.MIXED_PRECESION):
                    seg_outputs, _ = self.model(volume)
                    # import pdb; pdb.set_trace()

                    num = volume.shape[0]
                    final_seg[last_num:last_num+num] = seg_outputs.cpu()
                    final_gt[last_num:last_num+num] = gt_ins
                    last_num += volume.shape[0]
        
        final_seg = final_seg.cpu().numpy().astype(np.uint16)
        final_gt = final_gt.cpu().numpy().astype(np.uint16)
        # final_seg = final_seg[:,32:552,32:728]
        # final_gt = final_gt[:,32:552,32:728]
        import imageio as io
        io.volsave('ours_bbbc_test.tif', final_seg)
        # import pdb; pdb.set_trace()
        for i in range(final_gt.shape[0]):
            gt_relabel = remap_label(final_gt[i], by_size=False)
            pred_relabel = remap_label(final_seg[i], by_size=False)
            temp_aji = agg_jc_index(gt_relabel, pred_relabel)
            temp_dice = pixel_f1(gt_relabel, pred_relabel)
            
            pq_info_cur = get_fast_pq(gt_relabel, pred_relabel, match_iou=0.5)[0]
            temp_f1 = pq_info_cur[0]
            temp_pq = pq_info_cur[2]
            
            aji_score.append(temp_aji)
            dice_score.append(temp_dice)
            f1_score.append(temp_f1)
            pq_score.append(temp_pq)
                    
        aji_score = np.asarray(aji_score)
        dice_score = np.asarray(dice_score)
        f1_score = np.asarray(f1_score)
        pq_score = np.asarray(pq_score)

        mean_aji = np.mean(aji_score)
        std_aji = np.std(aji_score)
        mean_dice = np.mean(dice_score)
        std_dice = np.std(dice_score)
        mean_f1 = np.mean(f1_score)
        std_f1 = np.std(f1_score)
        mean_pq = np.mean(pq_score)
        std_pq = np.std(pq_score)

        output_txt = self.cfg.INFERENCE.OUTPUT_PATH
        with open(output_txt+"logging.txt", "a") as f:
            f.writelines(self.model_name)
            f.writelines("\n")
            f.writelines(" ".join([str(mean_aji), str(mean_dice), str(mean_f1), str(mean_pq)]))
            f.writelines("\n")

    # -----------------------------------------------------------------------------
    # Misc functions
    # -----------------------------------------------------------------------------

    def backward_pass(self, loss, global_step):
        if self.cfg.MODEL.MIXED_PRECESION:
            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            # Backward passes under autocast are not recommended.
            # Backward ops run in the same dtype autocast chose for corresponding forward ops.
            self.scaler.scale(loss).backward()

            # scaler.step() first unscales the gradients of the optimizer's assigned params.
            # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            self.scaler.step(self.optimizer)

            # Updates the scale for next iteration.
            self.scaler.update()

        else:  # standard backward pass
            loss.backward()
            self.optimizer.step()

    def save_checkpoint(self, iteration: int, is_best: bool = False):
        r"""Save the model checkpoint.
        """
        if self.is_main_process:
            print("Save model checkpoint at iteration ", iteration)
            state = {'iteration': iteration + 1,
                     # Saving DataParallel or DistributedDataParallel models
                     'state_dict': self.model.module.state_dict(),
                     'optimizer': self.optimizer.state_dict(),
                     'lr_scheduler': self.lr_scheduler.state_dict()}

            # Saves checkpoint to experiment directory
            filename = 'checkpoint_%06d.pth.tar' % (iteration + 1)
            if is_best:
                filename = 'checkpoint_best.pth.tar'
            filename = os.path.join(self.output_dir, filename)
            torch.save(state, filename)

    def update_checkpoint(self, checkpoint: Optional[str] = None):
        r"""Update the model with the specified checkpoint file path.
        """
        if checkpoint is None:
            if self.mode == 'test':
                warnings.warn("Test mode without specified checkpoint!")
            return # nothing to load

        # load pre-trained model
        print('Load pretrained checkpoint: ', checkpoint)
        checkpoint = torch.load(checkpoint, map_location='cpu')
        print('checkpoints: ', checkpoint.keys())

        # update model weights
        if 'state_dict' in checkpoint.keys():
            pretrained_dict = checkpoint['state_dict']
            pretrained_dict = update_state_dict(
                self.cfg, pretrained_dict, mode=self.mode)
            model_dict = self.model.module.state_dict()  # nn.DataParallel

            # show model keys that do not match pretrained_dict
            if not model_dict.keys() == pretrained_dict.keys():
                warnings.warn("Module keys in model.state_dict() do not exactly "
                              "match the keys in pretrained_dict!")
                for key in model_dict.keys():
                    if not key in pretrained_dict:
                        print(key)

            # 1. filter out unnecessary keys by name
            pretrained_dict = {k: v for k,
                               v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict (if size match)
            for param_tensor in pretrained_dict:
                if model_dict[param_tensor].size() == pretrained_dict[param_tensor].size():
                    model_dict[param_tensor] = pretrained_dict[param_tensor]
            # 3. load the new state dict
            self.model.module.load_state_dict(model_dict)  # nn.DataParallel

        if self.mode == 'train' and not self.cfg.SOLVER.ITERATION_RESTART:
            if hasattr(self, 'optimizer') and 'optimizer' in checkpoint.keys():
                self.optimizer.load_state_dict(checkpoint['optimizer'])

            if hasattr(self, 'lr_scheduler') and 'lr_scheduler' in checkpoint.keys():
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                self.lr_scheduler.max_iters = self.cfg.SOLVER.ITERATION_TOTAL # update total iteration

            if hasattr(self, 'start_iter') and 'iteration' in checkpoint.keys():
                self.start_iter = checkpoint['iteration']
            

    def maybe_save_swa_model(self):
        if not hasattr(self, 'swa_model'):
            return

        if self.cfg.MODEL.NORM_MODE in ['bn', 'sync_bn']:  # update bn statistics
            for _ in range(self.cfg.SOLVER.SWA.BN_UPDATE_ITER):
                sample = next(self.dataloader)
                volume = sample.out_input
                volume = volume.to(self.device, non_blocking=True)
                with autocast(enabled=self.cfg.MODEL.MIXED_PRECESION):
                    pred = self.swa_model(volume)

        # save swa model
        if self.is_main_process:
            print("Save SWA model checkpoint.")
            state = {'state_dict': self.swa_model.module.state_dict()}
            filename = 'checkpoint_swa.pth.tar'
            filename = os.path.join(self.output_dir, filename)
            torch.save(state, filename)

    def maybe_update_swa_model(self, iter_total):
        if not hasattr(self, 'swa_model'):
            return

        swa_start = self.cfg.SOLVER.SWA.START_ITER
        swa_merge = self.cfg.SOLVER.SWA.MERGE_ITER
        if iter_total >= swa_start and iter_total % swa_merge == 0:
            self.swa_model.update_parameters(self.model)

    def scheduler_step(self, iter_total, loss):
        if hasattr(self, 'swa_scheduler') and iter_total >= self.cfg.SOLVER.SWA.START_ITER:
            self.swa_scheduler.step()
            return

        if self.cfg.SOLVER.LR_SCHEDULER_NAME == 'ReduceLROnPlateau':
            self.lr_scheduler.step(loss)
        else:
            self.lr_scheduler.step()

    # -----------------------------------------------------------------------------
    # Chunk processing for TileDataset
    # -----------------------------------------------------------------------------
    def run_chunk(self, mode: str):
        r"""Run chunk-based training and inference for large-scale datasets.
        """
        self.dataset = get_dataset(self.cfg, self.augmentor, mode)
        if mode == 'train':
            num_chunk = self.total_iter_nums // self.cfg.DATASET.DATA_CHUNK_ITER
            self.total_iter_nums = self.cfg.DATASET.DATA_CHUNK_ITER
            for chunk in range(num_chunk):
                self.dataset.updatechunk()
                self.dataloader = build_dataloader(self.cfg, self.augmentor, mode,
                                                   dataset=self.dataset.dataset)
                self.dataloader = iter(self.dataloader)
                print('start train for chunk %d' % chunk)
                self.train()
                print('finished train for chunk %d' % chunk)
                self.start_iter += self.cfg.DATASET.DATA_CHUNK_ITER
                del self.dataloader
            return

        # inference mode
        num_chunk = len(self.dataset.chunk_ind)
        print("Total number of chunks: ", num_chunk)
        for chunk in range(num_chunk):
            self.dataset.updatechunk(do_load=False)
            self.test_filename = self.cfg.INFERENCE.OUTPUT_NAME + \
                '_' + self.dataset.get_coord_name() + '.h5'
            self.test_filename = self.augmentor.update_name(
                self.test_filename)
            if not os.path.exists(os.path.join(self.output_dir, self.test_filename)):
                self.dataset.loadchunk()
                self.dataloader = build_dataloader(self.cfg, self.augmentor, mode,
                                                   dataset=self.dataset.dataset)
                self.dataloader = iter(self.dataloader)
                self.test()


def merge_func(seg, step=4):
    seg = merge_small_object(seg)
    seg = merge_small_object(seg, threshold=20, window=11)
    seg = merge_small_object(seg, threshold=50, window=11)
    # seg = merge_small_object(seg, threshold=100, window=21)
    # seg = merge_small_object(seg, threshold=300, window=21)
    return seg

def merge_small_object(seg, threshold=5, window=5):
    uid, uc = np.unique(seg, return_counts=True)
    for (ids, size) in zip(uid, uc):
        if size > threshold:
            continue
        # print(seg.shape)
        # print(ids)
        # print(np.where(seg == ids))
        pos_x, pos_y = np.where(seg == ids)
        pos_x = int(np.sum(pos_x) // np.size(pos_x))
        pos_y = int(np.sum(pos_y) // np.size(pos_y))
        pos_x = pos_x - window // 2
        pos_y = pos_y - window // 2
        seg_crop = seg[pos_x:pos_x+window, pos_y:pos_y+window]
        temp_uid, temp_uc = np.unique(seg_crop, return_counts=True)
        rank = np.argsort(-temp_uc)
        if len(temp_uc) > 2:
            if temp_uid[rank[0]] == 0:
                if temp_uid[rank[1]] == ids:
                    max_ids = temp_uid[rank[2]]
                else:
                    max_ids = temp_uid[rank[1]]
            else:
                max_ids = temp_uid[rank[0]]
            seg[seg==ids] = max_ids
    return seg
