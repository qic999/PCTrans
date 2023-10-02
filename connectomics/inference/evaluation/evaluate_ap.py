import glob
import logging
import os
import sys

import h5py
import argparse
import numpy as np
import scipy.ndimage
from scipy.optimize import linear_sum_assignment
from skimage.segmentation import relabel_sequential
import tifffile
# import toml

logger = logging.getLogger(__name__)


class Metrics:
    def __init__(self, fn):
        self.metricsDict = {}
        self.metricsArray = []
        self.fn = fn
        self.outFl = open(self.fn+".txt", 'a')

    # def save(self):
    #     self.outFl.close()
    #     logger.info("saving %s", self.fn)
    #     tomlFl = open(self.fn+".toml", 'w')
    #     toml.dump(self.metricsDict, tomlFl)

    def addTable(self, name, dct=None):
        levels = name.split(".")
        if dct is None:
            dct = self.metricsDict
        if levels[0] not in dct:
            dct[levels[0]] = {}
        if len(levels) > 1:
            name = ".".join(levels[1:])
            self.addTable(name, dct[levels[0]])

    def getTable(self, name, dct=None):
        levels = name.split(".")
        if dct is None:
            dct = self.metricsDict
        if len(levels) == 1:
            return dct[levels[0]]
        else:
            name = ".".join(levels[1:])
            return self.getTable(name, dct=dct[levels[0]])

    def addMetric(self, table, name, value):
        as_str = "{}: {}".format(name, value)
        self.outFl.write(as_str+"\n")
        self.metricsArray.append(value)
        tbl = self.getTable(table)
        tbl[name] = value


def maybe_crop(pred_labels, gt_labels, overlapping_inst=False):
    if overlapping_inst:
        if gt_labels.shape[1:] == pred_labels.shape[1:]:
            return pred_labels, gt_labels
        else:
            if gt_labels.shape == pred_labels.shape:
                return pred_labels, gt_labels
            if gt_labels.shape[-1] > pred_labels.shape[-1]:
                bigger_arr = gt_labels
                smaller_arr = pred_labels
                swapped = False
            else:
                bigger_arr = pred_labels
                smaller_arr = gt_labels
                swapped = True

            begin = (np.array(bigger_arr.shape[-2:]) -
                     np.array(smaller_arr.shape[-2:])) // 2
            end = np.array(bigger_arr.shape[-2:]) - begin
            if (np.array(bigger_arr.shape[-2:]) -
                np.array(smaller_arr.shape[-2:]))[-1] % 2 == 1:
                end[-1] -= 1
            if (np.array(bigger_arr.shape[-2:]) -
                np.array(smaller_arr.shape[-2:]))[-2] % 2 == 1:
                end[-2] -= 1
            bigger_arr = bigger_arr[...,
                                    begin[0]:end[0],
                                    begin[1]:end[1]]
            if not swapped:
                gt_labels = bigger_arr
                pred_labels = smaller_arr
            else:
                pred_labels = bigger_arr
                gt_labels = smaller_arr
            logger.debug("gt shape cropped %s", gt_labels.shape)
            logger.debug("pred shape cropped %s", pred_labels.shape)

            return pred_labels, gt_labels
    else:
        if gt_labels.shape == pred_labels.shape:
            return pred_labels, gt_labels
        if gt_labels.shape[0] > pred_labels.shape[0]:
            bigger_arr = gt_labels
            smaller_arr = pred_labels
            swapped = False
        else:
            bigger_arr = pred_labels
            smaller_arr = gt_labels
            swapped = True
        begin = (np.array(bigger_arr.shape) -
                 np.array(smaller_arr.shape)) // 2
        end = np.array(bigger_arr.shape) - begin
        if len(bigger_arr.shape) == 2:
            bigger_arr = bigger_arr[begin[0]:end[0],
                                    begin[1]:end[1]]
        else:
            if (np.array(bigger_arr.shape) -
                np.array(smaller_arr.shape))[2] % 2 == 1:
                end[2] -= 1
            bigger_arr = bigger_arr[begin[0]:end[0],
                                    begin[1]:end[1],
                                    begin[2]:end[2]]
        if not swapped:
            gt_labels = bigger_arr
            pred_labels = smaller_arr
        else:
            pred_labels = bigger_arr
            gt_labels = smaller_arr
        logger.debug("gt shape cropped %s", gt_labels.shape)
        logger.debug("pred shape cropped %s", pred_labels.shape)

        return pred_labels, gt_labels

def evaluate_ap(pred_labels, gt_labels, background=0,
                  foreground_only=False, outFn='test', **kwargs):

    logger.debug("prediction shape %s", pred_labels.shape)
    logger.debug("gt min %f, max %f, shape %s", np.min(gt_labels),
                 np.max(gt_labels), gt_labels.shape)
    logger.debug("gt shape %s", gt_labels.shape)

    # heads up: should not crop channel dimensions, assuming channels first
    overlapping_inst = kwargs.get('overlapping_inst', False)
    pred_labels, gt_labels = maybe_crop(pred_labels, gt_labels,
                                        overlapping_inst)

    logger.debug("prediction %s, shape %s", np.unique(pred_labels),
                 pred_labels.shape)
    logger.debug("gt %s, shape %s", np.unique(gt_labels),
                 gt_labels.shape)

    if foreground_only:
        try:
            pred_labels[gt_labels==0] = 0
        except IndexError:
            pred_labels[:, np.any(gt_labels, axis=0).astype(np.int)==0] = 0

    # relabel gt labels in case of binary mask per channel
    if overlapping_inst and np.max(gt_labels) == 1:
        for i in range(gt_labels.shape[0]):
            gt_labels[i] = gt_labels[i] * (i + 1)

    if kwargs.get('use_linear_sum_assignment'):
        return evaluate_linear_sum_assignment(gt_labels, pred_labels, outFn,
                                              overlapping_inst,
                                              kwargs.get('filterSz', None),
                                              visualize=kwargs.get("visualize", False))

    # get gt cell ids and the size of the corresponding cell
    gt_labels_list, gt_counts = np.unique(gt_labels, return_counts=True)
    gt_labels_count_dict = {}
    logger.debug("%s %s", gt_labels_list, gt_counts)
    for (l, c) in zip(gt_labels_list, gt_counts):
        gt_labels_count_dict[l] = c

    # get pred cell ids
    pred_labels_list, pred_counts = np.unique(pred_labels,
                                              return_counts=True)
    logger.debug("%s %s", pred_labels_list, pred_counts)
    pred_labels_count_dict = {}
    for (l, c) in zip(pred_labels_list, pred_counts):
        pred_labels_count_dict[l] = c

    # get overlapping labels
    if overlapping_inst:
        pred_tile = [1,] * pred_labels.ndim
        pred_tile[0] = gt_labels.shape[0]
        gt_tile = [1,] * gt_labels.ndim
        gt_tile[1] = pred_labels.shape[0]
        pred_tiled = np.tile(pred_labels, pred_tile).flatten()
        gt_tiled = np.tile(gt_labels, gt_tile).flatten()
        mask = np.logical_or(pred_tiled > 0, gt_tiled > 0)
        overlay = np.array([
            pred_tiled[mask],
            gt_tiled[mask]
        ])
        overlay_labels, overlay_labels_counts = np.unique(
            overlay, return_counts=True, axis=1)
        overlay_labels = np.transpose(overlay_labels)
    else:
        overlay = np.array([pred_labels.flatten(),
                            gt_labels.flatten()])
        logger.debug("overlay shape %s", overlay.shape)
        # get overlaying cells and the size of the overlap
        overlay_labels, overlay_labels_counts = np.unique(overlay,
                                             return_counts=True, axis=1)
        overlay_labels = np.transpose(overlay_labels)

    # identify overlaying cells where more than 50% of gt cell is covered
    matchesSEG = np.asarray([c > 0.5 * float(gt_counts[gt_labels_list == v])
        for (u,v), c in zip(overlay_labels, overlay_labels_counts)],
                            dtype=np.bool)

    # get their ids
    matches_labels = overlay_labels[matchesSEG]

    # remove background
    if background is not None:
        pred_labels_list = pred_labels_list[pred_labels_list != background]
        gt_labels_list = gt_labels_list[gt_labels_list != background]

    matches_mat = np.zeros((len(pred_labels_list), len(gt_labels_list)))
    for (u, v) in matches_labels:
        if u > 0 and v > 0:
            matches_mat[np.where(pred_labels_list == u),
                        np.where(gt_labels_list == v)] = 1

    diceGT = {}
    iouGT = {}
    segGT = {}
    diceP = {}
    iouP = {}
    segP = {}
    segPrev = {}
    for (u,v), c in zip(overlay_labels, overlay_labels_counts):
        dice = 2.0 * c / (gt_labels_count_dict[v] + pred_labels_count_dict[u])
        iou = c / (gt_labels_count_dict[v] + pred_labels_count_dict[u] - c)

        if c > 0.5 * gt_labels_count_dict[v]:
            seg = iou
        else:
            seg = 0
        if c > 0.5 * pred_labels_count_dict[u]:
            seg2 = iou
        else:
            seg2 = 0

        if v not in diceGT:
            diceGT[v] = []
            iouGT[v] = []
            segGT[v] = []
        if u not in diceP:
            diceP[u] = []
            iouP[u] = []
            segP[u] = []
            segPrev[u] = []
        diceGT[v].append(dice)
        iouGT[v].append(iou)
        segGT[v].append(seg)
        diceP[u].append(dice)
        iouP[u].append(iou)
        segP[u].append(seg)
        segPrev[u].append(seg2)

    if background is not None:
        iouP.pop(background)
        iouGT.pop(background)
        diceP.pop(background)
        diceGT.pop(background)
        segP.pop(background)
        segPrev.pop(background)
        segGT.pop(background)

    if not diceGT:
        logger.error("No labels found in gt image")
        return
    if not diceP:
        logger.error("No labels found in pred image")
        return

    dice = 0
    cnt = 0
    for (k, vs) in diceGT.items():
        vs = sorted(vs, reverse=True)
        dice += vs[0]
        cnt += 1
    diceGT = dice/max(1, cnt)

    dice = 0
    cnt = 0
    for (k, vs) in diceP.items():
        vs = sorted(vs, reverse=True)
        dice += vs[0]
        cnt += 1
    diceP = dice/max(1, cnt)

    iou = []
    instances = gt_labels.copy().astype(np.float32)
    for (k, vs) in iouGT.items():
        vs = sorted(vs, reverse=True)
        iou.append(vs[0])
        instances[instances==k] = vs[0]
    iouGT = np.array(iou)
    iouGTMn = np.mean(iouGT)

    iou = []
    iouIDs = []
    instances = pred_labels.copy().astype(np.float32)
    for (k, vs) in iouP.items():
        vs = sorted(vs, reverse=True)
        iou.append(vs[0])
        iouP[k] = vs
        iouIDs.append(k)
        instances[instances==k] = vs[0]

    iouP_2 = np.array(iou)
    iouIDs = np.array(iouIDs)
    iouPMn = np.mean(iouP_2)

    seg = 0
    cnt = 0
    for (k, vs) in segGT.items():
        vs = sorted(vs, reverse=True)
        seg += vs[0]
        cnt += 1
    segGT = seg/max(1, cnt)

    seg = 0
    cnt = 0
    for (k, vs) in segP.items():
        vs = sorted(vs, reverse=True)
        seg += vs[0]
        cnt += 1
    segP = seg/max(1, cnt)

    seg = 0
    cnt = 0
    for (k, vs) in segPrev.items():
        vs = sorted(vs, reverse=True)
        seg += vs[0]
        cnt += 1
    segPrev = seg/max(1, cnt)

    # non-split vertices num non-empty cols - num non-empty rows
    # (more than one entry in col: predicted cell with more than one
    # ground truth cell assigned)
    # (other way around not possible due to 50% rule)
    ns = np.sum(np.count_nonzero(matches_mat, axis=0)) \
            - np.sum(np.count_nonzero(matches_mat, axis=1) > 0)
    ns = int(ns)

    # false negative: empty cols
    # (no predicted cell for ground truth cell)
    fn = np.sum(np.sum(matches_mat, axis=0) == 0)
    # tmp = np.sum(matches_mat, axis=0)==0
    # for i in range(len(tmp)):
    #     print(i, tmp[i], gt_labels_list[i])
    fn = int(fn)

    # false positive: empty rows
    # (predicted cell for non existing ground truth cell)
    fp = np.sum(np.sum(matches_mat, axis=1) == 0)
    # tmp = np.sum(matches_mat, axis=1)==0
    # for i in range(len(tmp)):
    #     print(i, tmp[i], pred_labels_list[i])
    # print(np.sum(matches_mat, axis=1)==0)
    fp = int(fp)

    # true positive: row with single entry (can be 0, 1, or more)
    tpP = np.sum(np.sum(matches_mat, axis=1) == 1)
    tpP = int(tpP)

    # true positive: non-empty col (can only be 0 or 1)
    tpGT = np.sum(np.sum(matches_mat, axis=0) > 0)
    tpGT = int(tpGT)


    metrics = Metrics(outFn)
    tblNameGen = "general"
    metrics.addTable(tblNameGen)
    metrics.addMetric(tblNameGen, "Num GT", len(gt_labels_list))
    metrics.addMetric(tblNameGen, "Num Pred", len(pred_labels_list))
    metrics.addMetric(tblNameGen, "GT/Ref -> Pred mean dice", diceGT)
    metrics.addMetric(tblNameGen, "Pred -> GT/Ref mean dice", diceP)
    metrics.addMetric(tblNameGen, "GT/Ref -> Pred mean iou", iouGTMn)
    metrics.addMetric(tblNameGen, "Pred -> GT/Ref mean iou", iouPMn)
    metrics.addMetric(tblNameGen, "GT/Ref -> Pred mean seg", segGT)
    metrics.addMetric(tblNameGen, "Pred -> GT/Ref mean seg", segP)
    metrics.addMetric(tblNameGen, "Pred -> GT/Ref mean seg rev", segPrev)
    metrics.addMetric(tblNameGen, "Pred -> GT/Ref NS", ns)
    metrics.addMetric(tblNameGen, "Pred -> GT/Ref FP", fp)
    metrics.addMetric(tblNameGen, "Pred -> GT/Ref TP", tpP)
    metrics.addMetric(tblNameGen, "GT/Ref -> Pred FN", fn)
    metrics.addMetric(tblNameGen, "GT/Ref -> Pred TP", tpGT)

    ths = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    aps = []
    metrics.addTable("confusion_matrix")
    for th in ths:
        tblname = "confusion_matrix.th_"+str(th).replace(".","_")
        metrics.addTable(tblname)
        apTP = 0
        for pID in np.nonzero(iouP_2 > th)[0]:
            if len(iouP[iouIDs[pID]]) == 0:
                pass
            elif len(iouP[iouIDs[pID]]) == 1:
                apTP += 1
            elif len(iouP[iouIDs[pID]]) > 1 and iouP[iouIDs[pID]][1] < th:
                apTP += 1
        metrics.addMetric(tblname, "AP_TP", apTP)
        apTP = np.count_nonzero(iouP_2[iouP_2>th])
        apFP = np.count_nonzero(iouP_2[iouP_2<=th])
        apFN = np.count_nonzero(iouGT[iouGT<=th])
        metrics.addMetric(tblname, "AP_TP", apTP)
        metrics.addMetric(tblname, "AP_FP", apFP)
        metrics.addMetric(tblname, "AP_FN", apFN)
        p = 1.*(apTP) / max(1, apTP +  apFP)
        rec = 1.*(apTP) / max(1, apTP +  apFN)
        aps.append(p*rec)
        metrics.addMetric(tblname, "AP", p*rec)

        precision = 1.*(apTP) / max(1, len(pred_labels_list))
        metrics.addMetric(tblname, "precision", precision)
        recall = 1.*(apTP) / max(1, len(gt_labels_list))
        metrics.addMetric(tblname, "recall", recall)
        if (precision + recall) > 0:
            fscore = (2. * precision * recall) / max(1, precision + recall)
        else:
            fscore = 0.0
        metrics.addMetric(tblname, 'fscore', fscore)

    avAP = np.mean(aps)
    metrics.addMetric("confusion_matrix", "avAP", avAP)

    # metrics.save()
    # import pdb; pdb.set_trace()
    # return metrics.metricsDict
    return metrics.metricsDict['confusion_matrix']['th_0_75']['AP'], metrics.metricsDict['confusion_matrix']['th_0_5']['AP'], avAP


def evaluate_linear_sum_assignment(gt_labels, pred_labels, outFn,
                                   overlapping_inst=False, filterSz=None,
                                   visualize=False):
    if filterSz is not None:
        ls, cs = np.unique(pred_labels, return_counts=True)
        pred_labels2 = np.copy(pred_labels)
        print(sorted(zip(cs, ls)))
        for l, c in zip(ls, cs):
            if c < filterSz:
                pred_labels[pred_labels==l] = 0
        print(outFn)

    pred_labels_rel, _, _ = relabel_sequential(pred_labels.astype(np.int))
    gt_labels_rel, _, _ = relabel_sequential(gt_labels)

    if overlapping_inst:
        pred_tile = [1, ] * pred_labels_rel.ndim
        pred_tile[0] = gt_labels_rel.shape[0]
        gt_tile = [1, ] * gt_labels_rel.ndim
        gt_tile[1] = pred_labels_rel.shape[0]
        pred_tiled = np.tile(pred_labels_rel, pred_tile).flatten()
        gt_tiled = np.tile(gt_labels_rel, gt_tile).flatten()
        mask = np.logical_or(pred_tiled > 0, gt_tiled > 0)
        overlay = np.array([
            pred_tiled[mask],
            gt_tiled[mask]
        ])
        overlay_labels, overlay_labels_counts = np.unique(
            overlay, return_counts=True, axis=1)
        overlay_labels = np.transpose(overlay_labels)
    else:
        overlay = np.array([pred_labels_rel.flatten(),
                            gt_labels_rel.flatten()])
        logger.debug("overlay shape relabeled %s", overlay.shape)
        # get overlaying cells and the size of the overlap
        overlay_labels, overlay_labels_counts = np.unique(
            overlay, return_counts=True, axis=1)
        overlay_labels = np.transpose(overlay_labels)

    # get gt cell ids and the size of the corresponding cell
    gt_labels_list, gt_counts = np.unique(gt_labels_rel, return_counts=True)
    gt_labels_count_dict = {}
    logger.debug("%s %s", gt_labels_list, gt_counts)
    for (l,c) in zip(gt_labels_list, gt_counts):
        gt_labels_count_dict[l] = c

    # get pred cell ids
    pred_labels_list, pred_counts = np.unique(pred_labels_rel,
                                              return_counts=True)
    logger.debug("%s %s", pred_labels_list, pred_counts)

    pred_labels_count_dict = {}
    for (l,c) in zip(pred_labels_list, pred_counts):
        pred_labels_count_dict[l] = c

    num_pred_labels = int(np.max(pred_labels_rel))
    num_gt_labels = int(np.max(gt_labels_rel))
    num_matches = min(num_gt_labels, num_pred_labels)
    iouMat = np.zeros((num_gt_labels+1, num_pred_labels+1),
                      dtype=np.float32)
    recallMat = np.zeros((num_gt_labels+1, num_pred_labels+1),
                         dtype=np.float32)
    precMat = np.zeros((num_gt_labels+1, num_pred_labels+1),
                       dtype=np.float32)
    fscoreMat = np.zeros((num_gt_labels+1, num_pred_labels+1),
                         dtype=np.float32)

    for (u,v), c in zip(overlay_labels, overlay_labels_counts):
        iou = c / (gt_labels_count_dict[v] + pred_labels_count_dict[u] - c)

        iouMat[v, u] = iou
        recallMat[v, u] = c / gt_labels_count_dict[v]
        precMat[v, u] = c / pred_labels_count_dict[u]
        fscoreMat[v, u] = 2 * (precMat[v, u] * recallMat[v, u]) / \
                              (precMat[v, u] + recallMat[v, u])
    iouMat = iouMat[1:, 1:]
    recallMat = recallMat[1:, 1:]
    precMat = precMat[1:, 1:]
    fscoreMat = fscoreMat[1:, 1:]

    metrics = Metrics(outFn)
    tblNameGen = "general"
    metrics.addTable(tblNameGen)
    metrics.addMetric(tblNameGen, "Num GT", num_gt_labels)
    metrics.addMetric(tblNameGen, "Num Pred", num_pred_labels)

    ths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    aps = []
    metrics.addTable("confusion_matrix")
    for th in ths:
        tblname = "confusion_matrix.th_"+str(th).replace(".", "_")
        metrics.addTable(tblname)
        fscore = 0
        if num_matches > 0 and np.max(iouMat) > th:
            costs = -(iouMat >= th).astype(float) - iouMat / (2*num_matches)
            logger.info("start computing lin sum assign for th %s (%s)",
                        th, outFn)
            gt_ind, pred_ind = linear_sum_assignment(costs)
            assert num_matches == len(gt_ind) == len(pred_ind)
            match_ok = iouMat[gt_ind, pred_ind] >= th
            tp = np.count_nonzero(match_ok)
            fscore_cnt = 0
            for idx, match in enumerate(match_ok):
                if match:
                    fscore = fscoreMat[gt_ind[idx], pred_ind[idx]]
                    if fscore >= 0.8:
                        fscore_cnt += 1
        else:
            tp = 0
            fscore_cnt = 0
        if visualize and tp > 0 and th == 0.5:
            vis_tp = np.zeros_like(gt_labels_rel, dtype=np.float32)
            vis_fp = np.zeros_like(gt_labels_rel, dtype=np.float32)
            vis_fn = np.zeros_like(gt_labels_rel, dtype=np.float32)
            vis_tp_seg = np.zeros_like(gt_labels_rel, dtype=np.float32)
            vis_tp_seg2 = np.zeros_like(gt_labels_rel, dtype=np.float32)
            vis_fp_seg = np.zeros_like(gt_labels_rel, dtype=np.float32)
            vis_fn_seg = np.zeros_like(gt_labels_rel, dtype=np.float32)
            if len(gt_labels_rel.shape) == 3:
                vis_fp_seg_bnd = np.zeros_like(gt_labels_rel, dtype=np.float32)
                vis_fn_seg_bnd = np.zeros_like(gt_labels_rel, dtype=np.float32)

            cntrs_gt = scipy.ndimage.measurements.center_of_mass(
                gt_labels_rel > 0,
                gt_labels_rel, sorted(list(np.unique(gt_labels_rel)))[1:])
            cntrs_pred = scipy.ndimage.measurements.center_of_mass(
                pred_labels_rel > 0,
                pred_labels_rel, sorted(list(np.unique(pred_labels_rel)))[1:])
            sz = 1
            for gti, pi, in zip(gt_ind, pred_ind):
                if iouMat[gti, pi] < th:
                    vis_fn_seg[gt_labels_rel == gti+1] = 1
                    if len(gt_labels_rel.shape) == 3:
                        set_boundary(gt_labels_rel, gti+1,
                                     vis_fn_seg_bnd)
                    vis_fp_seg[pred_labels_rel == pi+1] = 1
                    if len(gt_labels_rel.shape) == 3:
                        set_boundary(pred_labels_rel, pi+1,
                                     vis_fp_seg_bnd)
                    cntr = cntrs_gt[gti]
                    if len(gt_labels_rel.shape) == 3:
                        vis_fn[int(cntr[0]), int(cntr[1]), int(cntr[2])] = 1
                    else:
                        vis_fn[int(cntr[0]), int(cntr[1])] = 1
                    cntr = cntrs_pred[pi]
                    if len(gt_labels_rel.shape) == 3:
                        vis_fp[int(cntr[0]), int(cntr[1]), int(cntr[2])] = 1
                    else:
                        vis_fp[int(cntr[0]), int(cntr[1])] = 1
                else:
                    vis_tp_seg[gt_labels_rel == gti+1] = 1
                    cntr = cntrs_gt[gti]
                    if len(gt_labels_rel.shape) == 3:
                        vis_tp[int(cntr[0]), int(cntr[1]), int(cntr[2])] = 1
                    else:
                        vis_tp[int(cntr[0]), int(cntr[1])] = 1
                    vis_tp_seg2[pred_labels_rel == pi+1] = 1
            vis_tp = scipy.ndimage.gaussian_filter(vis_tp, sz, truncate=sz)
            for gti in range(num_gt_labels):
                if gti in gt_ind:
                    continue
                vis_fn_seg[gt_labels_rel == gti+1] = 1
                if len(gt_labels_rel.shape) == 3:
                    set_boundary(gt_labels_rel, gti+1,
                                 vis_fn_seg_bnd)
                cntr = cntrs_gt[gti]
                if len(gt_labels_rel.shape) == 3:
                    vis_fn[int(cntr[0]), int(cntr[1]), int(cntr[2])] = 1
                else:
                    vis_fn[int(cntr[0]), int(cntr[1])] = 1
            vis_fn = scipy.ndimage.gaussian_filter(vis_fn, sz, truncate=sz)
            for pi in range(num_pred_labels):
                if pi in pred_ind:
                    continue
                vis_fp_seg[pred_labels_rel == pi+1] = 1
                if len(gt_labels_rel.shape) == 3:
                    set_boundary(pred_labels_rel, pi+1,
                                 vis_fp_seg_bnd)
                cntr = cntrs_pred[pi]
                if len(gt_labels_rel.shape) == 3:
                    vis_fp[int(cntr[0]), int(cntr[1]), int(cntr[2])] = 1
                else:
                    vis_fp[int(cntr[0]), int(cntr[1])] = 1
            vis_fp = scipy.ndimage.gaussian_filter(vis_fp, sz, truncate=sz)
            vis_tp = vis_tp/np.max(vis_tp)
            vis_fp = vis_fp/np.max(vis_fp)
            vis_fn = vis_fn/np.max(vis_fn)
            with h5py.File(outFn + "_vis.hdf", 'w') as fi:
                fi.create_dataset(
                    'volumes/vis_tp',
                    data=vis_tp,
                    compression='gzip')
                fi.create_dataset(
                    'volumes/vis_fp',
                    data=vis_fp,
                    compression='gzip')
                fi.create_dataset(
                    'volumes/vis_fn',
                    data=vis_fn,
                    compression='gzip')
                fi.create_dataset(
                    'volumes/vis_tp_seg',
                    data=vis_tp_seg,
                    compression='gzip')
                fi.create_dataset(
                    'volumes/vis_tp_seg2',
                    data=vis_tp_seg2,
                    compression='gzip')
                fi.create_dataset(
                    'volumes/vis_fp_seg',
                    data=vis_fp_seg,
                    compression='gzip')
                fi.create_dataset(
                    'volumes/vis_fn_seg',
                    data=vis_fn_seg,
                    compression='gzip')
                if len(gt_labels_rel.shape) == 3:
                    fi.create_dataset(
                        'volumes/vis_fp_seg_bnd',
                        data=vis_fp_seg_bnd,
                        compression='gzip')
                    fi.create_dataset(
                        'volumes/vis_fn_seg_bnd',
                        data=vis_fn_seg_bnd,
                        compression='gzip')

        metrics.addMetric(tblname, "Fscore_cnt", fscore_cnt)
        fp = num_pred_labels - tp
        fn = num_gt_labels - tp
        metrics.addMetric(tblname, "AP_TP", tp)
        metrics.addMetric(tblname, "AP_FP", fp)
        metrics.addMetric(tblname, "AP_FN", fn)

        p = 1.*(tp) / max(1, tp +  fp)
        rec = 1.*(tp) / max(1, tp +  fn)
        aps.append(p*rec)
        metrics.addMetric(tblname, "AP", p*rec)

        precision = tp / max(1, tp + fp)
        metrics.addMetric(tblname, "precision", precision)
        recall = tp / max(1, tp + fn)
        metrics.addMetric(tblname, "recall", recall)
        if (precision + recall) > 0:
            fscore = (2. * precision * recall) / max(1, precision + recall)
        else:
            fscore = 0.0
        metrics.addMetric(tblname, 'fscore', fscore)

    avAP19 = np.mean(aps)
    avAP59 = np.mean(aps[4:])
    metrics.addMetric("confusion_matrix", "avAP", avAP59)
    metrics.addMetric("confusion_matrix", "avAP59", avAP59)
    metrics.addMetric("confusion_matrix", "avAP19", avAP19)

    # metrics.save()
    return metrics.metricsDict


def set_boundary(labels_rel, label, target):
    coords_z, coords_y, coords_x = np.nonzero(labels_rel == label)
    coords = {}
    for z,y,x in zip(coords_z, coords_y, coords_x):
        coords.setdefault(z, []).append((z, y, x))
    max_z = -1
    max_z_len = -1
    for z, v in coords.items():
        if len(v) > max_z_len:
            max_z_len = len(v)
            max_z = z
    tmp = np.zeros_like(labels_rel[max_z], dtype=np.float32)
    tmp = labels_rel[max_z]==label
    struct = scipy.ndimage.generate_binary_structure(2, 2)
    eroded_tmp = scipy.ndimage.binary_erosion(
        tmp,
        iterations=1,
        structure=struct,
        border_value=1)
    bnd = np.logical_xor(tmp, eroded_tmp)
    target[max_z][bnd] = 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_file', type=str,
                        help='path to res_file', required=True)
    parser.add_argument('--gt_file', type=str,
                        help='path to gt_file', required=True)
    parser.add_argument('--metric', type=str,
                        default="confusion_matrix.th_0_5.AP",
                        help='check if this metric already has been computed in possibly existing result files')
    parser.add_argument('--background', type=int,
                        help='label for background (use -1 for None)',
                        default="0")
    parser.add_argument("--use_gt_fg", help="usually not used (deprecated)",
                        action="store_true")
    parser.add_argument("--overlapping_inst", help="if there can be multiple instances per pixel",
                        action="store_true")
    parser.add_argument("--from_scratch",
                        help="recompute everything (instead of checking if results are already there)",
                        action="store_true")
    parser.add_argument("--no_use_linear_sum_assignment",
                        help="don't use Hungarian matching",
                        dest='use_linear_sum_assignment',
                        action="store_false")
    parser.add_argument("--visualize", help="",
                        action="store_true")
    parser.add_argument("--debug", help="",
                        action="store_true")

    logger.debug("arguments %s",tuple(sys.argv))
    args = parser.parse_args()
    if args.use_gt_fg:
        logger.info("using gt foreground")

    evaluate_ap(args.res_file, args.gt_file,
                  foreground_only=args.use_gt_fg,
                  background=args.background,
                  debug=args.debug, visualize=args.visualize)