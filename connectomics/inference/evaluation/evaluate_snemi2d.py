import numpy as np
from PIL import Image
import h5py
import argparse
import SimpleITK as sitk
import imageio
import pdb
from skimage.metrics import adapted_rand_error as adapted_rand_ref
from skimage.metrics import variation_of_information as voi_ref
from scipy.optimize import linear_sum_assignment
import numexpr as ne
from connectomics.data.utils import readh5

def remap_label(pred, by_size=False):
    """
    Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3]
    not [0, 2, 4, 6]. The ordering of instances (which one comes first)
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID
    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger nuclei has smaller id (on-top)
    """
    pred_id = list(np.unique(pred))
    pred_id.remove(0)
    if len(pred_id) == 0:
        return pred # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred


# fast version of Aggregated Jaccrd Index
def agg_jc_index(gt_ins, pred):
    from tqdm import tqdm_notebook
    """Calculate aggregated jaccard index for prediction & GT mask
    copy from: https://github.com/jimmy15923/Miccai_challenge_MONUSEG/blob/master/Aggregate_Jaccard_Index.py
    reference paper here: https://www.dropbox.com/s/j3154xgkkpkri9w/IEEE_TMI_NuceliSegmentation.pdf?dl=0
    mask: Ground truth mask, shape = [1000, 1000, instances]
    pred: Prediction mask, shape = [1000,1000], dtype = uint16, each number represent one instance
    Returns: Aggregated Jaccard index for GT & mask 
    """

    def compute_iou(m, pred, pred_mark_isused, idx_pred):
        # check the prediction has been used or not
        if pred_mark_isused[idx_pred]:
            intersect = 0
            union = np.count_nonzero(m)
        else:
            p = (pred == idx_pred)
            # replace multiply with bool operation
            s = ne.evaluate("m&p")
            intersect = np.count_nonzero(s)
            union = np.count_nonzero(m) + np.count_nonzero(p) - intersect
        return (intersect, union)
    mask = tras_gt(gt_ins.copy())
    mask = mask.astype(np.bool)
    c = 0  # count intersection
    u = 0  # count union
    pred_instance = pred.max()  # predcition instance number
    if pred_instance == 0:
        return 0
    pred_mark_used = []  # mask used
    pred_mark_isused = np.zeros((pred_instance + 1), dtype=bool)

    for idx_m in range(len(mask[0, 0, :])):
        m = np.take(mask, idx_m, axis=2)

        intersect_list, union_list = zip(
            *[compute_iou(m, pred, pred_mark_isused, idx_pred) for idx_pred in range(1, pred_instance + 1)])

        iou_list = np.array(intersect_list) / np.array(union_list)
        hit_idx = np.argmax(iou_list)
        c += intersect_list[hit_idx]
        u += union_list[hit_idx]
        pred_mark_used.append(hit_idx)
        pred_mark_isused[hit_idx + 1] = True

    pred_mark_used = [x + 1 for x in pred_mark_used]
    pred_fp = set(np.unique(pred)) - {0} - set(pred_mark_used)
    pred_fp_pixel = np.sum([np.sum(pred == i) for i in pred_fp])

    u += pred_fp_pixel
    return (c / u)

def tras_gt(gt):
    num_ins = np.amax(gt)
    out = np.zeros([gt.shape[0], gt.shape[1], num_ins], dtype = np.uint16)# .astype(np.uint16)
    for i in range (1, num_ins + 1):
        mask_cur = (gt == i)
        out[:,:,i-1] = mask_cur
    return out

def get_fast_pq(true, pred, match_iou=0.5):
    """
    `match_iou` is the IoU threshold level to determine the pairing between
    GT instances `p` and prediction instances `g`. `p` and `g` is a pair
    if IoU > `match_iou`. However, pair of `p` and `g` must be unique
    (1 prediction instance to 1 GT instance mapping).
    If `match_iou` < 0.5, Munkres assignment (solving minimum weight matching
    in bipartite graphs) is caculated to find the maximal amount of unique pairing.
    If `match_iou` >= 0.5, all IoU(p,g) > 0.5 pairing is proven to be unique and
    the number of pairs is also maximal.

    Fast computation requires instance IDs are in contiguous orderding
    i.e [1, 2, 3, 4] not [2, 3, 6, 10]. Please call `remap_label` beforehand
    and `by_size` flag has no effect on the result.
    Returns:
        [dq, sq, pq]: measurement statistic
        [paired_true, paired_pred, unpaired_true, unpaired_pred]:
                      pairing information to perform measurement

    """
    assert match_iou >= 0.0, "Cant' be negative"

    true = np.copy(true)
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [None, ]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [None, ]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_iou = np.zeros([len(true_id_list) - 1,
                             len(pred_id_list) - 1], dtype=np.float64)

    # caching pairwise iou
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            iou = inter / (total - inter)
            pairwise_iou[true_id - 1, pred_id - 1] = iou
    #
    if match_iou >= 0.5:
        paired_iou = pairwise_iou[pairwise_iou > match_iou]
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true += 1  # index is instance id - 1
        paired_pred += 1  # hence return back to original
    else:  # * Exhaustive maximal unique pairing
        #### Munkres pairing with scipy library
        # the algorithm return (row indices, matched column indices)
        # if there is multiple same cost in a row, index of first occurence
        # is return, thus the unique pairing is ensure
        # inverse pair to get high IoU as minimum
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        ### extract the paired cost and remove invalid pair
        paired_iou = pairwise_iou[paired_true, paired_pred]

        # now select those above threshold level
        # paired with iou = 0.0 i.e no intersection => FP or FN
        paired_true = list(paired_true[paired_iou > match_iou] + 1)
        paired_pred = list(paired_pred[paired_iou > match_iou] + 1)
        paired_iou = paired_iou[paired_iou > match_iou]

    # get the actual FP and FN
    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    # print(paired_iou.shape, paired_true.shape, len(unpaired_true), len(unpaired_pred))

    #
    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)
    # get the F1-score i.e DQ
    dq = tp / (tp + 0.5 * fp + 0.5 * fn)
    # get the SQ, no paired has 0 iou so not impact
    sq = paired_iou.sum() / (tp + 1.0e-6)

    return [dq, sq, dq * sq], [paired_true, paired_pred, unpaired_true, unpaired_pred]


# Functions for evaluating binary segmentation
def confusion_matrix(pred, gt, thres=0.5):
    """Calculate the confusion matrix given a probablility threshold in (0,1).
    """
    TP = np.sum((gt == 1) & (pred > thres))
    FP = np.sum((gt == 0) & (pred > thres))
    TN = np.sum((gt == 0) & (pred <= thres))
    FN = np.sum((gt == 1) & (pred <= thres))
    return (TP, FP, TN, FN)

def get_binary_jaccard(pred, gt):
    """Evaluate the binary prediction at multiple thresholds using the Jaccard 
    Index, which is also known as Intersection over Union (IoU). If the prediction
    is already binarized, different thresholds will result the same result.
    Args:
        pred (numpy.ndarray): foreground probability of shape :math:`(Z, Y, X)`.
        gt (numpy.ndarray): binary foreground label of shape identical to the prediction.
        thres (list): a list of probablility threshold in (0,1). Default: [0.5]
    Return:
        score (numpy.ndarray): a numpy array of shape :math:`(N, 4)`, where :math:`N` is the 
        number of element(s) in the probability threshold list. The four scores in each line
        are foreground IoU, IoU, precision and recall, respectively.
    """
    pred = pred.flatten()
    gt = gt.flatten()
    pred = np.uint8(pred != 0)
    gt = np.uint8(gt != 0)

    TP = np.count_nonzero((pred + gt) == 2)  # true positive
    TN = np.count_nonzero((pred + gt) == 0)  # true negative
    FP = np.count_nonzero(pred > gt)  # false positive
    FN = np.count_nonzero(pred < gt)  # false negative
    
    precision = float(TP)/(TP+FP)
    recall = float(TP)/(TP+FN)
    iou_fg = float(TP)/(TP+FP+FN)
    iou_bg = float(TN)/(TN+FP+FN)
    iou = (iou_fg + iou_bg) / 2.0
        
    return iou_fg, iou


def dice_coeff(pred, label):

    # convert to 1-D array for convinience
    pred = pred.flatten()
    label = label.flatten()
    # convert to 0-1 array
    pred = np.uint8(pred != 0)
    lable = np.uint8(label != 0)

    met_dict = {}  # metrics dictionary

    TP = np.count_nonzero((pred + lable) == 2)  # true positive
    TN = np.count_nonzero((pred + lable) == 0)  # true negative
    FP = np.count_nonzero(pred > lable)  # false positive
    FN = np.count_nonzero(pred < lable)  # false negative

    smooth = 1e-9  # avoid devide zero
    acc = (TP + TN) / (TP + TN + FP + FN + smooth)  # accuracy
    sn = TP / (TP + FP + smooth)  # sensitivity, or precision
    sp = TN / (TN + FN + smooth)  # specificity
    rc = TP / (TP + FN + smooth)  # recall
    f1 = 2 * sn * rc / (sn + rc + smooth)  # F1 mesure
    jac = TP / (TP + FN + FP + smooth)  # jaccard coefficient

    # return metrics as dictionary
    met_dict['TP'] = TP
    met_dict['TN'] = TN
    met_dict['FP'] = FP
    met_dict['FN'] = FN
    met_dict['acc'] = acc
    met_dict['sn'] = sn
    met_dict['sp'] = sp
    met_dict['rc'] = rc
    met_dict['f1'] = f1
    met_dict['jac'] = jac
    return met_dict

def metric(seg_img, lab):
    score = []
    jac_all = 0
    dice_all = 0
    aji_all = 0
    pq_all = 0
    aji_valid_num = 0
    pq_valid_num = 0
    depth = seg_img.shape[0]
    arand_all = 0
    voi_split_all = 0
    voi_merge_all = 0
    voi_sum_all = 0
    # print(seg_img.dtype) # int16
    # print(lab.dtype) # uint16
    # print(np.unique(seg_img))
    for k in range(depth):
        arand = adapted_rand_ref(lab[k], seg_img[k], ignore_labels=(0))[0]
        voi_split, voi_merge = voi_ref(lab[k], seg_img[k], ignore_labels=(0))
        voi_sum = voi_split + voi_merge
        arand_all = arand_all + arand
        voi_split_all = voi_split_all + voi_split
        voi_merge_all = voi_merge_all + voi_merge
        voi_sum_all = voi_sum_all + voi_sum

    #     dict = dice_coeff(seg_img[k, :, :], lab[k, :, :])
    #     gt = remap_label(lab[k, :, :])
    #     pred = remap_label(seg_img[k, :, :])
    #     aji = agg_jc_index(gt, pred)
    #     pq_info_cur = get_fast_pq(gt, pred, match_iou=0.5)[0]
    #     pq = pq_info_cur[2]
    #     jac_all = jac_all + dict["jac"]
    #     dice_all = dice_all + dict['f1']
    #     if not aji == 0:
    #         aji_all = aji_all + aji
    #         aji_valid_num = aji_valid_num + 1
    #     if not pq == 0:
    #         pq_valid_num = pq_valid_num + 1
    #         pq_all = pq_all + pq

    # if not pq_valid_num ==0 :
    #     final_pq = (pq_all / pq_valid_num)
    # else:
    #     final_pq = 0

    # if not aji_valid_num ==0 :
    #     final_aji = (aji_all / aji_valid_num)
    # else:
    #     final_aji = 0
    # final_jac = (jac_all / depth)
    # final_dice = (dice_all / depth)
    # print("dice:", final_dice)
    # print("jac:", final_jac)
    # print("aji:", final_aji)
    # print("pq:", final_pq)
    # score.append(str(final_dice))
    # score.append(str(final_jac))
    # score.append(str(final_aji))
    # score.append(str(final_pq))

    final_arand = arand_all/depth
    final_voi_split = voi_split_all/depth
    final_voi_merge = voi_merge_all/depth
    final_voi_sum = voi_sum_all/depth
    print("voi_split:", final_voi_split)
    print("voi_merge:", final_voi_merge)
    print("voi_sum:", final_voi_sum)
    print("arand:", final_arand)
    score.append(str(final_voi_split))
    score.append(str(final_voi_merge))
    score.append(str(final_voi_sum))
    score.append(str(final_arand))
    return score

def eval_snemi2d(gt_instance, p_instance, output_txt=None):
    
    # instance_target = sitk.ReadImage(gt_instance)
    # instance_target = sitk.GetArrayFromImage(instance_target)
    # instance_prediction = sitk.ReadImage(p_instance)
    # instance_prediction = sitk.GetArrayFromImage(instance_prediction)
    instance_target = readh5(gt_instance)
    instance_prediction = readh5(p_instance)


    score = metric(instance_prediction,instance_target)
    # fg_iou, iou = get_binary_jaccard(instance_prediction, instance_target)
    # print('fg_iou', fg_iou)
    # print('iou', iou)
    # score.append(str(fg_iou))
    # score.append(str(iou))

    # arand = adapted_rand_ref(instance_target, instance_prediction, ignore_labels=(0))[0]
    # voi_split, voi_merge = voi_ref(instance_target, instance_prediction, ignore_labels=(0))
    # voi_sum = voi_split + voi_merge
    # print('voi', voi_sum)
    # print('arand', arand)
    # score.append(str(voi_sum))
    # score.append(str(arand))

    # score: dice jac aji pq fg_iou iou
    if output_txt is not None:
        with open(output_txt+"/logging.txt", "a") as f:
            f.writelines(p_instance.split("/")[-1][:6])
            f.writelines("\n")
            f.writelines(" ".join(score))
            f.writelines("\n")
    return score

if __name__ == '__main__':
    gt_instance = ''
    p_instance = ''
    output_txt = ''
    eval_snemi2d(gt_instance, p_instance, output_txt)