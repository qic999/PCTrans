from skimage.metrics import adapted_rand_error as adapted_rand_ref
from skimage.metrics import variation_of_information as voi_ref
from connectomics.data.utils import readh5
import os

def eval_snemi(gt_instance, p_instance, output_txt=None):
    gt_seg = readh5(gt_instance)
    pre_seg = readh5(p_instance)
    arand = adapted_rand_ref(gt_seg, pre_seg, ignore_labels=(0))[0]
    voi_split, voi_merge = voi_ref(gt_seg, pre_seg, ignore_labels=(0))
    voi_sum = voi_split + voi_merge
    print('voi_split=%.6f, voi_merge=%.6f, voi_sum=%.6f, arand=%.6f' % \
            (voi_split, voi_merge, voi_sum, arand))
    if output_txt is not None:
        with open(output_txt+"/logging.txt", "a") as f:
            f.writelines(p_instance.split("/")[-1][:6])
            f.writelines("\n")
            f.writelines(" ".join([str(voi_split), str(voi_merge), str(voi_sum), str(arand)]))
            f.writelines("\n")
    return voi_split, voi_merge, voi_sum, arand

if __name__ == '__main__':
    gt_instance = ''
    p_instance = ''
    output_txt = ''
    eval_snemi(gt_instance, p_instance, output_txt)