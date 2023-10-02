from __future__ import print_function, division
from typing import Optional, Union, List
import numpy as np

from scipy import ndimage
from skimage.measure import label
from skimage.transform import resize
from skimage.morphology import dilation, binary_dilation
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects
from skimage.feature import peak_local_max

import mahotas

from connectomics.data.utils import getSegType, bbox_ND, crop_ND, replace_ND


__all__ = ['affi_watershed',
           'malis_watershed',
           'mc_baseline',
           'binary_connected',
           'binary_watershed',
           'bc_connected',
           'bc_watershed',
           'bcd_watershed',
           'polarity2instance']


import waterz
import malis
import gc

def malis_watershed(seed_map, thres1=0.9, thres2=0.8):
    if isinstance(seed_map, list):
        semantic = seed_map[0]
        boundary = seed_map[1]
        seed_map = (semantic > int(255*thres1)) * (boundary < int(255*thres2)) 
    elif isinstance(seed_map, np.ndarray):
        seed_map = seed_map
    else:
        raise RuntimeError("seed map is wrong!")    
    # generate affinity
    output_mixs = seed_map.astype(np.int32)
    # print('id(seed_map)',id(seed_map))
    # print('id(output_mixs)',id(output_mixs))
    affs = malis.seg_to_affgraph(output_mixs, malis.mknhood3d())
    del output_mixs
    gc.collect()
    affs = affs.astype(np.float32)
    
    # initial watershed + agglomerate
    seg = list(waterz.agglomerate(affs, [0.50]))[0]
    del affs
    gc.collect()
    seg = seg.astype(np.uint16)
    
    # grow boundary
    seg = dilation(seg, np.ones((1,7,7)))
    seg = remove_small_instances(seg)
    
    return seg

def affi_watershed(affs, seed_method, use_mahotas_watershed=True):
    affs_xy = 1.0 - 0.5*(affs[1] + affs[2])
    depth  = affs_xy.shape[0]
    fragments = np.zeros_like(affs[0]).astype(np.uint64)
    next_id = 1
    for z in range(depth):
        seeds, num_seeds = get_seeds(affs_xy[z], next_id=next_id, method=seed_method)
        if use_mahotas_watershed:
            fragments[z] = mahotas.cwatershed(affs_xy[z], seeds)
        else:
            fragments[z] = ndimage.watershed_ift((255.0*affs_xy[z]).astype(np.uint8), seeds)
        next_id += num_seeds
    return fragments

def get_seeds(boundary, method='grid', next_id=1, seed_distance=10):
    if method == 'grid':
        height = boundary.shape[0]
        width  = boundary.shape[1]
        seed_positions = np.ogrid[0:height:seed_distance, 0:width:seed_distance]
        num_seeds_y = seed_positions[0].size
        num_seeds_x = seed_positions[1].size
        num_seeds = num_seeds_x*num_seeds_y
        seeds = np.zeros_like(boundary).astype(np.int32)
        seeds[seed_positions] = np.arange(next_id, next_id + num_seeds).reshape((num_seeds_y,num_seeds_x))

    if method == 'minima':
        minima = mahotas.regmin(boundary)
        seeds, num_seeds = mahotas.label(minima)
        seeds += next_id
        seeds[seeds==next_id] = 0

    if method == 'maxima_distance':
        distance = mahotas.distance(boundary<0.5)
        maxima = mahotas.regmax(distance)
        seeds, num_seeds = mahotas.label(maxima)
        seeds += next_id
        seeds[seeds==next_id] = 0

    return seeds, num_seeds

def mc_baseline(affs, fragments=None, mc1_flag=True, mc2_flag=False, mc3_flag=False, mc4_flag=False):
    import elf.segmentation.multicut as mc
    import elf.segmentation.features as feats
    import elf.segmentation.watershed as ws
    boundary = False

    if affs.shape[0] == 2:
        affs = 1 - affs
        boundary_input = (affs[0]+affs[1])/2.0
        boundary = True
    elif affs.shape[0] == 3:
        affs = 1 - affs
        boundary_input = np.maximum(affs[1], affs[2])
        offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
    elif affs.shape[0] == 1:
        boundary_input = affs[0,...]
        boundary = True
    elif len(affs.shape) == 3:
        boundary_input = affs
        boundary = True

    # if fragments is None:
    #     fragments, max_id = ws.stacked_watershed(boundary_input, threshold=.25, sigma_seeds=2.)
    # rag = feats.compute_rag(fragments, n_labels=max_id+1)

    if boundary:
        if mc1_flag:
            if fragments is None:
                fragments = np.zeros_like(boundary_input, dtype='uint64')
                offset = 0
                for z in range(fragments.shape[0]):
                    wsz, max_id = ws.distance_transform_watershed(boundary_input[z], threshold=.5, sigma_seeds=3.0)
                    wsz += offset
                    offset += max_id
                    fragments[z] = wsz
            rag = feats.compute_rag(fragments)
            costs = feats.compute_boundary_features(rag, boundary_input)[:, 0]
            z_edges = feats.compute_z_edge_mask(rag, fragments)
            xy_edges = np.logical_not(z_edges) # np.logical_not(z_edges) = ~z_edges 
            edge_populations = [z_edges, xy_edges]
            edge_sizes = feats.compute_boundary_mean_and_length(rag, boundary_input)[:, 1]
            boundary_bias = .61
            costs = mc.transform_probabilities_to_costs(costs, beta=boundary_bias, edge_sizes=edge_sizes, edge_populations=edge_populations, weighting_exponent=1)
        if mc2_flag:
            fragments = ws.distance_transform_watershed(boundary_input, threshold=.6, sigma_seeds=2.)[0]
            rag = feats.compute_rag(fragments)
            costs = feats.compute_boundary_features(rag, boundary_input)[:, 0]
            z_edges = feats.compute_z_edge_mask(rag, fragments)
            xy_edges = np.logical_not(z_edges)
            edge_populations = [z_edges, xy_edges]
            edge_sizes = feats.compute_boundary_mean_and_length(rag, boundary_input)[:, 1]
            costs = mc.transform_probabilities_to_costs(costs, edge_sizes=edge_sizes, edge_populations=edge_populations)
        if mc3_flag:
            fragments = ws.distance_transform_watershed(boundary_input, threshold=.6, sigma_seeds=2.)[0]
            rag = feats.compute_rag(fragments)
            features = feats.compute_boundary_mean_and_length(rag, boundary_input)
            costs, sizes = features[:,0], features[:,1]
            boundary_bias = .45
            costs = mc.transform_probabilities_to_costs(costs, edge_sizes=sizes, beta=boundary_bias)
        if mc4_flag:
            fragments = ws.distance_transform_watershed(boundary_input, threshold=.6, sigma_seeds=2.)[0]
            rag = feats.compute_rag(fragments)
            costs = feats.compute_boundary_features(rag, boundary_input)[:, 0]
            edge_sizes = feats.compute_boundary_mean_and_length(rag, boundary_input)[:, 1]
            # edge_sizes = feats.compute_boundary_features(rag, boundary_input)[:, 1]
            costs = mc.transform_probabilities_to_costs(costs, edge_sizes=edge_sizes)
    else:
        if fragments is None:
            fragments = np.zeros_like(boundary_input, dtype='uint64')
            offset = 0
            for z in range(fragments.shape[0]):
                wsz, max_id = ws.distance_transform_watershed(boundary_input[z], threshold=.25, sigma_seeds=2.)
                wsz += offset
                offset += max_id
                fragments[z] = wsz
        rag = feats.compute_rag(fragments)
        costs = feats.compute_affinity_features(rag, affs, offsets)[:, 0]
        edge_sizes = feats.compute_boundary_mean_and_length(rag, boundary_input)[:, 1]
        costs = mc.transform_probabilities_to_costs(costs, edge_sizes=edge_sizes)
    node_labels = mc.multicut_kernighan_lin(rag, costs)
    segmentation = feats.project_node_labels_to_pixels(rag, node_labels)
    # io.volsave('segmentation.tif', segmentation.astype(np.uint8))
    # io.volsave('boundary_input.tif', boundary_input)
    return segmentation

# Post-processing functions of mitochondria instance segmentation model outputs
# as described in "MitoEM Dataset: Large-scale 3D Mitochondria Instance Segmentation
# from EM Images (MICCAI 2020, https://donglaiw.github.io/page/mitoEM/index.html)".
def binary_connected(volume, thres=0.8, thres_small=128, scale_factors=(1.0, 1.0, 1.0),
                     remove_small_mode='background'):
    r"""Convert binary foreground probability maps to instance masks via
    connected-component labeling.

    Args:
        volume (numpy.ndarray): foreground probability of shape :math:`(C, Z, Y, X)`.
        thres (float): threshold of foreground. Default: 0.8
        thres_small (int): size threshold of small objects to remove. Default: 128
        scale_factors (tuple): scale factors for resizing in :math:`(Z, Y, X)` order. Default: (1.0, 1.0, 1.0)
        remove_small_mode (str): ``'background'``, ``'neighbor'`` or ``'none'``. Default: ``'background'``
    """
    semantic = volume[0]
    foreground = (semantic > int(255*thres))
    segm = label(foreground)
    segm = remove_small_instances(segm, thres_small, remove_small_mode)

    if not all(x==1.0 for x in scale_factors):
        target_size = (int(semantic.shape[0]*scale_factors[0]),
                       int(semantic.shape[1]*scale_factors[1]),
                       int(semantic.shape[2]*scale_factors[2]))
        segm = resize(segm, target_size, order=0, anti_aliasing=False, preserve_range=True)

    return cast2dtype(segm)


def binary_watershed(volume, thres1=0.98, thres2=0.85, thres_small=128, scale_factors=(1.0, 1.0, 1.0),
                     remove_small_mode='background', seed_thres=32):
    r"""Convert binary foreground probability maps to instance masks via
    watershed segmentation algorithm.

    Note:
        This function uses the `skimage.segmentation.watershed <https://github.com/scikit-image/scikit-image/blob/master/skimage/segmentation/_watershed.py#L89>`_
        function that converts the input image into ``np.float64`` data type for processing. Therefore please make sure enough memory is allocated when handling large arrays.

    Args:
        volume (numpy.ndarray): foreground probability of shape :math:`(C, Z, Y, X)`.
        thres1 (float): threshold of seeds. Default: 0.98
        thres2 (float): threshold of foreground. Default: 0.85
        thres_small (int): size threshold of small objects to remove. Default: 128
        scale_factors (tuple): scale factors for resizing in :math:`(Z, Y, X)` order. Default: (1.0, 1.0, 1.0)
        remove_small_mode (str): ``'background'``, ``'neighbor'`` or ``'none'``. Default: ``'background'``
    """
    semantic = volume[0]
    seed_map = semantic > int(255*thres1)
    foreground = semantic > int(255*thres2)
    seed = label(seed_map)
    seed = remove_small_objects(seed, seed_thres)
    segm = watershed(-semantic.astype(np.float64), seed, mask=foreground)
    segm = remove_small_instances(segm, thres_small, remove_small_mode)

    if not all(x==1.0 for x in scale_factors):
        target_size = (int(semantic.shape[0]*scale_factors[0]),
                       int(semantic.shape[1]*scale_factors[1]),
                       int(semantic.shape[2]*scale_factors[2]))
        segm = resize(segm, target_size, order=0, anti_aliasing=False, preserve_range=True)

    return cast2dtype(segm)


def bc_connected(volume, thres1=0.8, thres2=0.5, thres_small=128, scale_factors=(1.0, 1.0, 1.0),
                 dilation_struct=(1,5,5), remove_small_mode='background'):
    r"""Convert binary foreground probability maps and instance contours to
    instance masks via connected-component labeling.

    Note:
        The instance contour provides additional supervision to distinguish closely touching
        objects. However, the decoding algorithm only keep the intersection of foreground and
        non-contour regions, which will systematically result in imcomplete instance masks.
        Therefore we apply morphological dilation (check :attr:`dilation_struct`) to enlarge
        the object masks.

    Args:
        volume (numpy.ndarray): foreground and contour probability of shape :math:`(C, Z, Y, X)`.
        thres1 (float): threshold of foreground. Default: 0.8
        thres2 (float): threshold of instance contours. Default: 0.5
        thres_small (int): size threshold of small objects to remove. Default: 128
        scale_factors (tuple): scale factors for resizing in :math:`(Z, Y, X)` order. Default: (1.0, 1.0, 1.0)
        dilation_struct (tuple): the shape of the structure for morphological dilation. Default: (1, 5, 5)
        remove_small_mode (str): ``'background'``, ``'neighbor'`` or ``'none'``. Default: ``'background'``
    """
    semantic = volume[0]
    boundary = volume[1]
    foreground = (semantic > int(255*thres1)) * (boundary < int(255*thres2))

    segm = label(foreground)
    struct = np.ones(dilation_struct)
    segm = dilation(segm, struct)
    segm = remove_small_instances(segm, thres_small, remove_small_mode)

    if not all(x==1.0 for x in scale_factors):
        target_size = (int(semantic.shape[0]*scale_factors[0]),
                       int(semantic.shape[1]*scale_factors[1]),
                       int(semantic.shape[2]*scale_factors[2]))
        segm = resize(segm, target_size, order=0, anti_aliasing=False, preserve_range=True)

    return cast2dtype(segm)


def bc_watershed(volume, thres1=0.9, thres2=0.8, thres3=0.85, thres_small=128, scale_factors=(1.0, 1.0, 1.0),
                 remove_small_mode='background', seed_thres=32, return_seed=False, precomputed_seed=None):
    r"""Convert binary foreground probability maps and instance contours to
    instance masks via watershed segmentation algorithm.

    Note:
        This function uses the `skimage.segmentation.watershed <https://github.com/scikit-image/scikit-image/blob/master/skimage/segmentation/_watershed.py#L89>`_
        function that converts the input image into ``np.float64`` data type for processing. Therefore please make sure enough memory is allocated when handling large arrays.

    Args:
        volume (numpy.ndarray): foreground and contour probability of shape :math:`(C, Z, Y, X)`.
        thres1 (float): threshold of seeds. Default: 0.9
        thres2 (float): threshold of instance contours. Default: 0.8
        thres3 (float): threshold of foreground. Default: 0.85
        thres_small (int): size threshold of small objects to remove. Default: 128
        scale_factors (tuple): scale factors for resizing in :math:`(Z, Y, X)` order. Default: (1.0, 1.0, 1.0)
        remove_small_mode (str): ``'background'``, ``'neighbor'`` or ``'none'``. Default: ``'background'``
    """
    assert volume.shape[0] == 2
    semantic = volume[0]
    boundary = volume[1]
    foreground = (semantic > int(255*thres3))

    if precomputed_seed is not None:
        seed = precomputed_seed
    else: # compute the instance seeds
        seed_map = (semantic > int(255*thres1)) * (boundary < int(255*thres2))
        seed = label(seed_map)
        seed = remove_small_objects(seed, seed_thres)

    segm = watershed(-semantic.astype(np.float64), seed, mask=foreground)
    segm = remove_small_instances(segm, thres_small, remove_small_mode)

    if not all(x==1.0 for x in scale_factors):
        target_size = (int(semantic.shape[0]*scale_factors[0]),
                       int(semantic.shape[1]*scale_factors[1]),
                       int(semantic.shape[2]*scale_factors[2]))
        segm = resize(segm, target_size, order=0, anti_aliasing=False, preserve_range=True)

    if not return_seed:
        return cast2dtype(segm)

    return cast2dtype(segm), seed


def bcd_watershed(volume, thres1=0.9, thres2=0.8, thres3=0.85, thres4=0.5, thres5=0.0, thres_small=128,
                  scale_factors=(1.0, 1.0, 1.0), remove_small_mode='background', seed_thres=32, return_seed=False,
                  precomputed_seed=None):
    r"""Convert binary foreground probability maps, instance contours and signed distance
    transform to instance masks via watershed segmentation algorithm.

    Note:
        This function uses the `skimage.segmentation.watershed <https://github.com/scikit-image/scikit-image/blob/master/skimage/segmentation/_watershed.py#L89>`_
        function that converts the input image into ``np.float64`` data type for processing. Therefore please make sure enough memory is allocated when handling large arrays.

    Args:
        volume (numpy.ndarray): foreground and contour probability of shape :math:`(C, Z, Y, X)`.
        thres1 (float): threshold of seeds. Default: 0.9
        thres2 (float): threshold of instance contours. Default: 0.8
        thres3 (float): threshold of foreground. Default: 0.85
        thres4 (float): threshold of signed distance for locating seeds. Default: 0.5
        thres5 (float): threshold of signed distance for foreground. Default: 0.0
        thres_small (int): size threshold of small objects to remove. Default: 128
        scale_factors (tuple): scale factors for resizing in :math:`(Z, Y, X)` order. Default: (1.0, 1.0, 1.0)
        remove_small_mode (str): ``'background'``, ``'neighbor'`` or ``'none'``. Default: ``'background'``
    """
    assert volume.shape[0] == 3
    semantic, boundary, distance = volume[0], volume[1], volume[2]
    distance = (distance / 255.0) * 2.0 - 1.0
    foreground = (semantic > int(255*thres3)) * (distance > thres5)

    if precomputed_seed is not None:
        seed = precomputed_seed
    else: # compute the instance seeds
        seed_map = (semantic > int(255*thres1)) * (boundary < int(255*thres2)) * (distance > thres4)
        seed = label(seed_map)
        seed = remove_small_objects(seed, seed_thres)

    segm = watershed(-semantic.astype(np.float64), seed, mask=foreground)
    segm = remove_small_instances(segm, thres_small, remove_small_mode)

    if not all(x==1.0 for x in scale_factors):
        target_size = (int(semantic.shape[0]*scale_factors[0]),
                       int(semantic.shape[1]*scale_factors[1]),
                       int(semantic.shape[2]*scale_factors[2]))
        segm = resize(segm, target_size, order=0, anti_aliasing=False, preserve_range=True)

    if not return_seed:
        return cast2dtype(segm)

    return cast2dtype(segm), seed


# Post-processing functions for synaptic polarity model outputs as described
# in "Two-Stream Active Query Suggestion for Active Learning in Connectomics
# (ECCV 2020, https://zudi-lin.github.io/projects/#two_stream_active)".
def polarity2instance(volume: np.ndarray, thres: float=0.5, thres_small: int=128, 
                      scale_factors: tuple=(1.0, 1.0, 1.0), semantic: bool=False, dilate_sz: int=5):
    r"""From synaptic polarity prediction to instance masks via connected-component
    labeling. The input volume should be a 3-channel probability map of shape :math:`(C, Z, Y, X)`
    where :math:`C=3`, representing pre-synaptic region, post-synaptic region and their
    union, respectively.

    Note:
        For each pair of pre- and post-synaptic segmentation, the decoding function will
        annotate pre-synaptic region as :math:`2n-1` and post-synaptic region as :math:`2n`,
        for :math:`n>0`. If :attr:`semantic=True`, all pre-synaptic pixels are labeled with
        while all post-synaptic pixels are labeled with 2. Both kinds of annotation are compatible
        with the ``TARGET_OPT: ['1']`` configuration in training.

    Note:
        The number of pre- and post-synaptic segments will be reported when setting :attr:`semantic=False`.
        Note that the numbers can be different due to either incomplete syanpses touching the volume borders,
        or errors in the prediction. We thus make a conservative estimate of the total number of synapses
        by using the relatively small number among the two.

    Args:
        volume (numpy.ndarray): 3-channel probability map of shape :math:`(3, Z, Y, X)`.
        thres (float): probability threshold of foreground. Default: 0.5
        thres_small (int): size threshold of small objects to remove. Default: 128
        scale_factors (tuple): scale factors for resizing the output volume in :math:`(Z, Y, X)` order. Default: :math:`(1.0, 1.0, 1.0)`
        semantic (bool): return only the semantic mask of pre- and post-synaptic regions. Default: False
        dilate_sz (int): define a struct of size (1, dilate_sz, dilate_sz) to dilate the masks. Default: 5

    Examples::
        >>> from connectomics.data.utils import readvol, savevol
        >>> from connectomics.utils.processing import polarity2instance
        >>> volume = readvol(input_name)
        >>> instances = polarity2instance(volume)
        >>> savevol(output_name, instances)
    """
    thres = int(255.0 * thres)
    temp = (volume > thres) # boolean array

    syn_pre = np.logical_and(temp[0], temp[2])
    syn_pre = remove_small_objects(syn_pre,
                min_size=thres_small, connectivity=1)
    syn_post = np.logical_and(temp[1], temp[2])
    syn_post = remove_small_objects(syn_post,
                min_size=thres_small, connectivity=1)

    if semantic:
        # Generate only the semantic mask. The pre-synaptic region is labeled
        # with 1, while the post-synaptic region is labeled with 2.
        segm = np.maximum(syn_pre.astype(np.uint8),
                          syn_post.astype(np.uint8) * 2)

    else:# Generate the instance mask.
        # The pre- and post-synaptic masks may not touch each other. Dilating the 
        # union masks to define each synapse instance.
        foreground = binary_dilation(temp[2], np.ones((1,dilate_sz,dilate_sz), bool))
        foreground = label(foreground)

        # Since non-zero pixels in seg_pos and seg_neg are subsets of temp[2],
        # they are naturally subsets of non-zero pixels in foreground.
        seg_pre = (foreground*2 - 1) * syn_pre.astype(foreground.dtype)
        seg_post = (foreground*2) * syn_post.astype(foreground.dtype)
        segm = np.maximum(seg_pre, seg_post)

        # Report the number of synapses
        num_pre = len(np.unique(seg_pre))-1
        num_post = len(np.unique(seg_post))-1
        num_syn = min(num_pre, num_post) # a conservative estimate
        print(f"Stats: found {num_pre} pre- and {num_post} post-synaptic segments.")
        print(f"There are {num_syn} synapses under a conservative estimate.")

    # resize the segmentation based on specified scale factors
    if not all(x==1.0 for x in scale_factors):
        target_size = (int(segm.shape[0]*scale_factors[0]),
                       int(segm.shape[1]*scale_factors[1]),
                       int(segm.shape[2]*scale_factors[2]))
        segm = resize(segm, target_size, order=0, anti_aliasing=False, preserve_range=True)

    return cast2dtype(segm)


# utils for post-processing
def binarize_and_median(pred, size=(7,7,7), thres=0.8):
    """First binarize the prediction with a given threshold, and
    then conduct median filtering to reduce noise.

    pred (numpy.ndarray): predicted foreground probability within (0,1).
    size (tuple): kernal size of filtering. Default: (7,7,7)
    thres (float): threshold for binarizing the prediction. Default: 0.8
    """
    pred = (pred > thres).astype(np.uint8)
    pred = ndimage.median_filter(pred, size=size)
    return pred


def remove_small_instances(segm: np.ndarray,
                           thres_small: int = 128,
                           mode: str = 'background'):
    """Remove small spurious instances.
    """
    assert mode in ['none',
                    'background',
                    'background_2d',
                    'neighbor',
                    'neighbor_2d']

    if mode == 'none':
        return segm

    # The function remove_small_objects expects ar to be an array with labeled objects, and 
    # removes objects smaller than min_size. If ar is bool, the image is first labeled. This 
    # leads to potentially different behavior for bool and 0-and-1 arrays. Reference:
    # https://scikit-image.org/docs/stable/api/skimage.morphology.html#remove-small-objects
    if mode == 'background':
        return remove_small_objects(segm, thres_small)
    elif mode == 'background_2d':
        temp = [remove_small_objects(segm[i], thres_small)
                for i in range(segm.shape[0])]
        return np.stack(temp, axis=0)

    if mode == 'neighbor':
        return merge_small_objects(segm, thres_small, do_3d=True)
    elif mode == 'neighbor_2d':
        temp = [merge_small_objects(segm[i], thres_small)
                for i in range(segm.shape[0])]
        return np.stack(temp, axis=0)


def merge_small_objects(segm, thres_small, do_3d=False):
    struct = np.ones((1,3,3)) if do_3d else np.ones((3,3))
    indices, counts = np.unique(segm, return_counts=True)

    for i in range(len(indices)):
        idx = indices[i]
        if counts[i] < thres_small:
            temp = (segm == idx).astype(np.uint8)
            coord = bbox_ND(temp, relax=2)
            cropped = crop_ND(temp, coord)

            diff = dilation(cropped, struct) - cropped
            diff_segm = crop_ND(segm, coord)
            diff_segm[np.where(diff==0)]=0

            u, ct = np.unique(diff_segm, return_counts=True)
            if len(u) > 1 and u[0] == 0:
                u, ct = u[1:], ct[1:]

            segm[np.where(segm==idx)] = u[np.argmax(ct)]

    return segm


def remove_large_instances(segm: np.ndarray,
                           max_size: int = 2000):
    """Remove large instances given a maximum size threshold.
    """
    out = np.copy(segm)
    component_sizes = np.bincount(segm.ravel())
    too_large = component_sizes > max_size
    too_large_mask = too_large[segm]
    out[too_large_mask] = 0
    return out


def cast2dtype(segm):
    """Cast the segmentation mask to the best dtype to save storage.
    """
    max_id = np.amax(np.unique(segm))
    m_type = getSegType(int(max_id))
    return segm.astype(m_type)


def stitch_3d(masks, stitch_threshold=0.25):
    r""" Takes a volume stack of 2D annotations and stitches into 3D annotations using IOU.

    Args:
        mask (numpy.ndarray): 3D volume comprised of a 2D annotations stack of shape :math:`(Z, Y, X)`.
        stitch_threshold (float): threshold for joining 2D annotations via IOU. Default: 0.25

    """
    mmax = masks[0].max()
    empty = 0
    
    for i in range(len(masks)-1):
        # retrive all intersecting pairs, discard background
        iou = intersection_over_union(masks[i+1], masks[i])[1:,1:]
        if not iou.size and empty == 0:
            mmax = masks[i+1].max()
        elif not iou.size and not empty == 0:
            icount = masks[i+1].max()
            istitch = np.arange(mmax+1, mmax + icount+1, 1, int)
            mmax += icount
            istitch = np.append(np.array(0), istitch)
            masks[i+1] = istitch[masks[i+1]]
        else:
            # set all iou value that did not breach the threshold to zero
            iou[iou < stitch_threshold] = 0.0
            # we calculated the IoU for each possible masks pair
            # for each mask only consider the pairing with the greatest IoU 
            iou[iou < iou.max(axis=0)] = 0.0
            istitch = iou.argmax(axis=1) + 1
            ino = np.nonzero(iou.max(axis=1)==0.0)[0]
            istitch[ino] = np.arange(mmax+1, mmax+len(ino)+1, 1, int)
            mmax += len(ino)
            istitch = np.append(np.array(0), istitch)
            masks[i+1] = istitch[masks[i+1]]
            empty = 1
            
    return masks


# Abducted from the cellpose repository (https://github.com/MouseLand/cellpose/blob/master/cellpose/metrics.py).
def intersection_over_union(masks_true, masks_pred):
    """ Calculates the intersection over union for all mask pairs.
    
    Args:
        x (numpy.ndarray): 2D label array where 0=NO masks; 1,2... are mask labels, shape :math: `(Y, X)`.
        y (numpy.ndarray): 2D label array where 0=NO masks; 1,2... are mask labels, shape :math: `(Y, X)`.

    Return:
        A ND-array recording the IoU score (flaot) for each label pair, size [x.max()+1, y.max()+1]
    """

    overlap = _label_overlap(masks_true, masks_pred)

    # index vise encoding of how often a predicted label coincides with true labels
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    # index vise encoding of how often a true label coincides with predicted labels
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    iou[np.isnan(iou)] = 0.0
    return iou


def _label_overlap(x, y):
    """ Creates a look up table that records the pixel overlap
        between two 2D label arryes.
        
    Args:
        x (numpy.ndarray): 2D label array where 0=NO masks; 1,2... are mask labels, shape :math: `(Y, X)`.
        y (numpy.ndarray): 2D label array where 0=NO masks; 1,2... are mask labels, shape :math: `(Y, X)`.
        
    
    Returns
        A ND-array matrix recording the pixel overlaps, size :math: `[x.max()+1, y.max()+1]`    
    """
    # flatten the 2D label arryes 
    x = x.ravel()
    y = y.ravel()
    
    assert len(x) == len(y), f"The label masks must have the same shape" 
    
    # initialize the lookup tabel
    overlap = np.zeros((1+x.max(),1+y.max()), dtype=np.uint)
    
    # loop over the labels in x and add to the corresponding
    # overlap entry. If label A in x and label B in y share P
    # pixels, then the resulting overlap is P
    for i in range(len(x)):
        overlap[x[i],y[i]] += 1
    return overlap


def remove_masks(vol: np.ndarray, indices: List[int]) -> np.ndarray:
    """Remove objects by indices from a segmentation volume.
    """
    for idx in indices:
        vol[np.where(vol==idx)] = 0
    return vol


def add_masks(vol_base: np.ndarray, vol: np.ndarray, indices: List[int]) -> np.ndarray:
    """Add the instances in a new segmentation volume to the 
    original one. A new instance can overwrite existing object
    pixels if the corresponding region contains non-background.
    """
    max_idx = max(np.unique(vol_base))
    for i, idx in enumerate(indices):
        vol_base[np.where(vol==idx)] = max_idx+i+1
    return vol_base


def merge_fmasks(vol: np.ndarray, indices: List[List[int]]) -> np.ndarray:
    """Merge two or more masks into a single one.
    """
    for merges in indices:
        temp = np.zeros_like(vol)
        for i, idx in enumerate(merges):
            if i == 0:
                main_idx = idx
            temp = temp + (vol==idx).astype(temp.dtype)
        vol[np.where(temp!=0)] = main_idx
    return vol  


def watershed_split(vol: np.ndarray, index: int, show_id: bool = False,
                    min_distance: int = 5) -> np.ndarray:
    """Apply watershed transform to split an 3D object into two or more 
    parts based on the given index.
    """
    assert vol.ndim == 3 # 3D label array
    max_idx = max(np.unique(vol))
    binary = (vol == index)
    bbox = bbox_ND(binary, relax=1) # avoid cropped object touching borders
    cropped = crop_ND(binary, bbox, end_included=True)

    # see https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html 
    distance = ndimage.distance_transform_edt(cropped)
    coords = peak_local_max(distance, min_distance=min_distance, labels=cropped)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers = label(mask)
    split_objects = watershed(-distance, markers, mask=cropped)

    seg_id = np.unique(split_objects)
    new_id = []
    if seg_id[0] == 0: seg_id = seg_id[1:] # ignore background pixels
    for i, idx in enumerate(seg_id):
        split_objects[np.where(split_objects==idx)] = max_idx + i + 1
        new_id.append(max_idx + i + 1)
    if show_id: print(new_id)

    vol = replace_ND(vol, split_objects, bbox, end_included=True)
    return vol
