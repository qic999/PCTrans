import numpy as np
import torch
import torch.nn as nn

from .arch import UNet3D, UNet2D, FPN3D, DeepLabV3, UNetPlus3D, UNetPlus2D, unet_residual_3d
from .backbone import RepVGG3D

MODEL_MAP = {
    'unet_3d': UNet3D,
    'unet_2d': UNet2D,
    'fpn_3d': FPN3D,
    'unet_plus_3d': UNetPlus3D,
    'unet_plus_2d': UNetPlus2D,
    'deeplabv3a': DeepLabV3,
    'deeplabv3b': DeepLabV3,
    'deeplabv3c': DeepLabV3,
    # 'MaskFormer': MaskFormer,
    # 'VideoMaskFormer': VideoMaskFormer,
    'unet_residual_3d': unet_residual_3d
}


def build_model(cfg, device, rank=None):

    model_arch = cfg.MODEL.ARCHITECTURE
    # assert model_arch in MODEL_MAP.keys()
    kwargs = {
        'block_type': cfg.MODEL.BLOCK_TYPE,
        'in_channel': cfg.MODEL.IN_PLANES,
        'out_channel': cfg.MODEL.OUT_PLANES,
        'filters': cfg.MODEL.FILTERS,
        'ks': cfg.MODEL.KERNEL_SIZES,
        'blocks': cfg.MODEL.BLOCKS,
        'attn': cfg.MODEL.ATTENTION,
        'is_isotropic': cfg.DATASET.IS_ISOTROPIC,
        'isotropy': cfg.MODEL.ISOTROPY,
        'pad_mode': cfg.MODEL.PAD_MODE,
        'act_mode': cfg.MODEL.ACT_MODE,
        'norm_mode': cfg.MODEL.NORM_MODE,
        'pooling': cfg.MODEL.POOLING_LAYER,
        'return_feats': cfg.MODEL.RETURN_FEATS,
    }

    if model_arch == 'fpn_3d':
        kwargs['backbone_type'] = cfg.MODEL.BACKBONES
        kwargs['deploy'] = cfg.MODEL.DEPLOY_MODE
        if cfg.MODEL.BACKBONES == 'botnet':
            kwargs['fmap_size'] = cfg.MODEL.INPUT_SIZE

    if model_arch[:7] == 'deeplab':
        kwargs['name'] = model_arch
        kwargs['backbone_type'] = cfg.MODEL.BACKBONES
        kwargs['aux_out'] = cfg.MODEL.AUX_OUT

    if model_arch == 'unet_residual_3d': # unet_residual_3d
        model = MODEL_MAP[cfg.MODEL.ARCHITECTURE](in_channel=cfg.MODEL.IN_PLANES, out_channel=cfg.MODEL.OUT_PLANES, filters=cfg.MODEL.FILTERS, \
                                             pad_mode=cfg.MODEL.PAD_MODE, norm_mode=cfg.MODEL.NORM_MODE, act_mode=cfg.MODEL.ACT_MODE,
                                             do_embedding=(cfg.MODEL.EMBEDDING==1), head_depth=cfg.MODEL.HEAD_DEPTH, output_act='sigmoid')
        print('model: ', model.__class__.__name__)
        return make_parallel(model, cfg, device, rank, find_unused_parameters=True)
    elif model_arch == 'MaskFormer':
        from detectron2.modeling import META_ARCH_REGISTRY, build_model
        model = META_ARCH_REGISTRY.get(model_arch)(cfg)
        print('model: ', model_arch)
        return make_parallel(model, cfg, device, rank)
    else:
        model = MODEL_MAP[cfg.MODEL.ARCHITECTURE](**kwargs)
        print('model: ', model.__class__.__name__)
        return make_parallel(model, cfg, device, rank)
    
    


def make_parallel(model, cfg, device, rank=None, find_unused_parameters=False):
    if cfg.SYSTEM.PARALLEL == 'DDP':
        print('Parallelism with DistributedDataParallel.')
        # Currently SyncBatchNorm only supports DistributedDataParallel (DDP)
        # with single GPU per process. Use torch.nn.SyncBatchNorm.convert_sync_batchnorm()
        # to convert BatchNorm*D layer to SyncBatchNorm before wrapping Network with DDP.
        if cfg.MODEL.NORM_MODE == "sync_bn":
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        model = model.to(device)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        assert rank is not None
        find_unused_parameters=True
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], output_device=rank, find_unused_parameters=find_unused_parameters)

    elif cfg.SYSTEM.PARALLEL == 'DP':
        gpu_device_ids = list(range(cfg.SYSTEM.NUM_GPUS))
        print('Parallelism with DataParallel.')
        model = nn.DataParallel(model, device_ids=gpu_device_ids)
        model = model.to(device)

    else:
        print('No parallelism across multiple GPUs.')
        model = model.to(device)

    return model.to(device)


def update_state_dict(cfg, model_dict, mode='train'):
    r"""Update state_dict based on the config options.
    """
    assert mode in ['train', 'test']

    arch = cfg.MODEL.ARCHITECTURE
    backbone = cfg.MODEL.BACKBONES
    if arch == 'fpn_3d' and backbone == 'repvgg' and mode == 'test':
        # convert to deploy mode weights for RepVGG3D backbone
        model_dict = RepVGG3D.repvgg_convert_as_backbone(model_dict)

    if 'n_averaged' in model_dict.keys():
        print("Number of models averaged in SWA: ", model_dict['n_averaged'])

    return model_dict
