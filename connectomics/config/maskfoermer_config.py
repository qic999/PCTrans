# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_maskformer2_config(cfg):
    """
    Add config for MASK_FORMER.
    """
    # NOTE: configs from original maskformer
    # data config
    # select the dataset mapper
    # cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"
    # Color augmentation
    # cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    # cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    # cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1
    # Used for `poly` learning rate schedule.
    cfg.SOLVER.POLY_LR_POWER = 0.9
    cfg.SOLVER.POLY_LR_CONSTANT_ENDING = 0.0


    # mask_former model config
    cfg.MODEL.BACKBONE = CN()
    cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"
    # Freeze the first several stages so they are not trained.
    # There are 5 stages in ResNet. The first is a convolution, and the following
    # stages are each group of residual blocks.
    cfg.MODEL.BACKBONE.FREEZE_AT = 2


    cfg.MODEL.WEIGHTS = ""
    # Values to be used for image normalization (BGR order, since INPUT.FORMAT defaults to BGR).
    # To train on images of different number of channels, just set different mean & std.
    # Default values are the mean pixel value from ImageNet: [103.53, 116.28, 123.675]
    cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]
    # When using pre-trained models in Detectron1 or any MSRA models,
    # std has been absorbed into its conv1 weights, so the std needs to be set 1.
    # Otherwise, you can use [57.375, 57.120, 58.395] (ImageNet std)
    cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]

    # ---------------------------------------------------------------------------- #
    # ResNe[X]t options (ResNets = {ResNet, ResNeXt}
    # Note that parts of a resnet may be used for both the backbone and the head
    # These options apply to both
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.RESNETS = CN()
    # deeplab的参数：STEM_TYPE、RES5_MULTI_GRID
    cfg.MODEL.RESNETS.STEM_TYPE = "basic"
    cfg.MODEL.RESNETS.RES5_MULTI_GRID = [1, 1, 1]

    cfg.MODEL.RESNETS.DEPTH = 50
    cfg.MODEL.RESNETS.OUT_FEATURES = ["res4"]  # res4 for C4 backbone, res2..5 for FPN backbone

    # Number of groups to use; 1 ==> ResNet; > 1 ==> ResNeXt
    cfg.MODEL.RESNETS.NUM_GROUPS = 1

    # Options: FrozenBN, GN, "SyncBN", "BN"
    cfg.MODEL.RESNETS.NORM = "FrozenBN"

    # Baseline width of each group.
    # Scaling this parameters will scale the width of all bottleneck layers.
    cfg.MODEL.RESNETS.WIDTH_PER_GROUP = 64

    # Place the stride 2 conv on the 1x1 filter
    # Use True only for the original MSRA ResNet; use False for C2 and Torch models
    cfg.MODEL.RESNETS.STRIDE_IN_1X1 = True

    # Apply dilation in stage "res5"
    cfg.MODEL.RESNETS.RES5_DILATION = 1

    # Output width of res2. Scaling this parameters will scale the width of all 1x1 convs in ResNet
    # For R18 and R34, this needs to be set to 64
    cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
    cfg.MODEL.RESNETS.STEM_OUT_CHANNELS = 64

    # Apply Deformable Convolution in stages
    # Specify if apply deform_conv on Res2, Res3, Res4, Res5
    cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE = [False, False, False, False]
    # Use True to use modulated deform_conv (DeformableV2, https://arxiv.org/abs/1811.11168);
    # Use False for DeformableV1.
    cfg.MODEL.RESNETS.DEFORM_MODULATED = False
    # Number of groups in deformable conv.
    cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS = 1


    # mask_former model config
    cfg.MODEL.MASK_FORMER = CN()

    # loss
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0
    cfg.MODEL.MASK_FORMER.REID_WEIGHT_QUERY = 2.0
    cfg.MODEL.MASK_FORMER.REID_WEIGHT_MASK = 2.0
    cfg.MODEL.MASK_FORMER.EMB_WEIGHT = 2.0
    cfg.MODEL.MASK_FORMER.REF_POINTS_WEIGHT = 5.0
    cfg.MODEL.MASK_FORMER.SEM_WEIGHT = 5.0
    cfg.MODEL.MASK_FORMER.SEMANTIC_LOSS_ON = False
    cfg.MODEL.MASK_FORMER.SEMANTIC_NORM = 'BN'
    cfg.MODEL.MASK_FORMER.REL_COORD = True

    # transformer config
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.PRE_NORM = False

    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100
    cfg.MODEL.MASK_FORMER.POSITION_POINTS_NUM = 1

    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False

    # mask_former inference config
    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    # mask_former model config
    cfg.MODEL.SEM_SEG_HEAD = CN()
    cfg.MODEL.SEM_SEG_HEAD.NAME = ''
    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE = -1
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 2
    cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT = 1.0
    cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM = 128
    cfg.MODEL.SEM_SEG_HEAD.NORM = "GN"
    cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE = 4
    cfg.MODEL.SEM_SEG_HEAD.ATTENTION_MASK_THRESHOLD = 0.5

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    # NOTE: maskformer2 extra configs
    # transformer module
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"

    # LSJ aug
    # cfg.INPUT.IMAGE_SIZE = 1024
    # cfg.INPUT.MIN_SCALE = 0.1
    # cfg.INPUT.MAX_SCALE = 2.0

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75

    # ---------------------------------------------------------------------------- #
    # Specific test options
    # ---------------------------------------------------------------------------- #
    cfg.TEST = CN()
    cfg.TEST.DETECTIONS_PER_IMAGE = 100

    cfg.TEST.THRESHOLD = 0.5
