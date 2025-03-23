# -*- coding: utf-8 -*-
# Copyright (c) Youngwan Lee (ETRI) All Rights Reserved.

from vovnet.config_.config_ import CfgNode as CN


def add_vovnet_config(cfg):
    """
    Add config for VoVNet.
    """
    _C = cfg

    _C.MODEL.VOVNET = CN()

    _C.MODEL.VOVNET.CONV_BODY = "V-39-eSE"
    _C.MODEL.VOVNET.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]

    # Options: FrozenBN, GN, "SyncBN", "BN"
    _C.MODEL.VOVNET.NORM = "FrozenBN"

    _C.MODEL.VOVNET.OUT_CHANNELS = 256

    _C.MODEL.VOVNET.BACKBONE_OUT_CHANNELS = 256

    return _C
