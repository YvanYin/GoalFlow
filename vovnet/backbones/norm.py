from typing import Any, Dict, Optional, Tuple, Type

import torch.nn as nn

__all__ = ["REGISTERED_NORMALIZATION_DICT", "build_norm"]

# register normalization function here
#   name: module, kwargs with default values
REGISTERED_NORMALIZATION_DICT: Dict[str, Tuple[Type, Dict[str, Any]]] = {
    "bn_3d": (nn.BatchNorm3d, {"num_features": None, "eps": 1e-5, "momentum": 0.1}),
    "bn_2d": (nn.BatchNorm2d, {"num_features": None, "eps": 1e-5, "momentum": 0.1}),
    "bn_1d": (nn.BatchNorm1d, {"num_features": None, "eps": 1e-5, "momentum": 0.1}),
    "sync_bn": (nn.SyncBatchNorm, {"num_features": None, "eps": 1e-5, "momentum": 0.1}),
    "gn": (nn.GroupNorm, {"num_groups": None, "num_channels": None, "eps": 1e-5}),
    "ln": (nn.LayerNorm, {"normalized_shape": None, "eps": 1e-5}),
}


def build_norm(norm_name="bn_2d", num_features=None, **kwargs) -> Optional[nn.Module]:
    if norm_name == "gn":
        kwargs["num_channels"] = num_features
    elif norm_name == "ln":
        kwargs["normalized_shape"] = num_features
    else:
        kwargs["num_features"] = num_features
    if norm_name in REGISTERED_NORMALIZATION_DICT:
        norm_module, default_args = REGISTERED_NORMALIZATION_DICT[norm_name]
        for key in default_args:
            if key in kwargs:
                default_args[key] = kwargs[key]
        return norm_module(**default_args)
    elif norm_name is None or norm_name.lower() == "none":
        return None
    else:
        raise ValueError("do not support: %s" % norm_name)
