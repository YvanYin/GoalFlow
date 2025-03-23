from typing import Any, Dict, Optional, Tuple, Type, Union

from torch import nn

__all__ = ["build_activation"]

# register activation function here
#   name: module, kwargs with default values
REGISTERED_ACT_DICT: Dict[str, Tuple[Type, Dict[str, Any]]] = {
    "relu": (nn.ReLU, {"inplace": True}),
    "relu6": (nn.ReLU6, {"inplace": True}),
    "leaky_relu": (nn.LeakyReLU, {"inplace": True, "negative_slope": 0.1}),
    "h_swish": (nn.Hardswish, {"inplace": True}),
    "h_sigmoid": (nn.Hardsigmoid, {"inplace": True}),
    "swish": (nn.SiLU, {"inplace": True}),
    "silu": (nn.SiLU, {"inplace": True}),
    "tanh": (nn.Tanh, {}),
    "sigmoid": (nn.Sigmoid, {}),
    "gelu": (nn.GELU, {}),
    "mish": (nn.Mish, {"inplace": True}),
}


def build_activation(act_func_name: Union[str, nn.Module], **kwargs) -> Optional[nn.Module]:
    if isinstance(act_func_name, nn.Module):
        return act_func_name
    if act_func_name in REGISTERED_ACT_DICT:
        act_module, default_args = REGISTERED_ACT_DICT[act_func_name]
        for key in default_args:
            if key in kwargs:
                default_args[key] = kwargs[key]
        return act_module(**default_args)
    elif act_func_name is None or act_func_name.lower() == "none":
        return None
    else:
        raise ValueError("do not support: %s" % act_func_name)
