from typing import Union, Tuple, List, Dict, Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

from vovnet.backbones.utils import get_same_padding, val2list, make_divisible, list_sum
from vovnet.backbones.norm import build_norm
from vovnet.backbones.act import build_activation


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        use_bias=False,
        norm="bn_2d",
        act_func="relu",
        first_layer=False,
        last_layer=False,
    ):
        super(ConvLayer, self).__init__()

        padding = get_same_padding(kernel_size)
        padding *= dilation

        self.first_layer = first_layer
        self.last_layer = last_layer
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
        )
        self.norm = build_norm(norm, num_features=out_channels)
        self.act = build_activation(act_func)

    @property
    def in_channels(self):
        return self.conv.in_channels

    @property
    def out_channels(self):
        return self.conv.out_channels

    @property
    def kernel_size(self):
        return self.conv.kernel_size[0]

    @property
    def stride(self):
        return self.conv.stride[0]

    @property
    def dilation(self):
        return self.conv.dilation[0]

    @property
    def groups(self):
        return self.conv.groups

    @property
    def use_bias(self):
        return self.conv.bias is not None

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x

    @staticmethod
    def _parse_config(config: dict):
        if config["norm"] is None:
            norm_name = None
        elif "norm_name" in config["norm"]:
            norm_name = config["norm"]["norm_name"]
        elif config["norm"]["name"] == "BatchNorm2d":
            norm_name = "bn_2d"
        else:
            norm_name = None
        if config["act"] is None:
            act_func_name = None
        elif "act_func_name" in config["act"]:
            act_func_name = config["act"]["act_func_name"]
        elif config["act"]["name"] == "ReLU":
            act_func_name = "relu"
        elif config["act"]["name"] == "ReLU6":
            act_func_name = "relu6"
        else:
            act_func_name = None
        return config, norm_name, act_func_name


class PoolingLayer(nn.Module):
    def __init__(
        self,
        pool_type="global_avg",
        kernel_size=2,
        stride=2,
        padding=0,
        ceil_mode=False,
        in_channels=None,
    ):
        super(PoolingLayer, self).__init__()
        self.pool_type = pool_type
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.in_channels = in_channels

        if pool_type == "global_avg":
            self.pooling = nn.AdaptiveAvgPool2d(output_size=1)
        elif pool_type == "avg":
            self.pooling = nn.AvgPool2d(kernel_size, stride, padding, ceil_mode=ceil_mode)
        elif pool_type == "max":
            self.pooling = nn.MaxPool2d(kernel_size, stride, padding, ceil_mode=ceil_mode)
        else:
            self.pooling = None

    def forward(self, x):
        if self.pooling:
            x = self.pooling(x)
        return x


class SPPBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        pool_size: List[int],
        pool_type: str = "max",
        kernel_size=1,
        act_func="relu6",
    ):
        super(SPPBlock, self).__init__()

        self.pyramid_pooling = nn.ModuleList(
            [
                PoolingLayer(
                    pool_type=pool_type,
                    kernel_size=p_size,
                    stride=1,
                    padding=get_same_padding(p_size),
                    ceil_mode=False,
                )
                for p_size in pool_size
            ]
        )
        self.merge_conv = ConvLayer(
            in_channels=in_channels * (len(pool_size) + 1),
            out_channels=in_channels,
            kernel_size=kernel_size,
            act_func=act_func,
        )

    @property
    def in_channels(self):
        return self.merge_conv.out_channels

    @property
    def pool_size(self):
        return [pool.kernel_size for pool in self.pyramid_pooling]

    @property
    def pool_type(self):
        return self.pyramid_pooling[0].pool_type

    @property
    def kernel_size(self):
        return self.merge_conv.kernel_size

    def forward(self, x):
        outs = [x]
        for pool in self.pyramid_pooling:
            outs.append(pool(x))
        outs = torch.cat(outs, dim=1)
        outs = self.merge_conv(outs)
        return outs


class UpSampleLayer(nn.Module):
    def __init__(
        self,
        mode="bilinear",
        size: Union[int, Tuple[int, int], List[int], None] = None,
        factor=2,
        align_corners=True,
    ):
        super(UpSampleLayer, self).__init__()
        self.mode = mode
        self.size = val2list(size, 2) if size is not None else None
        self.factor = None if self.size is not None else factor
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "repeat":
            if self.size is not None:
                raise NotImplementedError
            return x.repeat_interleave(repeats=self.factor, dim=-1).repeat_interleave(
                repeats=self.factor, dim=-2
            )
        elif self.mode in {"linear", "bilinear", "bicubic", "trilinear"}:
            return F.interpolate(
                x,
                size=self.size,
                scale_factor=self.factor,
                mode=self.mode,
                align_corners=self.align_corners,
            )
        elif self.mode in {"nearest", "area"}:
            return F.interpolate(x, size=self.size, scale_factor=self.factor, mode=self.mode)
        else:
            raise NotImplementedError("Upsample(mode=%s) not implemented." % self.mode)

    def __repr__(self) -> str:
        return (
            f"Upsample(mode={self.mode}, size={self.size}, factor={self.factor}, "
            f"align_corners={self.align_corners})"
        )


class MBV1Block(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size=3, stride=1, act_func=("relu6", None)
    ):
        super(MBV1Block, self).__init__()

        act_func = val2list(act_func, 2)
        self.depth_conv = ConvLayer(
            in_channels, in_channels, kernel_size, stride, groups=in_channels, act_func=act_func[0]
        )
        self.point_conv = ConvLayer(in_channels, out_channels, 1, act_func=act_func[1])

    @property
    def in_channels(self):
        return self.depth_conv.in_channels

    @property
    def out_channels(self):
        return self.point_conv.out_channels

    @property
    def kernel_size(self):
        return self.depth_conv.kernel_size

    @property
    def stride(self):
        return self.depth_conv.stride

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class MBV2Block(nn.Module):
    """A MobileNetV2 block as used in the MobileNetV2 architecture."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        act_func=("relu6", "relu6", None),
    ):
        super(MBV2Block, self).__init__()

        self.act_func = val2list(act_func, 3)
        if mid_channels is None:
            mid_channels = make_divisible(in_channels * expand_ratio, divisor=8)

        self.inverted_conv = ConvLayer(in_channels, mid_channels, 1, act_func=self.act_func[0])
        self.depth_conv = ConvLayer(
            mid_channels,
            mid_channels,
            kernel_size,
            stride,
            groups=mid_channels,
            act_func=self.act_func[1],
        )
        self.point_conv = ConvLayer(mid_channels, out_channels, 1, act_func=self.act_func[2])

    @property
    def in_channels(self):
        return self.inverted_conv.in_channels

    @property
    def out_channels(self):
        return self.point_conv.out_channels

    @property
    def kernel_size(self):
        return self.depth_conv.kernel_size

    @property
    def stride(self):
        return self.depth_conv.stride

    @property
    def mid_channels(self):
        return self.point_conv.in_channels

    @property
    def expand_ratio(self):
        return self.mid_channels / self.in_channels + 1e-10

    def forward(self, x):
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class FusedMBV2Block(nn.Module):
    """A MBV2 block with fused depthwise and channelwise convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        groups=1,
        act_func=("relu6", None),
    ):
        super(FusedMBV2Block, self).__init__()

        if isinstance(act_func, str):
            act_func = (act_func, None)
        self.act_func = val2list(act_func, 2)
        if mid_channels is None:
            mid_channels = make_divisible(in_channels * expand_ratio, divisor=8)

        self.spatial_conv = ConvLayer(
            in_channels, mid_channels, kernel_size, stride, groups=groups, act_func=self.act_func[0]
        )
        self.point_conv = ConvLayer(mid_channels, out_channels, 1, act_func=self.act_func[1])

    @property
    def in_channels(self):
        return self.spatial_conv.in_channels

    @property
    def out_channels(self):
        return self.point_conv.out_channels

    @property
    def kernel_size(self):
        return self.spatial_conv.kernel_size

    @property
    def stride(self):
        return self.spatial_conv.stride

    @property
    def mid_channels(self):
        return self.point_conv.in_channels

    @property
    def expand_ratio(self):
        return self.mid_channels / self.in_channels + 1e-10

    @property
    def groups(self):
        return self.spatial_conv.groups

    def forward(self, x):
        x = self.spatial_conv(x)
        x = self.point_conv(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, conv, shortcut, post_residual_act=None):
        super(ResidualBlock, self).__init__()

        self.conv = conv
        self.shortcut = shortcut
        self.post_residual_act = build_activation(post_residual_act)

    def forward(self, x):
        if self.conv is None:
            res = x
        elif self.shortcut is None:
            res = self.conv(x)
        else:
            res = self.conv(x) + self.shortcut(x)
            if self.post_residual_act:
                res = self.post_residual_act(res)
        return res


class DAGOp(nn.Module):
    def __init__(
        self,
        inputs: Dict,
        merge_mode: str,
        post_input_op: Optional[nn.Module],
        middle: nn.Module,
        outputs: Dict,
    ):
        super(DAGOp, self).__init__()

        self.input_keys = list(inputs.keys())
        self.input_ops = nn.ModuleList(list(inputs.values()))
        self.merge_mode = merge_mode
        self.post_input_op = post_input_op

        self.middle = middle

        self.output_keys = list(outputs.keys())
        self.output_ops = nn.ModuleList(list(outputs.values()))

    def forward(self, feature_dict: dict):
        feat = [op(feature_dict[key]) for key, op in zip(self.input_keys, self.input_ops)]

        if self.merge_mode == "cat":
            feat = self._cat_feat(feat)
        else:
            feat = self._add_feat(feat)

        if self.post_input_op is not None:
            feat = self.post_input_op(feat)

        feat = self.middle(feat)
        for key, op in zip(self.output_keys, self.output_ops):
            feature_dict[key] = op(feat)
        return feature_dict

    @staticmethod
    def _cat_feat(feat: list):
        return torch.cat(feat, dim=1)

    @staticmethod
    def _add_feat(feat: list):
        n, c, h, w = feat[0].shape[:4]
        for fi in feat[1:]:
            ci, hi, wi = fi.shape[1:4]
            assert hi == h and wi == w
            c = max(ci, c)
        for i, fi in enumerate(feat):
            if fi.size(1) < c:
                pad_tensor = torch.zeros((n, c - fi.size(1), h, w), device=fi.device)
                feat[i] = torch.cat([fi, pad_tensor], dim=1)
        return list_sum(feat)


class SeqBackbone(nn.Module):
    def __init__(self, input_stem: nn.Module, stages: List[nn.Module]) -> None:
        super().__init__()
        self.input_stem = input_stem
        self.stages = nn.ModuleList(stages)

    @property
    def n_stages(self) -> int:
        return len(self.stages)

    @property
    def stage_depth_info(self) -> Sequence[Optional[int]]:
        depths = []
        for stage in self.stages:
            if isinstance(stage, nn.Sequential):
                depth = len(stage)
            else:
                depth = None
            depths.append(depth)
        return depths

    @property
    def stage_width_info(self) -> List[Optional[int]]:
        widths = []
        for stage in self.stages:
            if isinstance(stage, nn.Sequential):
                block = stage[0]
            else:
                block = stage
            try:
                if isinstance(block, ResidualBlock):
                    width = block.conv.out_channels
                else:
                    width = block.out_channels
            except Exception:
                width = None
            widths.append(width)
        return widths

    def get_feature_dims(self, feature_id_list: Union[str, List[str]]):
        feature_dims = []
        for fid in val2list(feature_id_list):
            assert fid.startswith("stage")
            stage_id = int(fid[5:])
            if stage_id == 0:
                raise ValueError("Do not support using stage0 feature")
            else:
                feature_dims.append(self.stage_width_info[stage_id - 1])
        return squeeze_list(feature_dims)

    @property
    def blocks(self) -> List[nn.Module]:
        blocks = []
        for stage in self.stages:
            if isinstance(stage, nn.Sequential):
                blocks += list(stage)
            else:
                blocks += [stage]
        return blocks

    def zero_last_gamma(self) -> None:
        for block in self.blocks:
            if isinstance(block, ResidualBlock) and isinstance(block.shortcut, nn.Identity):
                if isinstance(block.conv, (MBV1Block, MBV2Block, FusedMBV2Block)):
                    nn.init.zeros_(block.conv.point_conv.norm.weight)
                elif isinstance(block.conv, BottleneckBlock):
                    nn.init.zeros_(block.conv.expand_conv.norm.weight)
                else:
                    raise NotImplementedError(type(block.conv))

    def forward(self, x):
        output_dict = {"input": x}
        output_dict["stage0"] = x = self.input_stem(x)
        for stage_id, stage in enumerate(self.stages, 1):
            output_dict["stage%d" % stage_id] = x = stage(x)
        output_dict["output"] = x
        return output_dict
