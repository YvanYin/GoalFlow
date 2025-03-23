from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from vovnet.backbones.utils import make_divisible
from vovnet.backbones.ops import (
	ConvLayer, PoolingLayer, SPPBlock, UpSampleLayer, MBV2Block, FusedMBV2Block, ResidualBlock, DAGOp
)

__all__ = ["FPN"]


def build_block(
	block_str: str, in_channels: int, out_channels: int, channel_att: Optional[str], act_func: str
) -> nn.Module:
	block_config = {"e": 4, "k": 3}
	block_str = block_str.split("_")
	block_config["name"] = block_str[0]
	for hparam in block_str[1:]:
		if hparam.startswith("e@"):
			block_config["e"] = float(hparam[2:])
		elif hparam.startswith("k@"):
			block_config["k"] = int(hparam[2:])
	
	mid_channels = make_divisible(in_channels * block_config["e"], 8)
	if channel_att is not None:
		raise NotImplementedError
	else:
		channel_att = None
	
	if block_config["name"] == "mb":
		block = MBV2Block(
			in_channels,
			out_channels,
			block_config["k"],
			expand_ratio=block_config["e"],
			act_func=(act_func, act_func, None),
		)
		if channel_att is not None:
			block = nn.Sequential(
				block.inverted_conv,
				block.depth_conv,
				channel_att,
				block.point_conv
			)
	elif block_config["name"] == "fmb":
		block = FusedMBV2Block(
			in_channels,
			out_channels,
			block_config["k"],
			expand_ratio=block_config["e"],
			act_func=(act_func, None),
		)
		if channel_att is not None:
			block = nn.Sequential(
				block.spatial_conv,
				channel_att,
				block.point_conv,
			)
	else:
		raise NotImplementedError
	
	if in_channels == out_channels:
		block = ResidualBlock(
			block,
			nn.Identity(),
		)
	return block


class FPN(nn.Module):
	"""Vanilla FPN and PAN"""
	
	def __init__(
		self,
		# inputs
		inputs: List[Tuple[str, int, int, int]],
		input_mode="cat_conv",
		# middle
		middle_config: Optional[Dict] = None,
		channel_att: Optional[str] = None,
		# general
		prefix="fpn",
		act_func="relu",
		spp_size: Optional[List] = None,
		use_pan=True,
		output_width: Optional[int] = None,
	):
		super(FPN, self).__init__()
		middle_config = middle_config or {}
		if "all" not in middle_config:
			middle_config["all"] = ["mbv2_e@4_k@5", "mbv2_e@4_k@5"]
		
		# sort inputs by stride
		inputs = sorted(inputs, key=lambda tup: tup[-1], reverse=True)
		self.inputs = inputs
		self.prefix = prefix
		self.output_width = output_width
		
		blocks = []
		extra_input = []
		for idx, (feature_id, in_channels, mid_channels, stride) in enumerate(inputs):
			# inputs
			dag_inputs, dag_merge_mode, dag_post_input_op = self.build_input(
				feature_id,
				in_channels,
				extra_input,
				input_mode,
				mid_channels,
				act_func,
			)
			# middle
			dag_middle_blocks = []
			if idx == 0 and spp_size is not None:
				spp_block = ResidualBlock(
					SPPBlock(
						mid_channels,
						pool_size=spp_size,
						pool_type="avg",
						act_func=act_func,
					),
					nn.Identity(),
				)
				dag_middle_blocks.append(spp_block)
			for block_str in middle_config.get(stride, middle_config["all"]):
				dag_middle_blocks.append(
					build_block(
						block_str,
						mid_channels,
						mid_channels,
						channel_att,
						act_func,
					)
				)
			# output
			if use_pan or self.output_width is None:
				output_module = nn.Identity()
			else:
				output_module = ConvLayer(
					mid_channels,
					self.output_width,
					1,
					act_func=act_func,
				)
			dag_outputs = {
				f"{prefix}_{'inner' if use_pan else 'out'}{idx + 1}": output_module
			}
			if idx < len(inputs) - 1:
				up_factor = stride // inputs[idx + 1][3]
				dag_outputs[f"{prefix}_up{idx + 1}"] = nn.Sequential(
					ConvLayer(
						mid_channels, inputs[idx + 1][2], 1, act_func=act_func
					),
					UpSampleLayer(
						factor=up_factor,
						mode="bilinear",
						align_corners=False,
					)
					if up_factor > 1
					else None,
				)
				extra_input = [(f"{prefix}_up{idx + 1}", inputs[idx + 1][2])]
			
			blocks.append(
				DAGOp(
					inputs=dag_inputs,
					merge_mode=dag_merge_mode,
					post_input_op=dag_post_input_op,
					middle=nn.Sequential(*dag_middle_blocks),
					outputs=dag_outputs,
				)
			)
		if use_pan:
			for idx in range(len(inputs) - 1, -1, -1):
				_, _, mid_channels, stride = inputs[idx]
				if idx < len(inputs) - 1:
					extra_input = [(f"{prefix}_down{idx + 1}", mid_channels)]
				else:
					extra_input = []
				dag_inputs, dag_merge_mode, dag_post_input_op = self.build_input(
					f"{prefix}_inner{idx + 1}",
					mid_channels,
					extra_input,
					input_mode,
					mid_channels,
					act_func,
				)
				# middle
				dag_middle_blocks = []
				for block_str in middle_config.get(stride, middle_config["all"]):
					dag_middle_blocks.append(
						build_block(
							block_str,
							mid_channels,
							mid_channels,
							channel_att,
							act_func,
						)
					)
				# output
				if self.output_width is None:
					output_module = nn.Identity()
				else:
					output_module = ConvLayer(
						mid_channels,
						self.output_width,
						1,
						act_func=act_func,
					)
				dag_outputs = {f"{prefix}_out{idx + 1}": output_module}
				if idx != 0:
					down_factor = inputs[idx - 1][3] // stride
					downsample = PoolingLayer(
						pool_type="avg",
						kernel_size=down_factor,
						stride=down_factor,
					)
					dag_outputs[f"{prefix}_down{idx}"] = nn.Sequential(
						downsample,
						ConvLayer(
							mid_channels, inputs[idx - 1][2], 1, act_func=act_func,
						),
					)
				blocks.append(
					DAGOp(
						inputs=dag_inputs,
						merge_mode=dag_merge_mode,
						post_input_op=dag_post_input_op,
						middle=nn.Sequential(*dag_middle_blocks),
						outputs=dag_outputs,
					)
				)
		
		self.blocks = nn.ModuleList(blocks)
	
	@staticmethod
	def build_input(
		feature_id: str,
		in_channels: int,
		extra_input: List[Tuple[str, int]],
		input_mode: str,
		mid_channels: int,
		act_func: str,
	) -> Tuple[Dict[str, nn.Module], str, Optional[nn.Module]]:
		if input_mode == "cat_conv":
			merge_mode = "cat"
			inputs = {feature_id: nn.Identity()}
			for extra_id, extra_in_channels in extra_input:
				inputs[extra_id] = nn.Identity()
			post_input_op = ConvLayer(
				in_channels=sum([in_channels] + [extra_c for _, extra_c in extra_input]),
				out_channels=mid_channels,
				kernel_size=1,
				act_func=act_func,
			)
		elif input_mode == "add":
			merge_mode = "add"
			inputs = {feature_id: nn.Identity()}
			for extra_id, extra_in_channels in extra_input:
				inputs[extra_id] = nn.Identity()
			post_input_op = None
		else:
			raise NotImplementedError
		return inputs, merge_mode, post_input_op
	
	def forward(self, feature_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
		for block in self.blocks:
			feature_dict = block(feature_dict)
		return feature_dict
