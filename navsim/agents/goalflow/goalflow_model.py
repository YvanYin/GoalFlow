import math
from typing import Dict
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from navsim.agents.goalflow.goalflow_config import GoalFlowConfig
from navsim.agents.goalflow.resnet_backbone import ResNetBackbone
# from navsim.agents.goalflow.vitl_backbone import ViTBackbone
from navsim.common.enums import StateSE2Index
from navsim.agents.goalflow.goalflow_features import BoundingBox2DIndex
from navsim.agents.goalflow.utils.conditional_unet1d import ConditionalUnet1D
from navsim.agents.goalflow.utils import pos2posemb2d
from navsim.agents.goalflow.utils import get_vcs2bev_img_mat
from navsim.agents.goalflow.scorer import Scorer
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.dataclasses import Trajectory
from hydra.utils import instantiate

X_NOISE=[0.0,3/60,5/60,9/60,11/60,15/60,19/60,21/60,22/60,25/60,27/60,30/60]

class GoalFlowModel(nn.Module):
    def __init__(self, config: GoalFlowConfig):

        super().__init__()

        self._query_splits = [
            1,
            config.num_bounding_boxes,
        ]

        self._config = config
        # self._backbone = ViTBackbone(config)
        self._backbone=goalflowBackbone(config)

        # usually, the BEV features are variable in size.
        self._bev_downscale = nn.Conv2d(512, config.tf_d_model, kernel_size=1)

        self._bev_semantic_head = nn.Sequential(
            nn.Conv2d(
                config.bev_features_channels,
                config.bev_features_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.bev_features_channels,
                config.num_bev_classes,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.Upsample(
                size=(config.lidar_resolution_height // 2, config.lidar_resolution_width),
                mode="bilinear",
                align_corners=False,
            ),
        )

        self._keyval_embedding = nn.Embedding(
            8**2 + 1, config.tf_d_model
        )  # 8x8 feature grid + trajectory
        self._query_embedding = nn.Embedding(sum(self._query_splits), config.tf_d_model)
        self._status_encoding = nn.Linear(4 + 2 + 2, config.tf_d_model)

        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )

        self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)
        self._agent_head = AgentHead(
            num_agents=config.num_bounding_boxes,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
        )

        self._trajectory_head = TrajectoryHead(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
        )

        self._navi_head = NaviHead(
            num_poses=1,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
        )

        # wd=64
        # self.bev_pos=nn.Embedding(wd,config.tf_d_model)

        # self.all_cond_decoder = nn.MultiheadAttention(
        #     embed_dim=config.tf_d_model,
        #     num_heads=8,
        #     batch_first=True
        # )

        # self.all_cond_layernorm_1 = nn.LayerNorm(config.tf_d_model)
        # self.all_cond_layernorm_2 = nn.LayerNorm(config.tf_d_model)
        # self.all_cond_mlp = nn.Sequential(
        #     nn.Linear(config.tf_d_model, 2 * config.tf_d_model),
        #     nn.GELU(),
        #     nn.Linear(2 * config.tf_d_model, config.tf_d_model),
        # )

        # self.ego_latent_embed = nn.Embedding(1, config.tf_d_model)

        # self.ego_all_cond_decoder = nn.MultiheadAttention(
        #     embed_dim=config.tf_d_model,
        #     num_heads=8,
        #     batch_first=True
        # )

        # self.ego_all_cond_mlp = nn.Sequential(
        #     nn.Linear(config.tf_d_model, 2 * config.tf_d_model),
        #     nn.GELU(),
        #     nn.Linear(2 * config.tf_d_model, config.tf_d_model),
        # )

        # self.ego_all_cond_layernorm_1 = nn.LayerNorm(config.tf_d_model)
        # self.ego_all_cond_layernorm_2 = nn.LayerNorm(config.tf_d_model)

        # self.vcs_range=(-51.2, -51.2, 108.8, 51.2)
        # self.vcs2bev_norm = torch.from_numpy(
        #     get_vcs2bev_img_mat(self.vcs_range, [1, 1])
        # ).float()

        # self.noise_pred_net=ConditionalUnet1D(
        #     input_dim=30,
        #     global_cond_dim=config.tf_d_model,
        #     # ===================== add zero point to start ========================
        #     # down_dims=[64,128],
        #     down_dims=[64,128,256], # defualt
        #     cond_predict_scale=False
        # )
        self.noise_pred_net2=ConditionalUnet1D(
            input_dim=30,
            global_cond_dim=config.tf_d_model,
            # ===================== add zero point to start ========================
            # down_dims=[64,128],
            down_dims=[64,128,256], # defualt
            cond_predict_scale=False
        )

        if self._config.freeze_layers1:
            self.freeze_layers(self._backbone)
            self.freeze_layers(self._bev_downscale)
            self.freeze_layers(self._bev_semantic_head)
            self.freeze_layers(self._keyval_embedding)
            self.freeze_layers(self._query_embedding)
            self.freeze_layers(self._status_encoding)
            self.freeze_layers(self._tf_decoder)
            self.freeze_layers(self._agent_head)
        if self._config.freeze_layers2:
            self.freeze_layers(self._navi_head)
            self.freeze_layers(self._trajectory_head)
        if self._config.freeze_layers3:
            self.freeze_layers(self.noise_pred_net2)


    def freeze_layers(self, layer):
        # 将传入的层的参数 requires_grad 设置为 False
        for param in layer.parameters():
            param.requires_grad = False

    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str,torch.Tensor]) -> Dict[str, torch.Tensor]:
        # token=features['token']
        camera_feature: torch.Tensor = features["camera_feature"]
        lidar_feature: torch.Tensor = features["lidar_feature"]
        status_feature: torch.Tensor = features["status_feature"]

        batch_size = status_feature.shape[0]

        bev_feature_upscale, bev_feature, _ = self._backbone(camera_feature, lidar_feature)
        # bev_feature_upscale(bz,64,64,64), bev_feature(bz,512,8,8)

        bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1)
        bev_feature = bev_feature.permute(0, 2, 1)
        # bev_feature(bz,64,256)
        gt_trajs=features['gt_trajs'].to(bev_feature)
        dtype=bev_feature.dtype
        device=bev_feature.device

        # =================================== ours method ==================================================
        # bev_feature_pos=self.bev_pos.weight[None].repeat(batch_size,1,1).to(bev_feature)
        # bev_feature_mask=torch.zeros((batch_size,bev_feature.shape[1]),device=bev_feature.device)

        # # print("gt_trajs.shape:",gt_trajs.shape)

        # # TODO add pv to the args
        # is_pv=False
        # if is_pv:
        #     pass
        # else:
        #     all_feature=bev_feature
        #     all_feature_pos=bev_feature_pos
        #     all_feature_mask=bev_feature_mask
        
        # # ================== ego query <---> bev/navi feature =====================================
        # # order_dict = (cross_attn, layernorm, ffn , layernorm)
        # all_cond_latent = self.all_cond_decoder(
        #     query = all_feature + all_feature_pos,
        #     key = all_feature + all_feature_pos,
        #     value = all_feature,
        #     key_padding_mask = all_feature_mask,
        # )[0]

        # all_cond_latent = self.all_cond_layernorm_1(all_cond_latent)
        # all_cond_latent = self.all_cond_mlp(all_cond_latent)
        # all_cond_latent = self.all_cond_layernorm_2(all_cond_latent)

        # ego_latent = self.ego_latent_embed.weight[None].repeat(batch_size, 1, 1)
        # ego_agent_pos = torch.zeros(
        #     (batch_size, 1, 2),device=ego_latent.device
        # )
        # ego_agent_ones = torch.ones (
        #     (batch_size, 1, 1),device=ego_latent.device
        # )
        # ego_agent_pos = (
        #     torch.cat([ego_agent_pos, ego_agent_ones],dim=2)
        #     @ self.vcs2bev_norm.to(ego_latent.device).T
        # )
        # ego_agent_pos = pos2posemb2d(ego_agent_pos,num_pos_feats=self._config.tf_d_model // 2)

        # # cross attention 
        # ego_latent = self.ego_all_cond_decoder(
        #     query = ego_latent + ego_agent_pos,
        #     # query = ego_latent,
        #     key = all_cond_latent,
        #     value = all_cond_latent,
        #     key_padding_mask = all_feature_mask,
        # )[0]
        # ego_latent = self.ego_all_cond_layernorm_1(ego_latent)
        # ego_latent = self.ego_all_cond_mlp(ego_latent)
        # ego_latent = self.ego_all_cond_layernorm_2(ego_latent)


        # =================================== goalflow ==================================================
        status_encoding = self._status_encoding(status_feature)
        # status_encoding (1,256)

        keyval = torch.concatenate([bev_feature, status_encoding[:, None]], dim=1)
        keyval += self._keyval_embedding.weight[None, ...]

        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        query_out = self._tf_decoder(query, keyval)

        trajectory_query, agents_query = query_out.split(self._query_splits, dim=1)
        # trajecotry_query(1,1,256), agents_query(1,30,256)
        # trajs
        goalflow_trajs = self._trajectory_head(trajectory_query)['trajectory']
        navi_point=self._navi_head(trajectory_query)['navi_point']
        if self._config.start:
            start_point=torch.zeros((goalflow_trajs.shape[0],1,3)).to(gt_trajs)
            dummy_goalflow_trajs=torch.cat([start_point,goalflow_trajs],dim=1)
        
        # test
        if torch.isnan(agents_query).any():
            print("agents_query",agents_query)
        agents = self._agent_head(agents_query)
        bev_semantic_map = self._bev_semantic_head(bev_feature_upscale)

        # =================================== flow ==================================================
        target=gt_trajs
        if self._config.has_navi:
            # navi=gt_trajs[:,7:8,:2].clone().to(device)
            navi=navi_point
            navi_feature=pos2posemb2d(navi,num_pos_feats=self._config.tf_d_model//2).squeeze(1)
            # global_feature=torch.cat([ego_latent.squeeze(1),navi_feature],dim=-1)
            global_feature=torch.cat([trajectory_query.squeeze(1),navi_feature],dim=-1)
        else:
            # global_feature=ego_latent.squeeze(1)
            global_feature=trajectory_query.squeeze(1)
        # print("global_feature.shape:",global_feature.shape)


        if self._config.start:
            start_point=torch.zeros((gt_trajs.shape[0],1,3)).to(gt_trajs)
            gt_trajs=torch.cat([start_point,gt_trajs],dim=1)
        target=torch.zeros_like(gt_trajs)
        normal_trajs = self.normalize_xy_rotation(gt_trajs, N=gt_trajs.shape[-2], times=10).to(gt_trajs)
        normal_trajs=normal_trajs.to(global_feature)
        if self._config.training:
            noise=torch.randn(normal_trajs.shape,device=normal_trajs.device,dtype=dtype).to(global_feature)*0.3
            # x_shift=torch.tensor(X_NOISE,dtype=torch.float32).to(global_feature).unsqueeze(0).repeat(batch_size,1)
            # noise=torch.randn(gt_trajs.shape,device=normal_trajs.device).to(global_feature)
            # noise=torch.cat([(noise[:,:,1]+x_shift).unsqueeze(-1),noise[:,:,1:]],dim=-1)*0.1
            # noise=self.xy_rotation(noise,N=gt_trajs.shape[-2],times=10).to(global_feature)
        else:
            noise=torch.randn(size=(batch_size*self._config.anchor_size,dummy_goalflow_trajs.shape[-2],30),dtype=dtype,device=device)*0.1
            # x_shift=torch.tensor(X_NOISE,dtype=torch.float32).to(global_feature).unsqueeze(0).repeat(batch_size*self._config.anchor_size,1)
            # noise=torch.randn(size=(batch_size*self._config.anchor_size,dummy_goalflow_trajs.shape[-2],3),device=normal_trajs.device).to(global_feature)
            # noise=torch.cat([(noise[:,:,1]+x_shift).unsqueeze(-1),noise[:,:,1:]],dim=-1)*0.1
            # noise=self.xy_rotation(noise,N=gt_trajs.shape[-2],times=10).to(global_feature)
        
        if self._config.training:

            batch_size = normal_trajs.shape[0]

            if self._config.start:
                noise[:,[0],:]=normal_trajs[:,[0],:]
            if self._config.end:
                noise[:,[-1],:]=normal_trajs[:,[-1],:]

            noisy_traj_points,t,target=get_train_tuple(z0=noise,z1=normal_trajs)

            timesteps=t.squeeze()*self._config.infer_steps
            
            # ===================== choose cfg ========================
            pred_cond = self.noise_pred_net2(
                sample=noisy_traj_points,
                timestep=timesteps,
                global_cond=global_feature,
                use_dropout=False
                )
            pred_uncond = self.noise_pred_net2(
                sample=noisy_traj_points,
                timestep=timesteps,
                global_cond=global_feature,
                force_dropout=True
                )
            pred=pred_uncond+self._config.cond_weight*(pred_cond-pred_uncond)
        
        else:

            normal_dummy_goalflow_trajs = self.normalize_xy_rotation(dummy_goalflow_trajs, N=dummy_goalflow_trajs.shape[-2], times=10).to(dummy_goalflow_trajs).mul_(0.1)
            if self._config.warm_up:
                noise=torch.randn(size=(batch_size*self._config.anchor_size,dummy_goalflow_trajs.shape[-2],30),dtype=dtype,device=device)
                rate=self._config.renoise_steps/self._config.infer_steps
                trajs=normal_dummy_goalflow_trajs*(1.-rate)+noise*rate
            else:
                # x_shift=torch.tensor(X_NOISE,dtype=torch.float32).to(global_feature)
                # noise=torch.randn(normal_trajs.shape,device=normal_trajs.device).to(global_feature)
                # noise=noise[:,0,:]+x_shift
                # noise=noise*0.1
                trajs=noise
            if self._config.start:
                trajs[:,[0],:]=normal_trajs[:,[0],:]
            if self._config.end:
                trajs[:,[-1],:]=normal_trajs[:,[-1],:]

            repeated_tensor=global_feature.unsqueeze(1).repeat(1,self._config.anchor_size,1)
            expanded_tensor=repeated_tensor.view(-1,self._config.tf_d_model)
            # ===================== choose different infer steps ========================
            # timesteps=torch.arange(0,self.infer_steps).to(device)
            if not self._config.warm_up:
                timesteps=torch.arange(0,self._config.infer_steps,1).to(device)
            else:
                timesteps=torch.arange(self._config.infer_steps-self._config.renoise_steps,self._config.infer_steps,1).to(device)

            for t in timesteps:
                net_output_cond = self.noise_pred_net2(
                    sample=trajs,
                    timestep=t,
                    global_cond=expanded_tensor,
                    use_dropout=False,
                    )
                net_output_uncond = self.noise_pred_net2(
                    sample=trajs,
                    timestep=t,
                    global_cond=expanded_tensor,
                    force_dropout=True,
                    )
                net_output=net_output_uncond+self._config.cond_weight*(net_output_cond-net_output_uncond)
                trajs=trajs.detach().clone()+net_output*(1 / self._config.infer_steps)
            
            diffusion_output = self.denormalize_xy_rotation(trajs, N=gt_trajs.shape[-2], times=10)

            pred_trajs=diffusion_output.reshape(batch_size,self._config.anchor_size,-1,3)

            # add scorer
            has_scorer=False
            if has_scorer:
                traj_sampling=TrajectorySampling(num_poses=40,interval_length=0.1)
                scorer=Scorer(proposal_sampling=traj_sampling)
                trajs_withmean=torch.cat([pred_trajs[:,:,:1+8,:],pred_trajs[:,:,:1+8,:].mean(1,keepdim=True)],dim=1)
                max_indices=[]
                for i in range(batch_size):
                    scores=scorer.score_proposals(trajs_withmean.squeeze(0).numpy(),bev_semantic_map.squeeze(0).numpy(),agents['agent_states'].squeeze(0).numpy(),targets['bev_semantic_map'].squeeze(0).numpy())
                    if scores[-1]==1.:
                        max_index=len(scores)-1
                    else:
                        max_index=np.argmax(scores)
                    max_indices.append(max_index)
                batch_indices=torch.arange(batch_size,dtype=torch.int32).to(device)
                max_indices=torch.tensor(max_indices,dtype=torch.int32).to(device)
                pred=trajs_withmean[batch_indices,max_indices,1:1+8,:]

                

            # =================================== draw anchors ==================================================
            if self._config.is_save_png:
                import os
                import random
                import string
                save_dir=self._config.save_dir
                output_dir = os.path.dirname(save_dir)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                rand_token = ''.join(random.choices(string.ascii_lowercase + string.digits, k=16))
                # save_path=save_dir+f'{token}_{rand_token}'+'.png'
                save_path=save_dir+f'{rand_token}'+'.png'
                # draw_trajs=pred_trajs[0,:,:,:].mean(0,keepdim=True)
                draw_trajs=pred_trajs[0,:,:,:]
                if has_scorer:
                    new_tmp=torch.cat([pred_trajs,pred_trajs.mean(1,keepdim=True)],dim=1)
                    chosen_trajectory=new_tmp[batch_indices,max_indices]
                draw_anchors(draw_trajs,gt_trajs,save_path,self._config.anchor_size)
                # draw_anchors_with_mlp(draw_trajs,chosen_trajectory,gt_trajs,save_path,self._config.anchor_size)
                # draw_trajs_mlp=dummy_goalflow_trajs[0,:,:].unsqueeze(0)
                # draw_anchors_with_mlp(draw_trajs,navi_point,gt_trajs,save_path,self._config.anchor_size)
            
            if not has_scorer:
                if self._config.start:
                    pred=pred_trajs[:,:,1:1+8,:].mean(1)
                else:
                    pred=pred_trajs[:,:,:8,:].mean(1)

        # goalflow query
        # pred = self._trajectory_head(trajectory_query)

        # output.update(trajectory)
        # [bz,num_classes(7),128,256]
        output: Dict[str, torch.Tensor] = {"bev_semantic_map": bev_semantic_map}
        output.update({'trajectory':pred})
        output.update({'target':target})
        output.update({'mlp_trajectory':goalflow_trajs})
        output.update({'navi_point':navi_point})
        # [bz,num,5]
        # output.update({"trajectory":gt_trajs})
        output.update(agents)

        return output
    
    def xy_rotation(self, trajectory, N=8, times=10):
        batch, num_pts, dim = trajectory.shape
        downsample_trajectory = trajectory[:, :N, :].detach().clone()

        rotated_trajectories = []
        for i in range(times):
            theta = 2 * math.pi * i / times  # 将角度均匀分布在0到2π之间
            rotation_matrix, _ = get_rotation_matrices(theta)
            # 扩展旋转矩阵以匹配批次大小
            rotation_matrix = rotation_matrix.unsqueeze(0).expand(downsample_trajectory.size(0), -1, -1).to(downsample_trajectory)
            
            rotated_trajectory = apply_rotation(downsample_trajectory[:,:,:2], rotation_matrix)
            rotated_trajectory=torch.cat([rotated_trajectory,downsample_trajectory[:,:,-1:].permute(0,2,1)],dim=1)
            rotated_trajectories.append(rotated_trajectory)
        resulting_trajectory = torch.cat(rotated_trajectories, 1)
        trajectory = resulting_trajectory.permute(0,2,1)
        return trajectory

    def normalize_xy_rotation(self, trajectory, N=8, times=10):
        # print(trajectory[0])
        # print(trajectory[0].max())
        # print(trajectory[0].min())
        batch, num_pts, dim = trajectory.shape
        downsample_trajectory = trajectory[:, :N, :].detach().clone()
        x_scale = 60
        y_scale = 15
        # x_scale = 40
        # y_scale = 5
        heading_scale = math.pi
        downsample_trajectory[:, :, 0] /= x_scale
        downsample_trajectory[:, :, 1] /= y_scale
        downsample_trajectory[:,:,2]/=heading_scale
        downsample_trajectory[:,:,2]=downsample_trajectory[:,:,2].atanh()
        
        rotated_trajectories = []
        for i in range(times):
            theta = 2 * math.pi * i / times  # 将角度均匀分布在0到2π之间
            rotation_matrix, _ = get_rotation_matrices(theta)
            # 扩展旋转矩阵以匹配批次大小
            rotation_matrix = rotation_matrix.unsqueeze(0).expand(downsample_trajectory.size(0), -1, -1).to(downsample_trajectory)
            
            rotated_trajectory = apply_rotation(downsample_trajectory[:,:,:2], rotation_matrix)
            rotated_trajectory=torch.cat([rotated_trajectory,downsample_trajectory[:,:,-1:].permute(0,2,1)],dim=1)
            rotated_trajectories.append(rotated_trajectory)
        resulting_trajectory = torch.cat(rotated_trajectories, 1)
        trajectory = resulting_trajectory.permute(0,2,1)
        return trajectory
    
    def denormalize_xy_rotation(self, trajectory, N=8, times=10):
        batch, num_pts, dim = trajectory.shape
        inverse_rotated_trajectories = []
        for i in range(times):
            theta = 2 * math.pi * i / 10  # 将角度均匀分布在0到2π之间
            rotation_matrix, inverse_rotation_matrix = get_rotation_matrices(theta)
            # 扩展旋转矩阵以匹配批次大小
            inverse_rotation_matrix = inverse_rotation_matrix.unsqueeze(0).expand(trajectory.size(0), -1, -1).to(trajectory)
        
            # 只对每个 2D 坐标对进行逆旋转
            inverse_rotated_trajectory = apply_rotation(trajectory[:, :, 3*i:3*i+2], inverse_rotation_matrix)
            inverse_rotated_trajectory=torch.cat([inverse_rotated_trajectory,trajectory[:,:,3*i+2:3*i+3].permute(0,2,1)],dim=1)
            inverse_rotated_trajectories.append(inverse_rotated_trajectory)

        final_trajectory = torch.cat(inverse_rotated_trajectories, 1).permute(0,2,1)
        
        final_trajectory = final_trajectory[:, :, :3]
        final_trajectory[:, :, 0] *= 60
        final_trajectory[:, :, 1] *= 15
        final_trajectory[:,:,2]=final_trajectory[:,:,2].tanh() * math.pi
        return final_trajectory
    
    def normalize_xy_rotation2(self, trajectory, N=8, times=10):
        # print(trajectory[0])
        # print(trajectory[0].max())
        # print(trajectory[0].min())
        batch, num_pts, dim = trajectory.shape
        downsample_trajectory = trajectory[:, :N, :].detach().clone()
        x_scale = 45.
        y_scale = 42.
        # x_scale = 40
        # y_scale = 5
        heading_scale = math.pi
        downsample_trajectory[:,:,0] -= x_scale
        downsample_trajectory[:, :, 0] /= x_scale
        downsample_trajectory[:, :, 1] /= y_scale
        downsample_trajectory[:,:,2]/=heading_scale
        # downsample_trajectory[:,:,2]=downsample_trajectory[:,:,2].atanh()
        
        rotated_trajectories = []
        for i in range(times):
            theta = 2 * math.pi * i / times  # 将角度均匀分布在0到2π之间
            rotation_matrix, _ = get_rotation_matrices(theta)
            # 扩展旋转矩阵以匹配批次大小
            rotation_matrix = rotation_matrix.unsqueeze(0).expand(downsample_trajectory.size(0), -1, -1).to(downsample_trajectory)
            
            rotated_trajectory = apply_rotation(downsample_trajectory[:,:,:2], rotation_matrix)
            rotated_trajectory=torch.cat([rotated_trajectory,downsample_trajectory[:,:,-1:].permute(0,2,1)],dim=1)
            rotated_trajectories.append(rotated_trajectory)
        resulting_trajectory = torch.cat(rotated_trajectories, 1)
        trajectory = resulting_trajectory.permute(0,2,1)
        return trajectory
    
    def denormalize_xy_rotation2(self, trajectory, N=8, times=10):
        batch, num_pts, dim = trajectory.shape
        inverse_rotated_trajectories = []
        for i in range(times):
            theta = 2 * math.pi * i / 10  # 将角度均匀分布在0到2π之间
            rotation_matrix, inverse_rotation_matrix = get_rotation_matrices(theta)
            # 扩展旋转矩阵以匹配批次大小
            inverse_rotation_matrix = inverse_rotation_matrix.unsqueeze(0).expand(trajectory.size(0), -1, -1).to(trajectory)
        
            # 只对每个 2D 坐标对进行逆旋转
            inverse_rotated_trajectory = apply_rotation(trajectory[:, :, 3*i:3*i+2], inverse_rotation_matrix)
            inverse_rotated_trajectory=torch.cat([inverse_rotated_trajectory,trajectory[:,:,3*i+2:3*i+3].permute(0,2,1)],dim=1)
            inverse_rotated_trajectories.append(inverse_rotated_trajectory)

        final_trajectory = torch.cat(inverse_rotated_trajectories, 1).permute(0,2,1)
        
        final_trajectory = final_trajectory[:, :, :3]
        final_trajectory[:, :, 0] *= 45.
        final_trajectory[:,:,0]+=45.
        final_trajectory[:, :, 1] *= 42.
        final_trajectory[:,:,2]=final_trajectory[:,:,2]*math.pi
        return final_trajectory


class AgentHead(nn.Module):
    def __init__(
        self,
        num_agents: int,
        d_ffn: int,
        d_model: int,
    ):
        super(AgentHead, self).__init__()

        self._num_objects = num_agents
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp_states = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, BoundingBox2DIndex.size()),
        )

        self._mlp_label = nn.Sequential(
            nn.Linear(self._d_model, 1),
        )

    def forward(self, agent_queries) -> Dict[str, torch.Tensor]:

        agent_states = self._mlp_states(agent_queries)
        if torch.isnan(agent_states).any():
            print("agent_states1",agent_states)
        agent_states[..., BoundingBox2DIndex.POINT] = (
            agent_states[..., BoundingBox2DIndex.POINT].tanh() * 32
        )
        agent_states[..., BoundingBox2DIndex.HEADING] = (
            agent_states[..., BoundingBox2DIndex.HEADING].tanh() * np.pi
        )
        if torch.isnan(agent_states).any():
            print("agent_states2",agent_states)
        agent_labels = self._mlp_label(agent_queries).squeeze(dim=-1)

        return {"agent_states": agent_states, "agent_labels": agent_labels}


class TrajectoryHead(nn.Module):
    def __init__(self, num_poses: int, d_ffn: int, d_model: int):
        super(TrajectoryHead, self).__init__()

        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, num_poses * StateSE2Index.size()),
        )

    def forward(self, object_queries) -> Dict[str, torch.Tensor]:
        poses = self._mlp(object_queries).reshape(-1, self._num_poses, StateSE2Index.size())
        poses[..., StateSE2Index.HEADING] = poses[..., StateSE2Index.HEADING].tanh() * np.pi
        return {"trajectory": poses}
    
class NaviHead(nn.Module):
    def __init__(self, num_poses: int, d_ffn: int, d_model: int):
        super(NaviHead, self).__init__()

        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, 2),
        )

    def forward(self, object_queries) -> Dict[str, torch.Tensor]:
        poses = self._mlp(object_queries).reshape(-1, self._num_poses, 2)
        return {"navi_point": poses}

def get_rotation_matrices(theta):
    """
    给定角度 theta, 返回旋转矩阵和逆旋转矩阵

    参数:
    theta (float): 旋转角度（以弧度表示）

    返回:
    rotation_matrix (torch.Tensor): 旋转矩阵
    inverse_rotation_matrix (torch.Tensor): 逆旋转矩阵
    """
    # 将角度转换为张量
    theta_tensor = torch.tensor(theta)
    
    # 计算旋转矩阵和逆旋转矩阵
    cos_theta = torch.cos(theta_tensor)
    sin_theta = torch.sin(theta_tensor)

    rotation_matrix = torch.tensor([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])

    inverse_rotation_matrix = torch.tensor([
        [cos_theta, sin_theta],
        [-sin_theta, cos_theta]
    ])
    
    return rotation_matrix, inverse_rotation_matrix

def apply_rotation(trajectory, rotation_matrix):
    # 将 (x, y) 坐标与旋转矩阵相乘
    rotated_trajectory = torch.einsum('bij,bkj->bik', rotation_matrix, trajectory)
    return rotated_trajectory

def get_train_tuple(z0=None, z1=None):
    t = torch.rand(z1.shape[0], 1, 1).to(z0.device)
    # t=torch.sigmoid(torch.randn(z0.shape[0],1,1)).to(z0.device)
    z_t =  t * z1 + (1.-t) * z0
    target = z1 - z0
    return z_t.float(), t.float(), target.float()

def draw_anchors_with_mlp(input_trajectories,mlp_trajs,gt_trajs,save_path,anchor_size=128):
    trajectories_to_plot = input_trajectories[:anchor_size]
    gt_traj=gt_trajs[0]
    plt.figure(figsize=(10, 8))
    # for i, trajectory in enumerate(trajectories_to_plot):
    #     x_coords = trajectory[:, 0].cpu().numpy()
    #     y_coords = trajectory[:, 1].cpu().numpy()
    #     plt.plot(x_coords, y_coords)
    x_coords=gt_traj[:9,0].cpu().numpy()
    y_coords=gt_traj[:9,1].cpu().numpy()
    plt.plot(x_coords, y_coords, color='red', linewidth=3, label='gt Trajectory')
    x_coords=input_trajectories.mean(0)[:9,0].cpu().numpy()
    y_coords=input_trajectories.mean(0)[:9,1].cpu().numpy()
    plt.plot(x_coords, y_coords, color='blue', linewidth=3, label='mean Trajectory')
    x_coords=mlp_trajs[0,:9,0].cpu().numpy()
    y_coords=mlp_trajs[0,:9,1].cpu().numpy()
    # plt.plot(x_coords, y_coords, color='green', linewidth=3, label='mlp Trajectory')
    plt.plot(x_coords, y_coords, color='green', linewidth=3, label='chosen Trajectory')  # s 参数控制点的大小
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title(f'{anchor_size} Trajectories')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)

def draw_anchors(input_trajectories,gt_trajs,save_path,anchor_size=128):
    trajectories_to_plot = input_trajectories[:anchor_size]
    gt_traj=gt_trajs[0]
    plt.figure(figsize=(10, 8))
    for i, trajectory in enumerate(trajectories_to_plot):
        x_coords = trajectory[:, 0].cpu().numpy()
        y_coords = trajectory[:, 1].cpu().numpy()
        plt.plot(x_coords, y_coords)
    x_coords=gt_traj[:,0].cpu().numpy()
    y_coords=gt_traj[:,1].cpu().numpy()
    plt.plot(x_coords, y_coords, color='red', linewidth=3, label='gt Trajectory')
    x_coords=input_trajectories.mean(0)[:,0].cpu().numpy()
    y_coords=input_trajectories.mean(0)[:,1].cpu().numpy()
    plt.plot(x_coords, y_coords, color='blue', linewidth=3, label='mean Trajectory')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title(f'{anchor_size} Trajectories')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)