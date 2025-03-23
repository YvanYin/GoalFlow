import math
from typing import Dict
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from navsim.agents.goalflow.goalflow_config import GoalFlowConfig
from navsim.agents.goalflow.resnet_backbone import GoalFlowBackbone
from navsim.common.enums import StateSE2Index
from navsim.agents.goalflow.goalflow_features import BoundingBox2DIndex
from navsim.agents.goalflow.utils import pos2posemb2d
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.dataclasses import Trajectory
from navsim.agents.goalflow.multihead_custom_attention import MultiheadCustomAttention
from navsim.agents.goalflow.diffusion_es import SinusoidalPosEmb
from navsim.agents.goalflow.diffusion_es import ParallelAttentionLayer
from navsim.agents.goalflow.diffusion_es import RotaryPositionEncoding
from navsim.agents.goalflow.v99_backbone import V299Backbone
from scipy.ndimage import distance_transform_edt
from kornia.contrib import distance_transform

class GoalFlowModel(nn.Module):
    def __init__(self, config: GoalFlowConfig):

        super().__init__()

        self._query_splits = [
            1,
            config.num_bounding_boxes,
        ]

        self._config = config
        # self._backbone = GoalFlowBackbone(config)
        self._backbone=V299Backbone(config)
        # self.cluster_points=np.load('/home/users/zebin.xing/workspace/navsim/mid_data/cluster_mid_256.npy')
        self.cluster_points=np.load('/horizon-bucket/saturn_v_dev/01_users/zebin.xing/mid_data/cluster_mid_8192_.npy')

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
        if self._config.has_drivable_map:
            self._drivable_area_head = nn.Sequential(
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
                    config.num_drivable_classes,
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
        self._status_encoding=nn.Linear(4+2+2,config.tf_d_model)
        if self._config.command_condition:
            self._command_encoding=nn.Linear(4,config.tf_d_model)
        # self._status_encoding=nn.Linear((4+3+2+2)*4,config.tf_d_model)

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

        # diffusion es
        self.sigma_encoder = nn.Sequential(
            SinusoidalPosEmb(self._config.tf_d_model),
            nn.Linear(self._config.tf_d_model, self._config.tf_d_model),
            nn.ReLU(),
            nn.Linear(self._config.tf_d_model, self._config.tf_d_model)
        )
        self.sigma_proj_layer = nn.Linear(self._config.tf_d_model * 2, self._config.tf_d_model)

        self.trajectory_encoder = nn.Linear(30, self._config.tf_d_model)
        self.trajectory_time_embeddings = RotaryPositionEncoding(self._config.tf_d_model)
        self.type_embedding = nn.Embedding(30, self._config.tf_d_model) # trajectory, noise token

        self.global_attention_layers = torch.nn.ModuleList([
            ParallelAttentionLayer(
                d_model=self._config.tf_d_model, 
                self_attention1=True, self_attention2=False,
                cross_attention1=False, cross_attention2=False,
                rotary_pe=True
            )
        for _ in range(8)])

        self.decoder_mlp = nn.Sequential(
            nn.Linear(self._config.tf_d_model, self._config.tf_d_model),
            nn.ReLU(),
            nn.Linear(self._config.tf_d_model, 30)
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


    def freeze_layers(self, layer):
        for param in layer.parameters():
            param.requires_grad = False

    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        token=features['token']
        camera_feature: torch.Tensor = features["camera_feature"]
        lidar_feature: torch.Tensor = features["lidar_feature"]
        status_feature: torch.Tensor = features["status_feature"]

        batch_size = status_feature.shape[0]

        bev_feature_upscale, bev_feature, _ = self._backbone(camera_feature, lidar_feature)
        # bev_feature_upscale(bz,64,64,64), bev_feature(bz,512,8,8)

        bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1)
        bev_feature = bev_feature.permute(0, 2, 1)
        # bev_feature(bz,64,256)
        gt_trajs=features['gt_trajs'].to(status_feature)
        dtype=status_feature.dtype
        device=status_feature.device

        # =================================== goalflow ==================================================
        if self._config.has_history:
            used_dims=[0,1,2,3,7,8,9,10]
            status_encoding = self._status_encoding(status_feature[:,-1,used_dims])
        else:
            status_encoding=self._status_encoding(status_feature)
        # status_encoding (1,256)

        keyval = torch.concatenate([bev_feature, status_encoding[:, None]], dim=1)
        keyval += self._keyval_embedding.weight[None, ...]

        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        query_out = self._tf_decoder(query, keyval)

        trajectory_query, agents_query = query_out.split(self._query_splits, dim=1)
        # trajecotry_query(1,1,256), agents_query(1,30,256)
        goalflow_trajs = self._trajectory_head(trajectory_query)['trajectory']
        # navi_point=self._navi_head(trajectory_query)['navi_point']
        if self._config.start:
            start_point=torch.zeros((goalflow_trajs.shape[0],1,3)).to(gt_trajs)
            dummy_goalflow_trajs=torch.cat([start_point,goalflow_trajs],dim=1)
        agents = self._agent_head(agents_query)
        bev_semantic_map = self._bev_semantic_head(bev_feature_upscale)
        # # =================================== flow ==================================================
        target=gt_trajs

        if self._config.has_navi:
            navi=gt_trajs[:,7:8,:2].clone().to(device)
            # navi=navi_point
            navi_feature=pos2posemb2d(navi,num_pos_feats=self._config.tf_d_model//2).squeeze(1)
            global_feature=trajectory_query.squeeze(1)
            # global_feature=torch.cat([ego_latent.squeeze(1),navi_feature],dim=-1)
            # global_feature=torch.cat([trajectory_query.squeeze(1),navi_feature],dim=-1)
        else:
            # global_feature=ego_latent.squeeze(1)
            global_feature=trajectory_query.squeeze(1)
            # # print("global_feature.shape:",global_feature.shape)

            # if self._config.has_history:
            #     history_traj=status_feature[:,:,4:6].clone().to(device)
            #     history_feature=pos2posemb2d(history_traj,num_pos_feats=self._config.tf_d_model//2)
            #     history_feature=self._history_downscale(history_feature.reshape(batch_size,-1))
            #     global_feature=torch.cat([trajectory_query.squeeze(1),history_feature],dim=-1)
            #     # global_feature=trajectory_query.squeeze(1)+history_feature

            # global_feature=self._status_encoding(status_feature.reshape(batch_size,-1)).squeeze(-2)

        if self._config.start:
            start_point=torch.zeros((gt_trajs.shape[0],1,3)).to(gt_trajs)
            gt_trajs=torch.cat([start_point,gt_trajs],dim=1)
        target=torch.zeros_like(gt_trajs)
        if self._config.use_normalize3:
            normal_trajs = self.normalize_xy_rotation3(gt_trajs, N=gt_trajs.shape[-2], times=10).to(gt_trajs)
        else:
            normal_trajs = self.normalize_xy_rotation(gt_trajs, N=gt_trajs.shape[-2], times=10).to(gt_trajs)
        normal_trajs=normal_trajs.to(global_feature)[...,:self._config.trajectory_generation_len,:]
        if self._config.training:
            noise=torch.randn(size=(batch_size,self._config.trajectory_generation_len,30),device=normal_trajs.device,dtype=dtype).to(global_feature)*self._config.train_scale
            # x_shift=torch.tensor(X_NOISE,dtype=torch.float32).to(global_feature).unsqueeze(0).repeat(batch_size,1)
            # noise=torch.randn(gt_trajs.shape,device=normal_trajs.device).to(global_feature)
            # noise=torch.cat([(noise[t:,:,1]+x_shift).unsqueeze(-1),noise[:,:,1:]],dim=-1)*0.1
            # noise=self.xy_rotation(noise,N=gt_trajs.shape[-2],times=10).to(global_feature)
        else:
            noise=torch.randn(size=(batch_size*self._config.anchor_size,self._config.trajectory_generation_len,30),dtype=dtype,device=device)*self._config.test_scale
            # x_shift=torch.tensor(X_NOISE,dtype=torch.float32).to(ture).unsqueeze(0).repeat(batch_size*self._config.anchor_size,1)
            # noise=torch.randn(size=(batch_size*self._config.anchor_size,dummy_goalflow_trajs.shape[-2],3),device=normal_trajs.device).to(global_feature)
            # noise=torch.cat([(noise[:,:,1]+x_shift).unsqueeze(-1),noise[:,:,1:]],dim=-1)*0.1
            # noise=self.xy_rotation(noise,N=gt_trajs.shape[-2],times=10).to(global_feature)
        
        if self._config.use_mid_point:
            global_feature1=self.encode_scene_features(global_feature.unsqueeze(1))
            # global_feature2=self.encode_status_features(status_feature)
            global_feature2=self.encode_navi_features(navi_feature.unsqueeze(1))
            global_feature3=self.encode_mid_features(mid_feature.unsqueeze(1))
            global_feature=(torch.cat([global_feature1[0],global_feature2[0],global_feature3[0]],dim=-2),torch.cat([global_feature1[1],global_feature2[1],global_feature3[1]],dim=-2))

        elif self._config.use_heading_point:
            global_feature1=self.encode_scene_features(global_feature.unsqueeze(1))
            # global_feature2=self.encode_status_features(status_feature)
            global_feature2=self.encode_navi_features(navi_feature.unsqueeze(1))
            global_feature3=self.encode_mid_features(heading_feature.unsqueeze(1))
            global_feature=(torch.cat([global_feature1[0],global_feature2[0],global_feature3[0]],dim=-2),torch.cat([global_feature1[1],global_feature2[1],global_feature3[1]],dim=-2))

        elif self._config.has_navi or self._config.has_student_navi:
            global_feature1=self.encode_scene_features(global_feature.unsqueeze(1))
            # global_feature2=self.encode_status_features(status_feature)
            global_feature2=self.encode_navi_features(navi_feature.unsqueeze(1))
            global_feature=(torch.cat([global_feature1[0],global_feature2[0]],dim=-2),torch.cat([global_feature1[1],global_feature2[1]],dim=-2))
        elif self._config.command_condition:
            global_feature1=self.encode_scene_features(global_feature.unsqueeze(1))
            global_feature2=self.encode_command_features(self._command_encoding(status_feature[:,:4]).unsqueeze(1))
            global_feature=(torch.cat([global_feature1[0],global_feature2[0]],dim=-2),torch.cat([global_feature1[1],global_feature2[1]],dim=-2))
        else:
            global_feature=self.encode_scene_features(global_feature.unsqueeze(1))
        if self._config.training:

            batch_size = normal_trajs.shape[0]

            if self._config.start:
                noise[:,[0],:]=normal_trajs[:,[0],:]
            if self._config.end:
                # noise[:,[-1],:]=normal_trajs[:,[-1],:]
                noise[:,[8],:]=normal_trajs[:,[8],:]

            if not self._config.trig_flow:
                noisy_traj_points,t,target=get_train_tuple(z0=noise,z1=normal_trajs)
            else:
                noisy_traj_points,t,target=get_train_tuple_trig(z0=noise,z1=normal_trajs)

            # timesteps=t.squeeze()*self._config.infer_steps
            timesteps=t*self._config.infer_steps
            
            # ===================== choose cfg ========================
            # pred_cond = self.noise_pred_net2(
            #     sample=noisy_traj_points,
            #     timestep=timesteps,
            #     global_cond=global_feature,
            #     use_dropout=False
            #     )
            # pred_uncond = self.noise_pred_net2(
            #     sample=noisy_traj_points,
            #     timestep=timesteps,
            #     global_cond=global_feature,
            #     force_dropout=True
            #     )
            # pred=pred_uncond+self._config.cond_weight*(pred_cond-pred_uncond)

            # pred_cond = self.noise_pred_net2(
            #     sample=noisy_traj_points,
            #     timestep=timesteps,
            #     global_cond=global_feature,
            #     use_dropout=False
            #     )
            import random
            if self._config.only_cond or not self._config.has_navi:
                flag=random.randint(1,2)
                if flag==1:
                    pred=self.denoise(noisy_traj_points,timesteps,global_feature).reshape(batch_size,-1,30)
                elif flag==2:
                    pred=self.denoise(noisy_traj_points,timesteps,global_feature,force_dropout=True).reshape(batch_size,-1,30)
            elif not self._config.drop_scene:
                if self._config.use_mid_point or self._config.use_heading_point:
                    flag=random.randint(1,4)
                    if flag==1:
                        pred=self.denoise(noisy_traj_points,timesteps,global_feature).reshape(batch_size,-1,30)
                    elif flag==2:
                        pred=self.denoise(noisy_traj_points,timesteps,global_feature,force_dropout=True).reshape(batch_size,-1,30)
                    elif flag==3:
                        pred=self.denoise(noisy_traj_points,timesteps,global_feature,navi_dropout=True).reshape(batch_size,-1,30)
                    elif flag==4:
                        pred=self.denoise(noisy_traj_points,timesteps,global_feature,mid_dropout=True).reshape(batch_size,-1,30)
                elif self._config.has_navi or self._config.command_condition:
                    flag=random.randint(1,3)
                    # flag=np.random.choice([1,2,3],p=[2/5,1/5,2/5])
                    if flag==1:
                        pred=self.denoise(noisy_traj_points,timesteps,global_feature).reshape(batch_size,-1,30)
                    elif flag==2:
                        pred=self.denoise(noisy_traj_points,timesteps,global_feature,force_dropout=True).reshape(batch_size,-1,30)
                    elif flag==3:
                        pred=self.denoise(noisy_traj_points,timesteps,global_feature,navi_dropout=True).reshape(batch_size,-1,30)
            else:
                flag=random.randint(1,4)
                if flag==1:
                    pred=self.denoise(noisy_traj_points,timesteps,global_feature).reshape(batch_size,-1,30)
                elif flag==2:
                    pred=self.denoise(noisy_traj_points,timesteps,global_feature,force_dropout=True).reshape(batch_size,-1,30)
                elif flag==3:
                    pred=self.denoise(noisy_traj_points,timesteps,global_feature,navi_dropout=True).reshape(batch_size,-1,30)
                elif flag==4:
                    pred=self.denoise(noisy_traj_points,timesteps,global_feature,scene_dropout=True).reshape(batch_size,-1,30)

            # pred_cond=self.denoise(noisy_traj_points,timesteps,global_feature)
            # pred_uncond=self.denoise(noisy_traj_points,timesteps,global_feature,force_dropout=True)
            # pred_nonavi=self.denoise(noisy_traj_points,timesteps,global_feature,navi_dropout=True)
            # pred=(0.5*pred_nonavi+0.1*pred_uncond+0.4*pred_cond).reshape(batch_size,-1,30)

            # old version dac loss
            # inv_traj=pred+noise
            # inv_traj=self.denormalize_xy_rotation(inv_traj,inv_traj.shape[-2],times=10)[:,:9,:]
            # sdf=bev_semantic_map.argmax(1,keepdim=True).float()
            # sdf[sdf==2]=0
            # sdf[sdf!=0]=1
            # # distance=distance_transform_edt(sdf)
            # # inv_distance=distance_transform_edt(1-sdf)
            # distance=distance_transform(1-sdf,h=0.1)
            # inv_distance=distance_transform(sdf,h=0.1)
            # drivable_sdf=(distance-inv_distance)*self._config.bev_pixel_size
            # # dac_loss=self.esdf_manager(inv_traj,torch.from_numpy(drivable_sdf).to(inv_traj))
            # dac_loss=self.esdf_manager(inv_traj,drivable_sdf.squeeze()) if not flag==2 else 0
            # print("dac_loss",dac_loss)

            # softmax version dac loss
            if self._config.has_dac_loss:
                temperature=1e-3
                inv_t=timesteps/self._config.infer_steps
                inv_traj=noisy_traj_points+(1.-inv_t)*pred
                inv_traj=self.denormalize_xy_rotation(inv_traj,inv_traj.shape[-2],times=10)[:,3:1+8,:]
                sdf=torch.nn.functional.softmax(drivable_area_map/temperature,dim=1)[:,0:1,:,:]
                distance=distance_transform(sdf,h=0.1)
                inv_distance=distance_transform(1-sdf,h=0.1)
                drivable_sdf=(distance-inv_distance)*self._config.bev_pixel_size
                # dac_loss=self.esdf_manager(inv_traj,drivable_sdf.squeeze()) if flag!=2 else torch.tensor([0.0]).to(gt_trajs)
                dac_loss,_=self.esdf_manager(inv_traj,drivable_sdf.squeeze(1))

            if self._config.has_gt_dac_loss:
                inv_t=timesteps/self._config.infer_steps
                # inv_traj=noisy_traj_points+(1.-inv_t)*pred
                inv_traj=noise+pred
                inv_traj=self.denormalize_xy_rotation(inv_traj,inv_traj.shape[-2],times=10)[:,3:1+8,:]
                sdf=targets['drivable_area_map'].cpu().numpy()
                distance=distance_transform_edt(sdf)
                inv_distance=distance_transform_edt(1-sdf)
                drivable_sdf=(distance-inv_distance)*self._config.bev_pixel_size
                drivable_sdf=torch.from_numpy(drivable_sdf).to(gt_trajs)
                dac_loss,_=self.esdf_manager(inv_traj,drivable_sdf,is_sum=True)
        else:

            # normal_dummy_goalflow_trajs = self.normalize_xy_rotation(dummy_goalflow_trajs, N=dummy_goalflow_trajs.shape[-2], times=10).to(dummy_goalflow_trajs).mul_(0.1)
            # if self._config.warm_up:
            #     noise=torch.randn(size=(batch_size*self._config.anchor_size,dummy_goalflow_trajs.shape[-2],30),dtype=dtype,device=device)
            #     rate=self._config.renoise_steps/self._config.infer_steps
            #     trajs=normal_dummy_goalflow_trajs*(1.-rate)+noise*rate
            # else:
            #     # x_shift=torch.tensor(X_NOISE,dtype=torch.float32).to(global_feature)
            #     # noise=torch.randn(normal_trajs.shape,device=normal_trajs.device).to(global_feature)
            #     # noise=noise[:,0,:]+x_shift
            #     # noise=noise*0.1
            #     trajs=noise
            trajs=noise
            if self._config.start:
                trajs[:,[0],:]=normal_trajs[:,[0],:]
            if self._config.end:
                trajs[:,[8],:]=normal_trajs[:,[8],:]

            features=global_feature[0].unsqueeze(1).repeat(1,self._config.anchor_size,1,1).view(-1,2,self._config.tf_d_model)
            embedding=global_feature[1].unsqueeze(1).repeat(1,self._config.anchor_size,1,1).view(-1,2,self._config.tf_d_model)
            global_feature=(features,embedding)
            # repeated_tensor=global_feature.unsqueeze(1).repeat(1,self._config.anchor_size,1)
            # expanded_tensor=repeated_tensor.view(-1,self.noise_net_dim)
            # ===================== choose different infer steps ========================
            # timesteps=torch.arange(0,self.infer_steps).to(device)
            if not self._config.warm_up:
                timesteps=torch.arange(0,self._config.infer_steps,1).to(device)
            else:
                timesteps=torch.arange(self._config.infer_steps-self._config.renoise_steps,self._config.infer_steps,1).to(device)

            if self._config.cur_sampling:
                timesteps = torch.linspace(0, 1, self._config.infer_steps+1)
                t_shifted = 1-(self._config.alpha * timesteps) / (1 + (self._config.alpha - 1) * timesteps)
                t_shifted = t_shifted.flip(0)
                t_shifted*=self._config.infer_steps

                for t_curr, t_prev in zip(t_shifted[:-1], t_shifted[1:]):
                    step = t_prev - t_curr
                    net_output_nonavi=self.denoise(trajs,t_curr,global_feature,navi_dropout=True)
                    net_output=net_output_nonavi.reshape(self._config.anchor_size*batch_size,12,30)
                    trajs=trajs.detach().clone()+net_output*(step / self._config.infer_steps)

            else:
                for t in timesteps:
                    net_output_nonavi=self.denoise(trajs,t,global_feature,navi_dropout=True)
                    # net_output_uncond=self.denoise(trajs,t,global_feature,force_dropout=True)
                    # net_output=(0.8*net_output_nonavi+0.1*net_output_uncond+0.1*net_output_cond).reshape(self._config.anchor_size*batch_size,12,30)
                    # net_output=(0.9*net_output_nonavi+0.1*net_output_uncond).reshape(self._config.anchor_size*batch_size,12,30)
                    net_output=net_output_nonavi.reshape(self._config.anchor_size*batch_size,12,30)
                    # net_output=(net_output_uncond+self._config.cond_weight*(net_output_nonavi-net_output_uncond)).reshape(self._config.anchor_size*batch_size,12,30)
                    trajs=trajs.detach().clone()+net_output*(1 / self._config.infer_steps)
            
            diffusion_output = self.denormalize_xy_rotation(trajs, N=gt_trajs.shape[-2], times=10)

            pred_trajs=diffusion_output.reshape(batch_size,self._config.anchor_size,-1,3)

            # bz, N, n_points, n_ = pred_trajs.shape

            # # Step 1: Compute the distance of each trajectory
            # distances = torch.norm(pred_trajs[:, :, 1:] - pred_trajs[:, :, :-1], dim=-1).sum(dim=-1)  # Shape: [bz, N]

            # # Step 2: Sort trajectories by distance
            # sorted_indices = torch.argsort(distances, dim=1, descending=True)  # Indices of sorted distances

            # # Step 3: Select the top 80% of trajectories
            # top_80_percent_count = int(N * 0.8)
            # selected_indices = sorted_indices[:, :top_80_percent_count]

            # # Step 4: Gather the top 80% trajectories and compute their mean
            # pred_trajs = torch.gather(pred_trajs, 1, selected_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, n_points, n_))

            # add scorer
            max_times=3
            flag=False
            if self._config.dac_scorer:
                # last_y=pred_trajs[0,:,-1,1]
                # y_max_indices=last_y.argmax()
                # y_min_indices=last_y.argmin()
                # traj_max_y=pred_trajs[0,y_max_indices,:1+8,:]
                # traj_min_y=pred_trajs[0,y_min_indices,:1+8,:]
                traj_mean=pred_trajs[0,:,1:1+8,:].mean(0)
                # scorer_trajs=torch.stack([traj_mean,traj_max_y,traj_min_y])

                scorer_trajs=traj_mean.unsqueeze(0)
                sdf=bev_semantic_map.argmax(1).cpu().numpy()
                # sdf=drivable_area_map.argmax(1).cpu().numpy()
                sdf[sdf==2]=0
                sdf[sdf!=0]=1
                # sdf=drivable_area_map.argmax(1).cpu().numpy()
                # path="/home/users/zebin.xing/workspace/navsim/mid_data/sdf_test/"
                # dummy_sdf=np.zeros_like(sdf)
                # visualize_bev(np.squeeze(np.concatenate([dummy_sdf,sdf],axis=1)),path,token)
                distance=distance_transform_edt(sdf)
                inv_distance=distance_transform_edt(1-sdf)
                drivable_sdf=(distance-inv_distance)*self._config.bev_pixel_size
                drivable_sdf=torch.from_numpy(drivable_sdf.repeat(scorer_trajs.shape[0],axis=0)).to(scorer_trajs)
                dac_loss,direction=self.esdf_manager(scorer_trajs[:,2:,:],drivable_sdf,is_sum=False,has_direction=False)
                dac_loss=dac_loss.sum(-1)
                chosen_traj=scorer_trajs[0]
                drivable_sdf=drivable_sdf.repeat(3,1,1)
                # if chosen_traj[-1,0]>35:
                #     mask = chosen_traj[:, 0] < 35
                #     indices = torch.where(mask)[0]
                #     farthest_point_index = indices[-1]
                #     new_farthest_points = chosen_traj[:farthest_point_index+1]
                if dac_loss[0]==0.0:
                    chosen_traj=scorer_trajs[0]
                else:
                    dac_min_index=dac_loss.argmin()
                    chosen_traj=scorer_trajs[dac_min_index]

            
            if self._config.start:
                if self._config.dac_scorer:
                    pred=chosen_traj[:8,:].unsqueeze(0)
                else:
                    pred=pred_trajs[:,:,1:1+8,:].mean(1)
            else:
                pred=pred_trajs[:,:,:8,:].mean(1)

        # goalflow query
        # pred = self._trajectory_head(trajectory_query)
        # output.update(trajectory)
        output: Dict[str,torch.Tensor]={}
        output: Dict[str, torch.Tensor] = {"bev_semantic_map": bev_semantic_map}
        output.update({'trajectory':pred})
        output.update({'target':target})
        output.update({'mlp_trajectory':goalflow_trajs})
        # output.update({'navi_point':navi_point})
        output.update(agents)
        if self._config.training and (self._config.has_dac_loss or self._config.has_gt_dac_loss):
            output.update({'dac_loss':dac_loss})

        return output


    def encode_scene_features(self, ego_agent_features):
        ego_features = ego_agent_features # Bx1xD

        ego_type_embedding = self.type_embedding(torch.as_tensor([[0]], device=ego_features.device))
        ego_type_embedding = ego_type_embedding.repeat(ego_features.shape[0],1,1)

        return ego_features, ego_type_embedding

    def encode_status_features(self, ego_agent_features):
        ego_features = self.history_encoder(ego_agent_features) # Bx4x11 -> Bx4xD

        ego_type_embedding = self.type_embedding(torch.as_tensor([[0]], device=ego_features.device))
        ego_type_embedding = ego_type_embedding.repeat(ego_features.shape[0],4,1)

        return ego_features, ego_type_embedding
    
    def encode_command_features(self, ego_agent_features):
        ego_features = ego_agent_features

        ego_type_embedding = self.type_embedding(torch.as_tensor([[2]], device=ego_features.device))
        ego_type_embedding = ego_type_embedding.repeat(ego_features.shape[0],1,1)

        return ego_features, ego_type_embedding

    def encode_navi_features(self, ego_agent_features):
        ego_features = ego_agent_features # Bx1xD

        ego_type_embedding = self.type_embedding(torch.as_tensor([[2]], device=ego_features.device))
        ego_type_embedding = ego_type_embedding.repeat(ego_features.shape[0],1,1)

        return ego_features, ego_type_embedding
    
    def encode_mid_features(self, ego_agent_features):
        ego_features = ego_agent_features # Bx1xD

        ego_type_embedding = self.type_embedding(torch.as_tensor([[3]], device=ego_features.device))
        ego_type_embedding = ego_type_embedding.repeat(ego_features.shape[0],1,1)

        return ego_features, ego_type_embedding

    def denoise(self, ego_trajectory, sigma, state_features,force_dropout=False,navi_dropout=False,scene_dropout=False,mid_dropout=False):
        batch_size = ego_trajectory.shape[0]

        state_features, state_type_embedding = state_features
        
        # Trajectory features
        ego_trajectory = ego_trajectory.reshape(ego_trajectory.shape[0],self._config.trajectory_generation_len,30)
        trajectory_features = self.trajectory_encoder(ego_trajectory)

        trajectory_type_embedding = self.type_embedding(
            torch.as_tensor([1], device=ego_trajectory.device)
        )[None].repeat(batch_size,self._config.trajectory_generation_len,1)

        # Concatenate all features
        if navi_dropout:
            state_features[:,1,:]*=0
        if mid_dropout:
            state_features[:,2,:]*=0
        if scene_dropout:
            state_features[:,0,:]*=0
        all_features = torch.cat([state_features, trajectory_features], dim=1)
        if force_dropout:
            all_features=all_features*0
        all_type_embedding = torch.cat([state_type_embedding, trajectory_type_embedding], dim=1)
        
        # Sigma encoding
        sigma = sigma.reshape(-1,1)
        if sigma.numel() == 1:
            sigma = sigma.repeat(batch_size,1)
        sigma = sigma.float() / self._config.infer_steps
        sigma_embeddings = self.sigma_encoder(sigma)
        sigma_embeddings = sigma_embeddings.reshape(batch_size,1,self._config.tf_d_model)

        # Concatenate sigma features and project back to original feature_dim
        sigma_embeddings = sigma_embeddings.repeat(1,all_features.shape[1],1)
        all_features = torch.cat([all_features, sigma_embeddings], dim=2)
        all_features = self.sigma_proj_layer(all_features)

        # Generate attention mask
        seq_len = all_features.shape[1]
        indices = torch.arange(seq_len, device=all_features.device)
        dists = (indices[None] - indices[:,None]).abs()
        attn_mask = dists > 1       # TODO: magic number

        # Generate relative temporal embeddings
        temporal_embedding = self.trajectory_time_embeddings(indices[None].repeat(batch_size,1))

        # Global self-attentions
        for layer in self.global_attention_layers:            
            all_features, _ = layer(
                all_features, None, None, None,
                seq1_pos=temporal_embedding,
                seq1_sem_pos=all_type_embedding,
                attn_mask_11=attn_mask
            )

        # trajectory_features = all_features[:,-12:]
        trajectory_features = all_features[:,-self._config.trajectory_generation_len:]
        out = self.decoder_mlp(trajectory_features).reshape(trajectory_features.shape[0],-1)

        return out # , all_weights

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

    def normalize_xy_rotation3(self, trajectory, N=8, times=10):
        # print(trajectory[0])
        # print(trajectory[0].max())
        # print(trajectory[0].min())
        batch, num_pts, dim = trajectory.shape
        downsample_trajectory = trajectory[:, :N, :].detach().clone()
        x_scale = 90
        y_scale = 45
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
    
    def denormalize_xy_rotation3(self, trajectory, N=8, times=10):
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
        final_trajectory[:, :, 0] *= 90
        final_trajectory[:, :, 1] *= 45
        final_trajectory[:,:,2]=final_trajectory[:,:,2].tanh() * math.pi
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
        agent_states[..., BoundingBox2DIndex.POINT] = (
            agent_states[..., BoundingBox2DIndex.POINT].tanh() * 32
        )
        agent_states[..., BoundingBox2DIndex.HEADING] = (
            agent_states[..., BoundingBox2DIndex.HEADING].tanh() * np.pi
        )

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

def get_train_tuple_trig(z0=None, z1=None):
    t=torch.randn(z1.shape[0],1,1).to(z0.device)*(torch.pi/2)
    z_t=torch.cos(t)*z1+torch.sin(t)*z0
    target=torch.cos(t)*z0-torch.sin(t)*z1
    return z_t.float(),t.float(),target.float()