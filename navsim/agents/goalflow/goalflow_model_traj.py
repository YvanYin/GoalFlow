import math
from typing import Dict
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from navsim.agents.goalflow.goalflow_config import GoalFlowConfig
from navsim.common.enums import StateSE2Index
from navsim.agents.goalflow.goalflow_features import BoundingBox2DIndex
from navsim.agents.goalflow.utils import pos2posemb2d
from navsim.agents.goalflow.diffusion_es import SinusoidalPosEmb
from navsim.agents.goalflow.diffusion_es import ParallelAttentionLayer
from navsim.agents.goalflow.diffusion_es import RotaryPositionEncoding
from navsim.agents.goalflow.v99_backbone import V299Backbone
import os

class GoalFlowTrajModel(nn.Module):
    def __init__(self, config: GoalFlowConfig):

        super().__init__()

        self._query_splits = [
            1,
            config.num_bounding_boxes,
        ]

        self._config = config
        # self._backbone = goalflowBackbone(config)
        self._backbone=V299Backbone(config)
        if not self._config.voc_path=='':
            self.cluster_points=np.load(self._config.voc_path)

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
        self._status_encoding=nn.Linear(4+2+2,config.tf_d_model)
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

        if not self._config.only_perception:
            # diffusion es: https://github.com/bhyang/diffusion-es
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

        if self._config.freeze_perception:
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
        def minmax(t_tensor):
            batch_min=torch.min(t_tensor,dim=-1).values.unsqueeze(-1)
            batch_max=torch.max(t_tensor,dim=-1).values.unsqueeze(-1)
            return (t_tensor-batch_min)/(batch_max-batch_min)
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


        # =================================== perception ==================================================
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

        agents = self._agent_head(agents_query)
        bev_semantic_map = self._bev_semantic_head(bev_feature_upscale)

        if self._config.only_perception:
            output: Dict[str, torch.Tensor] = {"bev_semantic_map": bev_semantic_map}
            output.update(agents)
            output.update({'trajectory':gt_trajs}) # use gt trajectorie as placeholder.
            return output

        # =================================== flow ==================================================
        target=gt_trajs
        if self._config.has_navi:
            navi=gt_trajs[:,7:8,:2].clone().to(device)
            navi_feature=pos2posemb2d(navi,num_pos_feats=self._config.tf_d_model//2).squeeze(1)
            global_feature=trajectory_query.squeeze(1)
        elif self._config.has_student_navi:
            if isinstance(features['token'], str):
                if os.path.isfile(f'{self._config.score_path}/im/{token[i]}.npy'):
                    im_score=np.load(f'{self._config.score_path}/im/{token[i]}.npy')
                else:
                    im_score=np.zeros((8192,))
                if os.path.isfile(f'{self._config.score_path}/dac/{token[i]}.npy'):
                    dac_score=np.load(f'{self._config.score_path}/dac/{token[i]}.npy')
                else:
                    dac_score=np.zeros((8192,))
                im_scores=torch.from_numpy(im_scores).to(gt_trajs)
                dac_scores=torch.from_numpy(dac_scores).to(gt_trajs)
                cluster_points_tensor=torch.from_numpy(self.cluster_points).to(gt_trajs)
                if self._config.ep_point_weight>0.0:
                    goal_distance=torch.norm(cluster_points_tensor[...,:2],dim=-1)
                    goal_distance_norm=minmax(goal_distance)
                    final_scores=0.1*torch.log(torch.nn.functional.softmax(im_scores.squeeze(),dim=-1)+1e-7)+self._config.theta*torch.log(torch.sigmoid(dac_scores.squeeze())+1e-7)\
                                +self._config.ep_point_weight*goal_distance_norm
                else:
                    final_scores=0.1*torch.log(torch.nn.functional.softmax(im_scores.squeeze(),dim=-1)+1e-7)+self._config.theta*torch.log(torch.sigmoid(dac_scores.squeeze())+1e-7)
                
                navi=cluster_points_tensor[final_scores.argmax()][:2].unsqueeze(0).unsqueeze(0)
                navi_feature=pos2posemb2d(navi,num_pos_feats=self._config.tf_d_model//2).squeeze(1).to(gt_trajs)
                global_feature=trajectory_query.squeeze(1)
            else:
                im_scores=[]
                dac_scores=[]
                for i in range(batch_size):
                    if os.path.isfile(f'{self._config.score_path}/im/{token[i]}.npy'):
                        im_score=np.load(f'{self._config.score_path}/im/{token[i]}.npy')
                    else:
                        im_score=np.zeros((8192,))
                    if os.path.isfile(f'{self._config.score_path}/dac/{token[i]}.npy'):
                        dac_score=np.load(f'{self._config.score_path}/dac/{token[i]}.npy')
                    else:
                        dac_score=np.zeros((8192,))
                    im_scores.append(torch.from_numpy(im_score).to(gt_trajs))
                    dac_scores.append(torch.from_numpy(dac_score).to(gt_trajs))
                im_scores=torch.stack(im_scores)
                dac_scores=torch.stack(dac_scores)
                cluster_points_tensor=torch.from_numpy(self.cluster_points).to(gt_trajs).unsqueeze(0).repeat(batch_size,1,1)
                if self._config.ep_point_weight>0.0:
                    goal_distance=torch.norm(cluster_points_tensor[...,:2],dim=-1)
                    goal_distance_norm=minmax(goal_distance)
                    final_scores=0.1*torch.log(torch.nn.functional.softmax(im_scores.squeeze(),dim=-1)+1e-7)+self._config.theta*torch.log(torch.sigmoid(dac_scores.squeeze())+1e-7)\
                            +self._config.ep_point_weight*goal_distance_norm
                else:
                    final_scores=0.1*torch.log(torch.nn.functional.softmax(im_scores.squeeze(),dim=-1)+1e-7)+self._config.theta*torch.log(torch.sigmoid(dac_scores.squeeze())+1e-7)
                
                topk_indices=torch.topk(final_scores,self._config.topk).indices
                topk_indices=topk_indices.unsqueeze(-1).expand(-1,-1,cluster_points_tensor.shape[-1])
                navi=torch.gather(cluster_points_tensor,dim=1,index=topk_indices).mean(1)[...,:2].unsqueeze(1)
                navi_feature=pos2posemb2d(navi,num_pos_feats=self._config.tf_d_model//2).squeeze(1).to(gt_trajs)
                global_feature=trajectory_query.squeeze(1)
        else:
            global_feature=trajectory_query.squeeze(1)

        if gt_trajs.shape[-2]==12:
            gt_trajs=gt_trajs[...,:11,:]
        gt_trajs_=gt_trajs.clone()
        if self._config.start:
            start_point=torch.zeros((gt_trajs.shape[0],1,3)).to(gt_trajs)
            gt_trajs_=torch.cat([start_point,gt_trajs_],dim=1)

        target=torch.zeros_like(gt_trajs_)
        normal_trajs = self.normalize_xy_rotation(gt_trajs_, N=gt_trajs_.shape[-2], times=10).to(gt_trajs_)
        if self._config.training:
            noise=torch.randn(size=(batch_size,12,30),device=normal_trajs.device,dtype=dtype).to(global_feature)*self._config.train_scale
        else:
            noise=torch.randn(size=(batch_size*self._config.anchor_size,12,30),dtype=dtype,device=device)*self._config.test_scale
        
        if self._config.has_navi or self._config.has_student_navi:
            global_feature1=self.encode_scene_features(global_feature.unsqueeze(1))
            global_feature2=self.encode_navi_features(navi_feature.unsqueeze(1))
            global_feature=(torch.cat([global_feature1[0],global_feature2[0]],dim=-2),torch.cat([global_feature1[1],global_feature2[1]],dim=-2))
        else:
            global_feature=self.encode_scene_features(global_feature.unsqueeze(1))

        # =================================== flow training ==================================================
        if self._config.training:

            batch_size = normal_trajs.shape[0]

            if self._config.start:
                noise[:,[0],:]=normal_trajs[:,[0],:]

            noisy_traj_points,t,target=get_train_tuple(z0=noise,z1=normal_trajs)

            timesteps=t*self._config.infer_steps
            
            import random
            if self._config.has_navi:
                flag=random.randint(1,3)
                if flag==1:
                    pred=self.denoise(noisy_traj_points,timesteps,global_feature).reshape(batch_size,-1,30)
                elif flag==2:
                    pred=self.denoise(noisy_traj_points,timesteps,global_feature,force_dropout=True).reshape(batch_size,-1,30)
                elif flag==3:
                    pred=self.denoise(noisy_traj_points,timesteps,global_feature,navi_dropout=True).reshape(batch_size,-1,30)
            else:
                flag=random.randint(1,2)
                if flag==1:
                    pred=self.denoise(noisy_traj_points,timesteps,global_feature).reshape(batch_size,-1,30)
                elif flag==2:
                    pred=self.denoise(noisy_traj_points,timesteps,global_feature,force_dropout=True).reshape(batch_size,-1,30)  
        
        # =================================== flow sampling ==================================================
        else:

            trajs=noise
            if self._config.start:
                trajs[:,[0],:]=normal_trajs[:1,[0],:]

            if self._config.has_navi or self._config.has_student_navi:
                features=global_feature[0].unsqueeze(1).repeat(1,self._config.anchor_size,1,1).view(-1,2,self._config.tf_d_model)
                embedding=global_feature[1].unsqueeze(1).repeat(1,self._config.anchor_size,1,1).view(-1,2,self._config.tf_d_model)
                global_feature=(features,embedding)
            else:
                features=global_feature[0].unsqueeze(1).repeat(1,self._config.anchor_size,1,1).view(-1,1,self._config.tf_d_model)
                embedding=global_feature[1].unsqueeze(1).repeat(1,self._config.anchor_size,1,1).view(-1,1,self._config.tf_d_model)
                global_feature=(features,embedding)
            timesteps=torch.arange(self._config.infer_steps,self._config.infer_steps,1).to(device)

            # ========================== fusion main trajectory and shadow trajectory ========================================
            if self._config.fusion:
                timesteps = torch.linspace(0, 1, self._config.infer_steps+1).to(device)
                t_shifted = 1-(self._config.alpha * timesteps) / (1 + (self._config.alpha - 1) * timesteps)
                t_shifted = t_shifted.flip(0)
                t_shifted*=self._config.infer_steps
                for t_curr, t_prev in zip(t_shifted[:-1], t_shifted[1:]):
                    step = t_prev - t_curr
                    net_output_cond=self.denoise(trajs,t_curr,global_feature)
                    net_output_cond=net_output_cond.reshape(self._config.anchor_size*batch_size,12,30)
                    trajs=trajs.detach().clone()+net_output_cond*(step / self._config.infer_steps)
                diffusion_output_cond = self.denormalize_xy_rotation(trajs, N=gt_trajs.shape[-2], times=10)
                diffusion_output_cond=diffusion_output_cond.reshape(batch_size,self._config.anchor_size,-1,3)
                if self._config.use_nearest:
                    distances=torch.norm(diffusion_output_cond[:,:,8,:2]-navi,dim=-1)
                    scores=distances
                    if self._config.ep_score_weight>0.0:
                        distances_norm=minmax(distances)
                        progress=torch.norm(diffusion_output_cond[:,:,8,:2],dim=-1)
                        progress_norm=minmax(progress)
                        scores=(1.-self._config.ep_score_weight)*distances_norm-self._config.ep_score_weight*progress_norm
                    min_index=torch.argmin(scores,dim=1)
                    pred_trajs_cond=diffusion_output_cond[torch.arange(batch_size),min_index].unsqueeze(1)
                else:
                    pred_trajs_cond=torch.mean(diffusion_output_cond,dim=1,keepdim=True)

                trajs=noise
                for t_curr, t_prev in zip(t_shifted[:-1], t_shifted[1:]):
                    step = t_prev - t_curr
                    net_output_nonavi=self.denoise(trajs,t_curr,global_feature,navi_dropout=True)
                    net_output_nonavi=net_output_nonavi.reshape(self._config.anchor_size*batch_size,12,30)
                    trajs=trajs.detach().clone()+net_output_nonavi*(step / self._config.infer_steps)
                diffusion_output_nonavi = self.denormalize_xy_rotation(trajs, N=gt_trajs.shape[-2], times=10)
                diffusion_output_nonavi=diffusion_output_nonavi.reshape(batch_size,self._config.anchor_size,-1,3)
                pred_trajs_nonavi=torch.mean(diffusion_output_nonavi,dim=1,keepdim=True)

                if self._config.cond_threshold>0.0:
                    distance=torch.norm(pred_trajs_nonavi[:,:,8,:2]-navi,dim=-1).squeeze()
                    progress=torch.norm(pred_trajs_nonavi[:,:,8,:2],dim=-1).squeeze()
                    cond_mask=(distance/progress)>self._config.cond_threshold
                    cond_mask=cond_mask.view(-1,1,1,1)
                    pred_trajs=torch.where(cond_mask,pred_trajs_nonavi,pred_trajs_cond)

                else:
                    pred_trajs=self._config.beta*pred_trajs_cond+(1.0-self._config.beta)*pred_trajs_nonavi
                
            # ========================== only use main trajectory ========================================
            else:
                if self._config.cur_sampling:
                    timesteps = torch.linspace(0, 1, self._config.infer_steps+1).to(device)
                    t_shifted = 1-(self._config.alpha * timesteps) / (1 + (self._config.alpha - 1) * timesteps)
                    t_shifted = t_shifted.flip(0)
                    t_shifted*=self._config.infer_steps

                    for t_curr, t_prev in zip(t_shifted[:-1], t_shifted[1:]):
                        step = t_prev - t_curr
                        if not self._config.cond_weight==1.0:
                            net_output_nonavi=self.denoise(trajs,t_curr,global_feature,navi_dropout=True)
                            net_output_cond=self.denoise(trajs,t_curr,global_feature)
                            net_output=((1.-self._config.cond_weight)*net_output_nonavi+self._config.cond_weight*net_output_cond).reshape(self._config.anchor_size*batch_size,12,30)
                        else:
                            if self._config.has_student_navi:
                                net_output_nonavi=self.denoise(trajs,t_curr,global_feature)
                            elif self._config.has_navi:
                                net_output_nonavi=self.denoise(trajs,t_curr,global_feature,navi_dropout=True)
                            else:
                                net_output_nonavi=self.denoise(trajs,t_curr,global_feature)
                            net_output=net_output_nonavi.reshape(self._config.anchor_size*batch_size,12,30)
                        trajs=trajs.detach().clone()+net_output*(step / self._config.infer_steps)

                else:
                    for t in timesteps:
                        if not self._config.cond_weight==1.0:
                            net_output_nonavi=self.denoise(trajs,t,global_feature,navi_dropout=True)
                            net_output_cond=self.denoise(trajs,t,global_feature)
                            net_output=((1.-self._config.cond_weight)*net_output_nonavi+self._config.cond_weight*net_output_cond).reshape(self._config.anchor_size*batch_size,12,30)
                        else:
                            if self._config.has_student_navi:
                                net_output_nonavi=self.denoise(trajs,t,global_feature)
                            elif self._config.has_navi:
                                net_output_nonavi=self.denoise(trajs,t,global_feature,navi_dropout=True)
                            else:
                                net_output_nonavi=self.denoise(trajs,t,global_feature)
                            net_output=net_output_nonavi.reshape(self._config.anchor_size*batch_size,12,30)
                        trajs=trajs.detach().clone()+net_output*(1 / self._config.infer_steps)
                
                diffusion_output = self.denormalize_xy_rotation(trajs, N=gt_trajs.shape[-2], times=10)

                pred_trajs=diffusion_output.reshape(batch_size,self._config.anchor_size,-1,3)

            # ========================== trajectory scorer ========================================
            if self._config.use_nearest and not self._config.fusion:
                distances=torch.norm(pred_trajs[:,:,8,:2]-navi,dim=-1)
                scores=distances
                if self._config.ep_score_weight>0.0:
                    distances_norm=minmax(distances)
                    progress=torch.norm(pred_trajs[:,:,8,:2],dim=-1)
                    progress_norm=minmax(progress)
                    scores=(1.-self._config.ep_score_weight)*distances_norm-self._config.ep_score_weight*progress_norm
                min_index=torch.argmin(scores,dim=1)
                pred_trajs=pred_trajs[torch.arange(batch_size),min_index].unsqueeze(1)

            if self._config.start:
                pred=pred_trajs[:,:,1:1+8,:].mean(1)
            else:
                pred=pred_trajs[:,:,:8,:].mean(1)

        output: Dict[str, torch.Tensor] = {"bev_semantic_map": bev_semantic_map}
        output.update({'trajectory':pred})
        output.update({'target':target})
        output.update(agents)

        return output


    def encode_scene_features(self, ego_agent_features):
        ego_features = ego_agent_features

        ego_type_embedding = self.type_embedding(torch.as_tensor([[0]], device=ego_features.device))
        ego_type_embedding = ego_type_embedding.repeat(ego_features.shape[0],1,1)

        return ego_features, ego_type_embedding

    def encode_status_features(self, ego_agent_features):
        ego_features = self.history_encoder(ego_agent_features)

        ego_type_embedding = self.type_embedding(torch.as_tensor([[2]], device=ego_features.device))
        ego_type_embedding = ego_type_embedding.repeat(ego_features.shape[0],4,1)

        return ego_features, ego_type_embedding
    
    def encode_navi_features(self, ego_agent_features):
        ego_features = ego_agent_features # Bx1xD

        ego_type_embedding = self.type_embedding(torch.as_tensor([[2]], device=ego_features.device))
        ego_type_embedding = ego_type_embedding.repeat(ego_features.shape[0],1,1)

        return ego_features, ego_type_embedding

    def denoise(self, ego_trajectory, sigma, state_features,force_dropout=False,navi_dropout=False):
        batch_size = ego_trajectory.shape[0]

        state_features, state_type_embedding = state_features
        
        # Trajectory features
        ego_trajectory = ego_trajectory.reshape(ego_trajectory.shape[0],12,30)
        trajectory_features = self.trajectory_encoder(ego_trajectory)

        trajectory_type_embedding = self.type_embedding(
            torch.as_tensor([1], device=ego_trajectory.device)
        )[None].repeat(batch_size,12,1)

        # Concatenate all features
        if navi_dropout:
            state_features[:,-1,:]*=0
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

        trajectory_features = all_features[:,-12:]
        out = self.decoder_mlp(trajectory_features).reshape(trajectory_features.shape[0],-1)

        return out # , all_weights

    def xy_rotation(self, trajectory, N=8, times=10):
        batch, num_pts, dim = trajectory.shape
        downsample_trajectory = trajectory[:, :N, :].detach().clone()

        rotated_trajectories = []
        for i in range(times):
            theta = 2 * math.pi * i / times
            rotation_matrix, _ = get_rotation_matrices(theta)
            rotation_matrix = rotation_matrix.unsqueeze(0).expand(downsample_trajectory.size(0), -1, -1).to(downsample_trajectory)
            
            rotated_trajectory = apply_rotation(downsample_trajectory[:,:,:2], rotation_matrix)
            rotated_trajectory=torch.cat([rotated_trajectory,downsample_trajectory[:,:,-1:].permute(0,2,1)],dim=1)
            rotated_trajectories.append(rotated_trajectory)
        resulting_trajectory = torch.cat(rotated_trajectories, 1)
        trajectory = resulting_trajectory.permute(0,2,1)
        return trajectory

    def normalize_xy_rotation(self, trajectory, N=8, times=10):
        downsample_trajectory = trajectory[:, :N, :].detach().clone()
        x_scale = 60
        y_scale = 15
        heading_scale = math.pi
        downsample_trajectory[:, :, 0] /= x_scale
        downsample_trajectory[:, :, 1] /= y_scale
        downsample_trajectory[:,:,2]/=heading_scale
        downsample_trajectory[:,:,2]=downsample_trajectory[:,:,2].atanh()
        
        rotated_trajectories = []
        for i in range(times):
            theta = 2 * math.pi * i / times
            rotation_matrix, _ = get_rotation_matrices(theta)
            rotation_matrix = rotation_matrix.unsqueeze(0).expand(downsample_trajectory.size(0), -1, -1).to(downsample_trajectory)
            
            rotated_trajectory = apply_rotation(downsample_trajectory[:,:,:2], rotation_matrix)
            rotated_trajectory=torch.cat([rotated_trajectory,downsample_trajectory[:,:,-1:].permute(0,2,1)],dim=1)
            rotated_trajectories.append(rotated_trajectory)
        resulting_trajectory = torch.cat(rotated_trajectories, 1)
        trajectory = resulting_trajectory.permute(0,2,1)
        return trajectory
    
    def denormalize_xy_rotation(self, trajectory, N=8, times=10):
        inverse_rotated_trajectories = []
        for i in range(times):
            theta = 2 * math.pi * i / 10
            rotation_matrix, inverse_rotation_matrix = get_rotation_matrices(theta)
            inverse_rotation_matrix = inverse_rotation_matrix.unsqueeze(0).expand(trajectory.size(0), -1, -1).to(trajectory)
        
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
        downsample_trajectory = trajectory[:, :N, :].detach().clone()
        x_scale = 45.
        y_scale = 42.
        heading_scale = math.pi
        downsample_trajectory[:,:,0] -= x_scale
        downsample_trajectory[:, :, 0] /= x_scale
        downsample_trajectory[:, :, 1] /= y_scale
        downsample_trajectory[:,:,2]/=heading_scale
        
        rotated_trajectories = []
        for i in range(times):
            theta = 2 * math.pi * i / times
            rotation_matrix, _ = get_rotation_matrices(theta)
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
            theta = 2 * math.pi * i / 10
            rotation_matrix, inverse_rotation_matrix = get_rotation_matrices(theta)
            inverse_rotation_matrix = inverse_rotation_matrix.unsqueeze(0).expand(trajectory.size(0), -1, -1).to(trajectory)
        
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
        downsample_trajectory = trajectory[:, :N, :].detach().clone()
        x_scale = 90
        y_scale = 45
        heading_scale = math.pi
        downsample_trajectory[:, :, 0] /= x_scale
        downsample_trajectory[:, :, 1] /= y_scale
        downsample_trajectory[:,:,2]/=heading_scale
        downsample_trajectory[:,:,2]=downsample_trajectory[:,:,2].atanh()
        
        rotated_trajectories = []
        for i in range(times):
            theta = 2 * math.pi * i / times
            rotation_matrix, _ = get_rotation_matrices(theta)
            rotation_matrix = rotation_matrix.unsqueeze(0).expand(downsample_trajectory.size(0), -1, -1).to(downsample_trajectory)
            
            rotated_trajectory = apply_rotation(downsample_trajectory[:,:,:2], rotation_matrix)
            rotated_trajectory=torch.cat([rotated_trajectory,downsample_trajectory[:,:,-1:].permute(0,2,1)],dim=1)
            rotated_trajectories.append(rotated_trajectory)
        resulting_trajectory = torch.cat(rotated_trajectories, 1)
        trajectory = resulting_trajectory.permute(0,2,1)
        return trajectory
    
    def denormalize_xy_rotation3(self, trajectory, N=8, times=10):
        inverse_rotated_trajectories = []
        for i in range(times):
            theta = 2 * math.pi * i / 10
            rotation_matrix, inverse_rotation_matrix = get_rotation_matrices(theta)
            inverse_rotation_matrix = inverse_rotation_matrix.unsqueeze(0).expand(trajectory.size(0), -1, -1).to(trajectory)
        
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
    theta_tensor = torch.tensor(theta)
    
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
    rotated_trajectory = torch.einsum('bij,bkj->bik', rotation_matrix, trajectory)
    return rotated_trajectory

def get_train_tuple(z0=None, z1=None):
    t = torch.rand(z1.shape[0], 1, 1).to(z0.device)
    z_t =  t * z1 + (1.-t) * z0
    target = z1 - z0
    return z_t.float(), t.float(), target.float()


def coords_to_pixel(coords):
    height,width=256,256
    config=GoalFlowConfig()
    pixel_center = np.array([[0.0, width /2.0]])
    coords_idcs = (coords / config.bev_pixel_size) + pixel_center
    return coords_idcs.astype(np.int32)

def load_feature_from_npz(path):
    with open(path,'rb') as f:
        data=np.load(f,allow_pickle=True)
        array_data=data['traj'].astype(np.float32)
    return array_data