import pytorch_lightning as pl
from torch import Tensor
from typing import Dict, Tuple
import os
import numpy as np

from navsim.agents.abstract_agent import AbstractAgent

class AgentLightningModule(pl.LightningModule):
    def __init__(
        self,
        agent: AbstractAgent,
    ):
        super().__init__()
        self.agent = agent

    def _step(
        self,
        batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
        logging_prefix: str,
    ):
        features, targets = batch
        prediction = self.agent.forward(features,targets)
        loss_dict = self.agent.compute_loss(features, targets, prediction)
        for k,v in loss_dict.items():
            if v is not None:
                self.log(f"{logging_prefix}/{k}", v, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True,batch_size=len(batch[0]))
        return loss_dict['loss']
    
    def training_step(
        self,
        batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
        batch_idx: int
    ):
        return self._step(batch, "train")

    def validation_step(
        self,
        batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
        batch_idx: int
    ):
        return self._step(batch, "val")
    
    def test_step(
        self,
        batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
        batch_idx: int
    ):
        features, targets = batch
        prediction = self.agent.forward(features,targets)
        if self.agent._config.generate=='goal_score':
            log_dir = self.logger.log_dir if self.logger else "./default_log_dir"
            navis_dir_path = os.path.join(log_dir, "navis")
            os.makedirs(navis_dir_path, exist_ok=True)
            im_path=os.path.join(navis_dir_path,'im')
            dac_path=os.path.join(navis_dir_path,'dac')
            os.makedirs(im_path, exist_ok=True)
            os.makedirs(dac_path,exist_ok=True)
            batch_size=prediction['im_scores'].shape[0]
            for i in range(batch_size):
                token=features['token'][i]
                im_score=prediction['im_scores'][i].squeeze().cpu().numpy()
                dac_score=prediction['dac_scores'][i].squeeze().cpu().numpy()
                np.save(f'{dac_path}/{token}.npy',dac_score)
                np.save(f'{im_path}/{token}.npy',im_score)

        elif self.agent._config.generate=='trajectory':
            log_dir = self.logger.log_dir if self.logger else "./default_log_dir"
            trajs_dir_path = os.path.join(log_dir, "trajs")
            os.makedirs(trajs_dir_path, exist_ok=True)
            
            trajs_num = prediction['trajectory'].shape[0]
            for i in range(trajs_num):
                token = features['token'][i]
                traj = prediction['trajectory'][i].squeeze(0).cpu().numpy()
                np.save(f'{trajs_dir_path}/{token}.npy', traj)
        else:
            raise Exception('generate should be in (goal_score,trajectory)')
        return prediction

    def configure_optimizers(self):
        return self.agent.get_optimizers()
