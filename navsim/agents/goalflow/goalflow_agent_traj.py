from typing import Any, List, Dict, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import pytorch_lightning as pl

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)
from navsim.agents.goalflow.goalflow_config import GoalFlowConfig
from navsim.agents.goalflow.goalflow_model_traj import GoalFlowTrajModel
from navsim.agents.goalflow.goalflow_callback import GoalFlowCallback
from navsim.agents.goalflow.goalflow_loss import goalflow_loss
from navsim.agents.goalflow.goalflow_features import (
    GoalFlowFeatureBuilder,
    GoalFlowTargetBuilder,
)
from torch.optim.lr_scheduler import StepLR


class GoalFlowTrajAgent(AbstractAgent):
    def __init__(
        self,
        config: GoalFlowConfig,
        lr: float,
        step_size: int = 20,
        gamma: float = 0.8,
        checkpoint_path: str = None,
    ):
        super().__init__()

        self._config = config
        self._lr = lr
        self._step_size=step_size
        self._gamma=gamma

        self._checkpoint_path = checkpoint_path
        self._goalflow_model = GoalFlowTrajModel(config)

    def name(self) -> str:
        """Inherited, see superclass."""

        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
        else:
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))["state_dict"]
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig.build_all_sensors(include=[3])

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [GoalFlowTargetBuilder(config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [GoalFlowFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self._goalflow_model(features,targets)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> Dict:
        return goalflow_loss(targets, predictions, self._config)

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self._goalflow_model.parameters()),
            lr=self._lr
        )
        
        scheduler = StepLR(optimizer, step_size=self._step_size, gamma=self._gamma)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }
        # return torch.optim.Adam(filter(lambda p: p.requires_grad, self._goalflow_model.parameters()), lr=self._lr)

    def get_training_callbacks(self) -> List[pl.Callback]:
        return [GoalFlowCallback(self._config),pl.callbacks.ModelCheckpoint(every_n_epochs=5, save_top_k=-1)]
