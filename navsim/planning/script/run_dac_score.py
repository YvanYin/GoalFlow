import pandas as pd
from tqdm import tqdm
import traceback

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import numpy as np

from pathlib import Path
from typing import Any, Dict, List, Union, Tuple
from dataclasses import asdict
from datetime import datetime
import logging
import lzma
import pickle
import os
import uuid

from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_utils import worker_map

from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.common.dataloader import MetricCacheLoader
from navsim.agents.abstract_agent import AbstractAgent
from navsim.evaluate.pdm_score import pdm_score,dac_score
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator
)
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.common.dataloader import SceneLoader, SceneFilter
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.common.dataclasses import SensorConfig
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.dataclasses import Trajectory

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    build_logger(cfg)
    worker = build_worker(cfg)

    # Extract scenes based on scene-loader to know which tokens to distribute across workers
    # TODO: infer the tokens per log from metadata, to not have to load metric cache and scenes here
    scene_loader = SceneLoader(
        sensor_blobs_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    num_missing_metric_cache_tokens = len(set(scene_loader.tokens) - set(metric_cache_loader.tokens))
    num_unused_metric_cache_tokens = len(set(metric_cache_loader.tokens) - set(scene_loader.tokens))
    if num_missing_metric_cache_tokens > 0:
        logger.warning(f"Missing metric cache for {num_missing_metric_cache_tokens} tokens. Skipping these tokens.")
    if num_unused_metric_cache_tokens > 0:
        logger.warning(f"Unused metric cache for {num_unused_metric_cache_tokens} tokens. Skipping these tokens.")
    logger.info("Starting pdm scoring of %s scenarios...", str(len(tokens_to_evaluate)))
    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens_list,
        }
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
    ]
    score_rows: List[Tuple[Dict[str, Any], int, int]] = worker_map(worker, run_pdm_score, data_points)
    
    pdm_score_df = pd.DataFrame(score_rows)

    save_path = Path(cfg.output_dir)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    pdm_score_df.to_csv(save_path / f"{timestamp}.csv")

def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Dict[str, Any]]:
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]

    traj_sampling_sim=TrajectorySampling(num_poses=1,interval_length=4.0)
    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    simulator.proposal_sampling=traj_sampling_sim
    scorer.proposal_sampling=traj_sampling_sim
    assert simulator.proposal_sampling == scorer.proposal_sampling, "Simulator and scorer proposal sampling has to be identical"

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    scene_filter: SceneFilter =instantiate(cfg.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    pdm_results: List[Dict[str, Any]] = []
    cluster_points=np.load(cfg.agent.config.voc_path)

    traj_sampling=TrajectorySampling(time_horizon=4.0,interval_length=4.0)
    traj_list=[Trajectory(cluster_points[i,None,:],traj_sampling) for i in range(len(cluster_points))]
    for idx, (token) in enumerate(tokens_to_evaluate):
        logger.info(
            f"Processing scenario {idx + 1} / {len(tokens_to_evaluate)} in thread_id={thread_id}, node_id={node_id}"
        )
        score_row: Dict[str, Any] = {"token": token, "valid": True}
        try:
            metric_cache_path = metric_cache_loader.metric_cache_paths[token]
            with lzma.open(metric_cache_path, "rb") as f:
                metric_cache: MetricCache = pickle.load(f)
            
            drivable_area_compliance = dac_score(
                metric_cache=metric_cache,
                model_trajectory=traj_list,
                future_sampling=traj_sampling_sim,
                simulator=simulator,
                scorer=scorer,
            )
            os.makedirs(cfg.dac_label_path,exist_ok=True)
            new_file_name=f"{token}.gz.npz"
            gz_path=Path(cfg.dac_label_path) / new_file_name
            np.savez_compressed(gz_path, array1=drivable_area_compliance)

        except Exception as e:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()
            score_row["valid"] = False

        pdm_results.append(score_row)
    return pdm_results

if __name__ == "__main__":
    main()
