# LD_LIBRARY_PATH="/usr/local/cuda/lib64"
# export CUDA_VISIBLE_DEVICES=0,5 # 1,2,3,4
# export HYDRA_FULL_ERROR=1

FEATURE_CACHE='' # set your feature_cache path
VOC_PATH=$NAVSIM_DEVKIT_ROOT/data/cluster_points_8192_.npy
CHECKPOINT_PATH=$NAVSIM_DEVKIT_ROOT/data/goalflow_traj_epoch_54-step_18260.ckpt
GOAL_POINT_SCORES=$NAVSIM_DEVKIT_ROOT/data/goal_point_scores


python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_generate_trajs.py \
agent=goalflow_agent_traj \
experiment_name=a_test_release \
scene_filter=navtest \
split=test \
cache_path=$FEATURE_CACHE \
scene_filter.num_future_frames=10 \
dataloader.params.batch_size=4 \
use_cache_without_dataset=True \
agent.config.score_path=$GOAL_POINT_SCORES \
agent.config.voc_path=$VOC_PATH \
agent.config.generate='trajectory' \
agent.config.topk=15 \
agent.config.fusion=True \
agent.config.beta=0.0 \
agent.config.cond_threshold=1.0 \
agent.config.cond_weight=1.0 \
agent.config.training=False \
agent.config.has_navi=False \
agent.config.has_student_navi=True \
agent.config.start=True \
agent.config.cur_sampling=True \
agent.config.use_nearest=True \
agent.config.train_scale=0.1 \
agent.config.test_scale=0.1 \
agent.config.theta=4.5 \
agent.config.ep_score_weight=0.2 \
agent.config.ep_point_weight=0.5 \
agent.config.tf_d_model=1024 \
agent.config.infer_steps=5 \
agent.config.anchor_size=384 \
agent.checkpoint_path=$CHECKPOINT_PATH