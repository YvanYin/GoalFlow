# LD_LIBRARY_PATH="/usr/local/cuda/lib64"
# export CUDA_VISIBLE_DEVICES=3
# export HYDRA_FULL_ERROR=1

FEATURE_CACHE='' # set your feature_cache path
V99_PRETRAINED_PATH=$NAVSIM_DEVKIT_ROOT/data/depth_pretrained_v99-3jlw0p36-20210423_010520-model_final-remapped.pth
CHECKPOINT_PATH='' # add the checkpoint path you want load from
VOC_PATH=$NAVSIM_DEVKIT_ROOT/data/cluster_points_8192_.npy
ONLY_PERCEPTION=True
FREEZE_PERCEPTION=False # you can choose False and increase batch_size if the GPU are sufficient

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
agent=goalflow_agent_traj \
experiment_name=a_train_traj \
scene_filter=navtrain \
split=trainval \
cache_path=$FEATURE_CACHE \
trainer.params.max_epochs=100 \
agent.config.training=True \
agent.config.has_navi=True \
agent.config.start=True \
agent.config.freeze_perception=$FREEZE_PERCEPTION \
agent.config.only_perception=$ONLY_PERCEPTION \
agent.config.train_scale=0.1 \
agent.config.tf_d_model=1024 \
agent.config.trajectory_weight=0.0 \
agent.config.agent_class_weight=10.0 \
agent.config.agent_box_weight=1.0 \
agent.config.bev_semantic_weight=10.0 \
agent.config.agent_loss=True \
dataloader.params.batch_size=2 \
use_cache_without_dataset=True \
agent.config.v99_pretrained_path=$V99_PRETRAINED_PATH \
agent.checkpoint_path=$CHECKPOINT_PATH \
agent.config.voc_path=$VOC_PATH