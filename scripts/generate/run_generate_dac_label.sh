# export LD_LIBRARY_PATH="/usr/local/cuda/lib64"

SPLIT=trainval
METRIC_CACHE='' # metric cache in run_dataset_cache_trainval you set
VOC_PATH=$NAVSIM_DEVKIT_ROOT/data/cluster_points_8192_.npy
DAC_LABEL_PATH='' # add the dac label path you want to storage

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dac_score.py \
agent=goalflow_agent_traj \
experiment_name=dac_score \
scene_filter=navtrain \
split=$SPLIT \
metric_cache_path=$METRIC_CACHE \
agent.config.voc_path=$VOC_PATH \
dac_label_path=$DAC_LABEL_PATH