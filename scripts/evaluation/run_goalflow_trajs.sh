# export LD_LIBRARY_PATH="/usr/local/cuda/lib64"
# export HYDRA_FULL_ERROR=1

SPLIT=test
METRIC_CACHE="" # set your metric path 
TRAJS_CACHE="" # set your trajectories path

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_trajs.py \
agent=goalflow_agent_traj \
agent.checkpoint_path=$CHECKPOINT \
experiment_name=a_test_release_result \
scene_filter=navtest \
split=$SPLIT \
metric_cache_path=$METRIC_CACHE \
trajs_cache_path=$TRAJS_CACHE \