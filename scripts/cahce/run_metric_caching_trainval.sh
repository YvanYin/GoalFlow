# export LD_LIBRARY_PATH="/usr/local/cuda/lib64"

SPLIT=trainval
SCNEE_FILTER=navtrain
CACHE_TO_SAVE='' #set your metric cache path to save

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
scene_filter=$SCNEE_FILTER \
split=$SPLIT \
cache.cache_path=$CACHE_TO_SAVE \
scene_filter.frame_interval=1 \
