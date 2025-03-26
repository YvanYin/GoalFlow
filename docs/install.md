# Preparation of GoalFlow Environment

Follow the steps below to set up the GoalFlow environment.

### 1. Clone the GoalFlow Repository

First, clone the repository and create a data directory:

```bash
git clone https://github.com/YvanYin/GoalFlow.git
cd GoalFlow
mkdir data
```
Make sure that the ``navsim_log_path`` and ``sensor_blobs_path`` in ``default_evaluation.yaml`` and ``default_metric_caching.yaml`` match your dataset path.

### 2. Install packages

Firstly, create conda environment with python 3.10 and intall the packages
```bash
conda create -n goalflow python=3.10
conda activate goalflow
pip install -r requirements.txt
pip install -e nuplan-devkit
pip install -e .
```

### 2. Prepare the Cache
In NAVSIM, it is recommended to store features and metrics as a cache to speed up training and evaluation.

### Step 1: Cache Features
Cache features to train the model and generate trajectories.
- If you only need to evaluate the model, you **only** need to run ``run_dataset_cache_test.sh``.
```bash
sh scripts/cache/run_dataset_cache_test.sh
sh scripts/cache/run_dataset_cache_trainval.sh
```

### Step 2: Cache Metrics
Cache metrics to evaluate trajectories. The test cache is used to generate DAC score labels for training the goal point module.
- If you only need to evaluate your model, you **only** need to run ``run_metric_caching_test.sh``.
```bash
sh scripts/cache/run_metric_caching_test.sh
sh scripts/cache/run_metric_caching_trainval.sh
```
### Step 2: Cache DAC score labels (Optional)
Cache DAC score labels. You need to specify ``METRIC_PATH`` obtained from ``run_metric_caching_trainval.sh``
- We also provide precomputed [DAC score labels](https://drive.google.com/drive/folders/1iWsPwpqM4WaUVVRZU3xIMPdOaJVB2Kub?usp=drive_link), which you can directly use.
```bash
sh scripts/generate/run_generate_dac_label.sh
```
