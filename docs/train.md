# Training

All our experiments use 4 machines with 8 RTX 4090 GPUs. We recommend using a large batch size and a multi-machine training strategy.  
**If using a single machine with 8 RTX 4090 GPUs or a similar configuration, we suggest freezing the perception module during Step 2 and Step 3. This can achieve similar performance while reducing training time.**

## Step 1: Train the Perception Module
First, download the pretrained [V99 backbone](https://drive.google.com/drive/folders/1iWsPwpqM4WaUVVRZU3xIMPdOaJVB2Kub?usp=drive_link) to data. 
Then, train the perception module separately before training other modules.  
- Set `ONLY_PERCEPTION` to `True`.  
- Specify the path to the ``V99_PRETRAINED_PATH``.  
- The model output will be saved in the experiment log.

```bash
sh scripts/training/run_goalflow_training_perception.sh
```

## Step 2: Train the Trajectory Planning Module
- Set `ONLY_PERCEPTION` to `False`.
- To accelerate training, you can set `FREEZE_PERCEPTION` to True.(Optional)
- Specify the path to the perception model using ``CHECKPOINT_PATH`` and `FEATURE_PATH` obtrained from ``run_dataset_cache_trainval.sh``.
```bash
sh scripts/training/run_goalflow_training_traj.sh
```

## Step 3: Train the Goal Point Construction Module (Optional)
- To accelerate training, you can set `FREEZE_PERCEPTION` to True.(Optional)
- Specify the path to the imported model using `CHECKPOINT_PATH` and `FEATURE_PATH` obtrained from ``run_dataset_cache_trainval.sh``.
```bash
sh scripts/training/run_goalflow_training_navi.sh
```