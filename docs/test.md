# Evaluation

GoalFlow first generates trajectories for all test scenarios and then performs testing.  
**The Goal Point Construction Module is a pluggable component. We provide scores for the predicted goal points, which you can directly use in your own model.**

## Preparation for Evaluation

Download the following files from [Google Drive](https://drive.google.com/drive/folders/1iWsPwpqM4WaUVVRZU3xIMPdOaJVB2Kub?usp=drive_link) and place them in the `data` directory:

- `goalflow_traj_epoch=54-step=18260.ckpt` (perception module and trajectory decoder)
- `goal_point_scores.gz` (dac scores and distance scores predicted by goal point construction)
- `cluster_points_8192_.npy` (goal point vocabulary)
- `goalflow_navi_epoch=99-step=132500.ckpt` (optional, perception module and goal point construction module)

## Evaluating GoalFlow

1. First, generate trajectories using `run_generate_trajs.sh`. The generated trajectories will be stored in your log directory.
2. Then, run `run_goalflow_trajs.sh` with your evaluation metric path and the path to the generated trajectories.

```bash
sh scripts/generate/run_generate_trajs.sh
sh scripts/generate/run_goalflow_trajs.sh
```

## Evaluating the Goal Point Construction Module (Optional)
If needed, you can generate ``goal_point_scores`` using ``run_generate_navi.sh``. The generated navigation scores will be stored in your log directory.
```bash
sh scripts/generate/run_generate_navi.sh
```

