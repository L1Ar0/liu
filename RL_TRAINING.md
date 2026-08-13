# Stage A: BC + SAC Training

The recommended Stage A workflow uses the existing geometric controller only
as a training-time teacher. The deployed policy still receives only the
table-relative depth/height image and robot proprioception.

## 1. Collect balanced teacher demonstrations

Keep CoppeliaSim open with the project scene loaded, then run:

```powershell
.\.venv\Scripts\python.exe collect_grasp_demonstrations.py `
  --episodes-per-position 20 `
  --max-steps 30 `
  --output end_to_end_grasp_rl_output\stage_a_demonstrations.npz
```

This collects the same number of trajectories for center, left, right,
Y-minus and Y-plus. Ground-truth target coordinates are used only inside the
teacher that creates this offline file.

## 2. Behavior-clone the SAC actor

```powershell
.\.venv\Scripts\python.exe pretrain_grasp_bc.py `
  --demonstrations end_to_end_grasp_rl_output\stage_a_demonstrations.npz `
  --epochs 30 `
  --batch-size 128 `
  --output end_to_end_grasp_rl_output\stage_a_bc_sac.zip
```

## 3. Fine-tune with SAC

```powershell
.\.venv\Scripts\python.exe train_depth_grasp_rl.py `
  --algorithm sac `
  --stage A `
  --timesteps 50000 `
  --max-steps 30 `
  --scene-mode separated `
  --execution-mode kinematic `
  --sim-steps-per-action 1 `
  --bc-model end_to_end_grasp_rl_output\stage_a_bc_sac.zip `
  --demonstrations end_to_end_grasp_rl_output\stage_a_demonstrations.npz `
  --buffer-size 25000 `
  --batch-size 128 `
  --eval-freq 2000 `
  --eval-episodes-per-position 2 `
  --save-replay-buffer `
  --output-dir end_to_end_grasp_rl_output\stage_A_sac
```

Important outputs:

- `best_model.zip`: highest fixed-seed canonical evaluation score.
- `sac_grasp_final.zip`: final update, which may not be the best policy.
- `evaluations.jsonl`: deterministic evaluation history by position.
- `monitor_*.csv.monitor.csv`: training episodes with canonical position ID.
- `sac_replay_buffer.pkl`: optional replay state for exact continuation.

## Resume training

```powershell
.\.venv\Scripts\python.exe train_depth_grasp_rl.py `
  --algorithm sac `
  --stage A `
  --timesteps 20000 `
  --resume end_to_end_grasp_rl_output\stage_A_sac\sac_grasp_final.zip `
  --replay-buffer end_to_end_grasp_rl_output\stage_A_sac\sac_replay_buffer.pkl `
  --output-dir end_to_end_grasp_rl_output\stage_A_sac_resume
```

## Evaluate or run the best model

```powershell
.\.venv\Scripts\python.exe run_depth_grasp_policy.py `
  --algorithm sac `
  --checkpoint end_to_end_grasp_rl_output\stage_A_sac\best_model.zip `
  --stage A `
  --episodes 10 `
  --max-steps 30 `
  --scene-mode separated `
  --execution-mode kinematic
```

The replay buffer stores depth images as `uint8` and restores them to `[0, 1]`
when sampling. With the default 25,000 transitions it uses about 0.8 GiB for
current and next depth frames instead of about 3.1 GiB as float32.
