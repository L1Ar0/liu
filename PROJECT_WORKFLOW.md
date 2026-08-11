# Geometry-Only Multi-View Primitive Pipeline

The basic-task pipeline uses only RGB-D geometry for instance segmentation,
classification and 6-D pose estimation. RGB values may be stored for visual
inspection, but they are never used to split instances or select a class.

```text
Initial eye-in-hand RGB-D view
  -> RANSAC table removal
  -> spatial DBSCAN
  -> horizontal support-plane split for real vertical stacks
  -> normal/curvature over-segmentation
  -> finite-primitive patch agglomeration
  -> oversized-cluster multi-model fitting and contact split
  -> coarse primitive class and 6-D pose
  -> uncertainty/elevation-aware second-view target selection
  -> safe oblique camera pose with IK
  -> second RGB-D view in Robot Base coordinates
  -> two-view point-cloud fusion
  -> complete geometry-only re-segmentation
  -> catalog-constrained pose refinement
  -> symmetry-aware pose output
  -> grasp accessibility hints
  -> Ground Truth evaluation
```

Supported classes are `cube`, `cuboid`, `cylinder`, `cone`, `sphere` and
`spheroid`. Symmetric, intrinsically unobservable rotations are evaluated with
the corresponding symmetry model.

Run a random planned scene with two views:

```powershell
.\.venv\Scripts\python.exe pipeline_runner.py `
  --mode planned `
  --planned-layout auto `
  --views 2 `
  --headless
```

Add `--seed 20260840` only when a failing scene must be reproduced. Use
`--views 1` for a single-view baseline comparison.

The planned layout and seed are both optional. This command chooses a layout
and seed automatically. Planned and physics modes use two views by default;
pass `--views 1` only for a single-view baseline:

```powershell
.\.venv\Scripts\python.exe pipeline_runner.py `
  --mode planned `
  --views 2 `
  --headless
```

For deterministic contact-scene regression, use one of `leaning`, `stack`,
`bridge`, `partial_support`, `side_contact`, `table_only`, or `mixed` through
`--planned-layout`.

Main outputs:

- `segmentation_output/view_01_object_cloud.ply`
- `segmentation_output/view_02_object_cloud.ply`
- `segmentation_output/fused_object_cloud.ply`
- `segmentation_output/segmentation_metadata.json`
- `recognition_output/coarse_recognition_results.json`
- `recognition_output/recognition_results.json`
- `evaluation_output/evaluation_summary.json`
- `random_scene_contacts.json`

`segmentation_metadata.json` records `uses_color_features: false`, camera
poses for both views, geometric split diagnostics and the fusion artifacts.
The second-view stage is active perception only. Continuous visual-servo
feedback is provided by the optional PBVS stage described below. SAM-6D and
Docker are not part of this pipeline.

## Geometry-only visual servo

The advanced-task implementation keeps RGB values out of target tracking. It
uses the final primitive prediction to select a safe topmost target, moves to a
coarse pregrasp pose, then repeatedly captures depth, tracks the target in a
local 3-D ROI, computes a PBVS pose error and applies a bounded incremental IK
correction. The controller stops when the target leaves the usable image area,
tracking confidence is lost, IK fails, or table clearance would be violated.

Run recognition followed by visual-servo alignment:

```powershell
.\.venv\Scripts\python.exe pipeline_runner.py `
  --mode planned `
  --planned-layout auto `
  --views 2 `
  --visual-servo `
  --headless
```

The safest topmost prediction is selected automatically. To reproduce a test
with a specific prediction, add `--servo-target-id 2`.

RG2 closure and lift verification are deliberately explicit because planned
layouts contain static shapes. Use a dynamic/physics scene for a meaningful
physical grasp test:

```powershell
.\.venv\Scripts\python.exe pipeline_runner.py `
  --mode physics `
  --views 2 `
  --visual-servo `
  --servo-execute-grasp `
  --headless
```

Use `--views 1` only as a single-view baseline. Physical piles can occlude
large parts of a primitive, so two geometry-only views are the normal grasp
configuration.

Visual-servo outputs:

- `visual_servo_output/visual_servo_summary.json`
- `visual_servo_output/visual_servo_trace.csv`

The summary records convergence, final pose error, target-loss state, grasp
verification and `uses_color_features: false`.

The default grasp orientation is `auto`: upright objects use a top-down frame,
while a geometrically fitted tilted object uses a surface-aligned frame capped
by the configured reachable tilt cone. Force one mode with
`--servo-grasp-orientation top_down` or `surface`.

After the close command is issued, grasp execution uses an explicit
CoppeliaSim connector by default. In the planned/static scene the default
connector path keeps dynamics paused and records the close command before
attaching the selected shape; this avoids an artificial fall of the iiwa while
the stock RG2 child script is stepping. The summary records the attached target
and connector handle, so this result is not confused with a pure-friction
grasp. Set `ROBOT_GRASP_CONNECTOR_KINEMATIC_ONLY=0` and use `--no-connector`
only when validating the gripper/contact dynamics itself.

The runner first normalizes CoppeliaSim to `STOPPED`, so a failed physics or
grasp attempt can normally be retried without manually pressing Stop. The
physics scene is left `PAUSED` during RGB-D processing, but its generated
workpieces remain dynamic and respondable. Pausing stabilizes the captured
geometry; it must not be implemented by changing the workpieces to static
bodies, because static bodies cannot be lifted by RG2.

If a Python run is force-terminated while CoppeliaSim is in stepping mode and
even `getSimulationState()` no longer responds, close and reopen CoppeliaSim
before retrying. A dead remote API server cannot be recovered by the pipeline
itself.

## Reproducible validation

Run the four geometry-only complex layouts with three fixed seeds (12 cases):

```powershell
.\.venv\Scripts\python.exe run_complex_scene_benchmark.py --keep-going
```

Artifacts are written to `complex_benchmark_output/`, including each pipeline
log, inference/evaluation JSON, `per_case_metrics.csv`, aggregate JSON and a
benchmark plot. A case passes only when all instances match, classification is
at least 99.9%, and both mean and worst position error are at most 10 mm.

Run the same-disturbance open/closed-loop PBVS comparison:

```powershell
.\.venv\Scripts\python.exe run_visual_servo_6d_comparison.py --keep-going
```

The default disturbance is `[+12,-8,+5] mm` plus xyz Euler
`[+6,+6,+12] deg`. Both runs regenerate the same layout/seed, use the same
depth-only target tracker and differ only in whether PBVS feedback is applied.

## Optional depth-to-6D PPO

PPO is an optional coarse-proposal experiment, not a replacement for primitive
recognition, collision checking, PBVS or IK. Install its separate dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-rl.txt
```

Collect a CoppeliaSim depth dataset and train the contextual-bandit policy:

```powershell
.\.venv\Scripts\python.exe collect_depth_grasp_dataset.py `
  --samples 80 `
  --layout leaning `
  --output depth_grasp_dataset.npz

.\.venv\Scripts\python.exe train_depth_grasp_rl.py `
  --dataset depth_grasp_dataset.npz `
  --episodes 10000
```

Run the best checkpoint on the current scene:

```powershell
.\.venv\Scripts\python.exe run_depth_grasp_policy.py `
  --checkpoint depth_grasp_rl_output\best_checkpoint.pt
```

The output is a depth-only coarse pixel/yaw/normal proposal. The normal geometry
pipeline and PBVS controller remain responsible for the final executable pose.
