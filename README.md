# Inverse MPM Soft-Body Simulation

This branch contains a Taichi-based 3D Material Point Method (MPM) soft-body simulator with robust inverse physics pipelines customized to estimate the **plasticity yield limit (`yield_min`)** from trajectory observables.

## Project Layout

- `Inversive/mpm_softbody_demo.py`: Interactive forward simulation demo.
- `Inversive/code/sim_config.py`: Shared simulation and training configuration.
- `Inversive/code/gen_target_data.py`: Generates the target trajectory dataset (ground truth).
- `Inversive/code/inverse_train.py`: Trains the NN + Finite-Difference inverse model and runs inference.
- `Inversive/code/optimize_yield_search.py`: Derivative-free global optimization baseline using Golden-Section search.
- `Inversive/code/replay_collision_compare.py`: Default Taichi GUI collision-scene comparison and video recorder.
- `Inversive/code/render_trajectory_compare.py`: Side-by-side particle rollout visualization.
- `Inversive/code/plot_results.py`: Generates presentation-ready convergence and trajectory error plots.
- `Inversive/code/mpm_sim.py`: Differentiable MPM physics kernels with Von-Mises/Neo-Hookean plasticity.
- `Inversive/code/observables.py`: Trajectory tracking features (mass center height, spatial covariance, deformation gradient) and multi-channel loss computation.
- `Inversive/code/nn_layers.py`: Lightweight Taichi-backed neural network layers and AdamW optimizer.

Generated data and model checkpoints are written to `Inversive/data/`, which is
ignored by Git.

## Setup

```powershell
pip install -r requirements.txt
```

## Run

Generate target data:

```powershell
cd Inversive
python code\gen_target_data.py
```

Target generation uses a deterministic high-drop initial particle cloud, runs a
fixed warm-up phase, then records the collision-rich trajectory segment. The
warm-up end state is saved as `x0/v0/C0/F0`, and training, inference, and the
baseline all start from that same state.

To change how much pre-collision motion is skipped:

```powershell
python code\gen_target_data.py --warmup_steps 170
```

For the current scene, `--warmup_steps` in the range `170` to `180` is
recommended.

To test another target material, regenerate target data with a chosen `E`, then
train/infer again:

```powershell
python code\gen_target_data.py --yield_min 0.93 --warmup_steps 170
python code\inverse_train.py --train --lr 3e-3
python code\inverse_train.py --infer
```

If `trace(s)` stays almost constant and `det(F_mean)` remains near `1.0`
through the recorded trajectory, the object has not meaningfully collided yet.
Increase `--warmup_steps` slightly within that range.

## Method A: NN+FD Inverse Model

This route predicts `yield_min` with a small Taichi-backed neural network. The physics
gradient `dL/dY` is estimated with finite differences, then backpropagated only
through the neural network. This avoids reverse-mode AD through the full MPM
rollout while keeping the neural inverse estimator.

Train the neural model:

```powershell
python code\inverse_train.py --train --lr 3e-3
```

Inference after training:

```powershell
python code\inverse_train.py --infer
```

Generate the default Taichi collision-scene comparison video:

```powershell
python code\replay_collision_compare.py --record
```

This requires `ffmpeg` for MP4 encoding:

```powershell
conda install -c conda-forge ffmpeg
```

Open the same replay interactively:

```powershell
python code\replay_collision_compare.py
```

Generate the optional non-scene particle-cloud comparison video:

```powershell
python code\render_trajectory_compare.py --format mp4
```

The optional non-scene renderer can still export frames when needed:

```powershell
python code\render_trajectory_compare.py --format frames
```

Fast smoke test:

```powershell
python code\inverse_train.py --quick_test --tiny
```

## Method B: Derivative-Free Baseline

This route does not use NN training or reverse-mode AD. It directly searches
for the `yield_min` value whose forward simulation minimizes the same observable loss.
Because the current problem has only one unknown parameter, this is the most
stable baseline.

Estimate yield_min with golden-section search:

```powershell
python code\optimize_yield_search.py
```

Generate result plots:

```powershell
python code\plot_results.py --method nn
python code\plot_results.py --method baseline
```

NN plots are written to `Inversive\data\plots\nn\`; baseline plots are written
to `Inversive\data\plots\baseline\`. Training writes
`Inversive\data\training_log.npz`; NN inference writes
`Inversive\data\predicted_trajectory.npz`; the baseline writes
`Inversive\data\yield_search_result.npz`. Side-by-side rollout videos are written
to `Inversive\data\renders\`.

Export Formatted Report Table:
Compile all per-epoch data, baseline convergence checkpoints, and rollout MSE errors into a structured spreadsheet
```powershell
python code\generate_csv.py
```


Interactive demo:

```powershell
cd ..
python mpm_softbody_demo.py
```

The scripts try CUDA first and fall back to CPU where supported. The interactive
demo uses Taichi's GUI window, so it is best run on a local machine with graphics
support.
