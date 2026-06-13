# Inverse MPM Soft-Body Simulation

This branch contains a Taichi-based 3D MPM soft-body simulator and clean inverse
pipelines that estimate material parameters from trajectory observables.

## Project Layout

- `Inversive/mpm_softbody_demo.py`: interactive forward simulation demo.
- `Inversive/code/sim_config.py`: shared simulation and training configuration.
- `Inversive/code/gen_target_data.py`: generates the target trajectory dataset.
- `Inversive/code/inverse_train.py`: trains the NN+finite-difference inverse model and runs inference.
- `Inversive/code/optimize_E_search.py`: derivative-free inverse baseline.
- `Inversive/code/replay_collision_compare.py`: default Taichi collision-scene comparison video.
- `Inversive/code/render_trajectory_compare.py`: side-by-side particle rollout visualization.
- `Inversive/code/mpm_sim.py`: differentiable MPM kernels.
- `Inversive/code/observables.py`: trajectory features and loss computation.
- `Inversive/code/nn_layers.py`: lightweight Taichi NN layers and AdamW.

Generated data and model checkpoints are written to `Inversive/data/`, which is
ignored by Git.

## Setup

```powershell
pip install -r requirements.txt
```

## Run

Generate the default noisy target data for the main experiment:

```powershell
cd Inversive
python code\gen_target_data.py --E 450 --nu 0.35 --warmup_steps 170 --noise_h 0.001 --noise_seed 0
```

Target generation uses a deterministic high-drop initial particle cloud, runs a
fixed warm-up phase, then records the collision-rich trajectory segment. The
warm-up end state is saved as `x0/v0/C0/F0`, and training, inference, and the
baseline all start from that same state. Noise is added only to saved target
observables, not to the forward simulator. The file still stores clean arrays
as `h_clean`, `s_clean`, and `F_mean_clean`, while training uses noisy `h`, `s`,
and `F_mean`.

Train the main NN+FD inverse model:

```powershell
python code\inverse_train.py --train --learn_param both
```

Run inference:

```powershell
python code\inverse_train.py --infer --learn_param both
```

Generate plots:

```powershell
python code\plot_results.py --method nn
```

For the current scene, `--warmup_steps` in the range `170` to `180` is
recommended. If `trace(s)` stays almost constant and `det(F_mean)` remains near
`1.0` through the recorded trajectory, the object has not meaningfully collided
yet; increase `--warmup_steps` slightly within that range.

The default inverse loss is the realistic external-observation setting:

```text
0.1 * h_loss + 1000 * s_loss
```

This avoids using `F_mean`, which is an internal simulator deformation
quantity, and downweights noisy height observations. Useful switches:

- `--learn_param E`, `--learn_param nu`, or `--learn_param both`.
- `--obs_mode external`, `full`, or `height`.
- `--obs_alpha_h`, `--obs_alpha_s`, and `--obs_alpha_F` for loss-weight ablations.
- `--noise_h`, `--noise_s`, and `--noise_F` for target-observation noise.

## Method A: NN+FD Inverse Model

This route predicts material parameters with a small Taichi-backed neural
network. The physics gradient is estimated with finite differences, then
backpropagated only through the neural network. This avoids reverse-mode AD
through the full MPM rollout while keeping the neural inverse estimator.

The default learning rate is `3e-3`, override it with `--lr` when
running ablations.

Fast smoke test:

```powershell
python code\inverse_train.py --quick_test --tiny
```

## Method B: Derivative-Free Baseline

This route does not use NN training or reverse-mode AD. It directly searches
for material parameters whose forward simulation minimizes the same observable
loss. The default `E` case uses the original 1D golden-section search; `nu` and
`both` use a coarse-to-fine grid search.

Estimate E with golden-section search:

```powershell
python code\optimize_E_search.py
```

Estimate both `E` and `nu` with the derivative-free baseline:

```powershell
python code\optimize_E_search.py --learn_param both --grid_size 7 --levels 3
```

## Analysis and Visualization

Generate result plots:

```powershell
python code\plot_results.py --method nn
python code\plot_results.py --method baseline
python code\plot_results.py --method all
```

`--method all` also creates a comparison figure with parameter error, weighted
loss, runtime, and the number of forward evaluations when both NN+FD and
baseline result files are available.

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

NN plots are written to `Inversive\data\plots\nn\`; baseline plots are written
to `Inversive\data\plots\baseline\`. Training writes
`Inversive\data\training_log.npz`; NN inference writes
`Inversive\data\predicted_trajectory.npz`; the baseline writes
`Inversive\data\baseline_search_result.npz` and keeps
`Inversive\data\E_search_result.npz` for default E-only compatibility.
Training and baseline result files record wall-clock runtime for lightweight
method comparisons.
Side-by-side rollout videos are written to `Inversive\data\renders\`.

Interactive demo:

```powershell
cd ..
python mpm_softbody_demo.py
```

The scripts try CUDA first and fall back to CPU where supported. The interactive
demo uses Taichi's GUI window, so it is best run on a local machine with graphics
support.
