# Inverse MPM Soft-Body Simulation

This branch contains a Taichi-based 3D MPM soft-body simulator and clean inverse
pipelines that estimate Young's modulus from trajectory observables.

## Project Layout

- `Inversive/mpm_softbody_demo.py`: interactive forward simulation demo.
- `Inversive/code/sim_config.py`: shared simulation and training configuration.
- `Inversive/code/gen_target_data.py`: generates the target trajectory dataset.
- `Inversive/code/inverse_train.py`: trains the NN+finite-difference inverse model and runs inference.
- `Inversive/code/optimize_E_search.py`: derivative-free inverse baseline.
- `Inversive/code/mpm_sim.py`: differentiable MPM kernels.
- `Inversive/code/observables.py`: trajectory features and loss computation.
- `Inversive/code/nn_layers.py`: lightweight Taichi NN layers and AdamW.

Generated data and model checkpoints are written to `Inversive/data/`, which is
ignored by Git.

## Setup

```powershell
pip install -r requirements.txt
```

If PowerShell blocks virtualenv activation, run the Python commands through
`.\.venv\Scripts\python.exe` directly.

## Run

Generate target data:

```powershell
cd Inversive
python code\gen_target_data.py
```

Target generation, training, and inference use the same deterministic
pseudo-random initial particle cloud. This removes per-epoch physical
initial-condition noise while keeping an irregular particle distribution.

## Method A: NN+FD Inverse Model

This route predicts `E` with a small Taichi-backed neural network. The physics
gradient `dL/dE` is estimated with finite differences, then backpropagated only
through the neural network. This avoids reverse-mode AD through the full MPM
rollout while keeping the neural inverse estimator.

Train the neural model:

```powershell
python code\inverse_train.py --train --lr 3e-4 --epochs 80
```

Inference after training:

```powershell
python code\inverse_train.py --infer
```

Fast smoke test:

```powershell
python code\inverse_train.py --quick_test --tiny
```

## Method B: Derivative-Free Baseline

This route does not use NN training or reverse-mode AD. It directly searches
for the `E` value whose forward simulation minimizes the same observable loss.
Because the current problem has only one unknown parameter, this is the most
stable baseline.

Estimate E with golden-section search:

```powershell
python code\optimize_E_search.py
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
`Inversive\data\E_search_result.npz`.

Interactive demo:

```powershell
python cd ..
python mpm_softbody_demo.py
```

The scripts try CUDA first and fall back to CPU where supported. The interactive
demo uses Taichi's GUI window, so it is best run on a local machine with graphics
support.
