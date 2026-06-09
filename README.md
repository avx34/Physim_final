# Inverse MPM Soft-Body Simulation

This branch contains a Taichi-based 3D MPM soft-body simulator and inverse
pipelines that estimate Young's modulus from trajectory observables.

## Project Layout

- `Inversive/mpm_softbody_demo.py`: interactive forward simulation demo.
- `Inversive/code/sim_config.py`: shared simulation and training configuration.
- `Inversive/code/gen_target_data.py`: generates the target trajectory dataset.
- `Inversive/code/inverse_train.py`: trains the inverse model and runs inference.
- `Inversive/code/optimize_E_search.py`: derivative-free inverse baseline.
- `Inversive/code/mpm_sim.py`: differentiable MPM kernels.
- `Inversive/code/observables.py`: trajectory features and loss computation.
- `Inversive/code/nn_layers.py`: lightweight Taichi NN layers and AdamW.

Generated data and model checkpoints are written to `Inversive/data/`, which is
ignored by Git.

## Setup

```powershell
cd "D:\PKU Personal\Course Projects\vcx\FinalProject"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r Inversive\requirements.txt
```

If PowerShell blocks virtualenv activation, run the Python commands through
`.\.venv\Scripts\python.exe` directly.

## Run

Generate target data:

```powershell
cd Inversive
python code\gen_target_data.py
```

By default, target generation, training, and inference use a deterministic
pseudo-random initial particle cloud. To reproduce the original stochastic
experiment, add `--random_init` to both target generation and training/inference
commands.

Fast smoke test for training:

```powershell
python code\inverse_train.py --quick_test --tiny
```

## Method A: Neural AD Experiment

This route predicts `E` with a small Taichi-backed neural network and attempts
to train it through Taichi reverse-mode AD. It is useful for diagnosing the
differentiable simulation path, but current gradient checks show that the
segmented AD gradient can be numerically unreliable for this MPM setup.

Train the neural model:

```powershell
python code\inverse_train.py --train --seg_len 2
```

For a gentler NN experiment, reduce the learning rate:

```powershell
python code\inverse_train.py --train --seg_len 2 --lr 3e-4
```

To keep the neural predictor but avoid reverse-mode AD through the MPM rollout,
train with finite-difference physics gradients:

```powershell
python code\inverse_train.py --train --fd_train --lr 3e-4 --epochs 80
```

Inference after training:

```powershell
python code\inverse_train.py --infer
```

Compare the true finite-difference gradient with the segmented Taichi AD
gradient used by training:

```powershell
python code\check_E_gradient.py --E 425 --seg_len 2
```

## Method B: Derivative-Free Baseline

This route does not use NN training or reverse-mode AD. It directly searches
for the `E` value whose forward simulation minimizes the same observable loss.
Because the current problem has only one unknown parameter, this is the most
stable baseline.

Scan the forward loss directly over candidate material parameters:

```powershell
python code\scan_E_loss.py
```

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
python mpm_softbody_demo.py
```

The scripts try CUDA first and fall back to CPU where supported. The interactive
demo uses Taichi's GUI window, so it is best run on a local machine with graphics
support.
