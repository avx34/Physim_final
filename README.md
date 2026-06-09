# NN+AD Inverse MPM Archive

This branch archives the original neural-network inverse-simulation experiment:
a small Taichi-backed NN predicts Young's modulus `E`, and training attempts to
differentiate through the MPM rollout with Taichi reverse-mode AD.

This branch is meant for reproducing and diagnosing the earlier NN+AD approach.
Other inverse methods live outside this archive branch.

## Project Layout

- `Inversive/mpm_softbody_demo.py`: interactive forward simulation demo.
- `Inversive/code/sim_config.py`: shared simulation and training configuration.
- `Inversive/code/gen_target_data.py`: generates the target trajectory dataset.
- `Inversive/code/inverse_train.py`: trains and runs inference for NN+AD.
- `Inversive/code/check_E_gradient.py`: compares finite-difference and Taichi AD
  gradients for diagnostic purposes.
- `Inversive/code/scan_E_loss.py`: scans the true forward loss over candidate
  `E` values.
- `Inversive/code/plot_results.py`: creates NN+AD training, inference, and
  forward-scan plots.
- `Inversive/code/mpm_sim.py`: differentiable MPM kernels.
- `Inversive/code/observables.py`: trajectory features and loss computation.
- `Inversive/code/nn_layers.py`: lightweight Taichi NN layers and AdamW.

Generated data and model checkpoints are written to `Inversive/data/`, which is
ignored by Git.

## Setup

```powershell
cd "D:\PKU Personal\Course Projects\vcx\FinalProject\Inversive"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks virtualenv activation, run commands through
`.\.venv\Scripts\python.exe` directly.

## Run NN+AD

Generate target data first:

```powershell
python code\gen_target_data.py
```

Train the NN+AD model:

```powershell
python code\inverse_train.py --train --seg_len 1
```

You can also try a longer AD tape:

```powershell
python code\inverse_train.py --train --seg_len 2
```

Run inference after training:

```powershell
python code\inverse_train.py --infer
```

Create plots:

```powershell
python code\plot_results.py
```

Plots are written to `Inversive\data\plots\nn_ad\`. Training writes
`Inversive\data\training_log.npz`, and inference writes
`Inversive\data\predicted_trajectory.npz`.

## Diagnostics

Compare the forward finite-difference gradient with the segmented Taichi AD
gradient used by training:

```powershell
python code\check_E_gradient.py --E 425 --seg_len 1
python code\check_E_gradient.py --E 425 --seg_len 2
```

Scan the true forward loss landscape:

```powershell
python code\scan_E_loss.py
python code\plot_results.py
```

These diagnostics are the main reason this branch is preserved: they make it
easy to show where NN+AD behaves plausibly and where the Taichi AD gradient
becomes unreliable for this MPM rollout.

## Notes

By default, target generation, training, and inference use a deterministic
pseudo-random initial particle cloud. To reproduce the original stochastic
experiment, add `--random_init` to both target generation and training/inference
commands.

The scripts try CUDA first and fall back to CPU where supported. The interactive
demo uses Taichi's GUI window, so it is best run on a local machine with graphics
support:

```powershell
python mpm_softbody_demo.py
```
