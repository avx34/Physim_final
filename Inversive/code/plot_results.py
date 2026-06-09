"""Create presentation-ready plots for the NN+Taichi-AD archive branch.

Examples:

    python code/plot_results.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PLOT_DIR = os.path.join(DATA_DIR, "plots", "nn_ad")


def ensure_plot_dir():
    os.makedirs(PLOT_DIR, exist_ok=True)


def savefig(name):
    path = os.path.join(PLOT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"Saved {path}")


def load_npz(name):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")
    return np.load(path)


def plot_training_curves():
    log_path = os.path.join(DATA_DIR, "training_log.npz")
    loss_path = os.path.join(DATA_DIR, "loss_history.npy")
    if not os.path.exists(log_path):
        if not os.path.exists(loss_path):
            print("Skip training plots: no training_log.npz or loss_history.npy")
            return
        loss = np.load(loss_path)
        epoch = np.arange(loss.shape[0])
        plt.figure(figsize=(7.2, 4.2))
        plt.semilogy(epoch, np.maximum(loss, 1e-16),
                     color="#2458a6", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss (log scale)")
        plt.title("NN+AD Training Loss")
        plt.grid(True, which="both", alpha=0.28)
        savefig("loss_curve.png")
        print("Only loss_history.npy was found; rerun training to get E and "
              "gradient diagnostic plots.")
        return

    log = load_npz("training_log.npz")
    epoch = log["epoch"]
    loss = log["loss"]
    e_pred = log["E_pred"]
    e_abs_error = log["E_abs_error"]
    max_grad = log["max_grad"]
    epoch_time = log["epoch_time"]
    e_true = float(log["E_true"])

    plt.figure(figsize=(7.2, 4.2))
    plt.semilogy(epoch, np.maximum(loss, 1e-16), color="#2458a6", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.title("NN+AD Training Loss")
    plt.grid(True, which="both", alpha=0.28)
    savefig("loss_curve.png")

    plt.figure(figsize=(7.2, 4.2))
    plt.plot(epoch, e_pred, label="Predicted E", color="#1f8a70", linewidth=2)
    plt.axhline(e_true, label=f"True E = {e_true:.1f}",
                color="#c44536", linestyle="--", linewidth=1.8)
    plt.xlabel("Epoch")
    plt.ylabel("Young's modulus E")
    plt.title("NN+AD Parameter Trajectory")
    plt.legend()
    plt.grid(True, alpha=0.28)
    savefig("E_prediction_curve.png")

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True)
    axes[0].semilogy(epoch, np.maximum(e_abs_error, 1e-8),
                     color="#c44536", linewidth=2)
    axes[0].set_ylabel("|E_pred - E_true|")
    axes[0].set_title("Parameter Error")
    axes[0].grid(True, which="both", alpha=0.28)
    axes[1].semilogy(epoch, np.maximum(max_grad, 1e-16),
                     color="#6a4c93", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("max |gradient|")
    axes[1].set_title("AD Gradient Diagnostic")
    axes[1].grid(True, which="both", alpha=0.28)
    savefig("training_diagnostics.png")

    plt.figure(figsize=(7.2, 4.2))
    plt.plot(epoch, epoch_time, color="#dd7f20", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Seconds")
    plt.title("Epoch Runtime")
    plt.grid(True, alpha=0.28)
    savefig("epoch_runtime.png")


def plot_inference_curves():
    pred_path = os.path.join(DATA_DIR, "predicted_trajectory.npz")
    target_path = os.path.join(DATA_DIR, "target_trajectory.npz")
    if not os.path.exists(pred_path) or not os.path.exists(target_path):
        print("Skip inference plots: run `python code\\inverse_train.py --infer` "
              "after training.")
        return

    pred = load_npz("predicted_trajectory.npz")
    target = load_npz("target_trajectory.npz")

    pred_h = pred["h"]
    pred_s = pred["s"]
    pred_F = pred["F_mean"]
    target_h = target["h"][:pred_h.shape[0]]
    target_s = target["s"][:pred_s.shape[0]]
    target_F = target["F_mean"][:pred_F.shape[0]]
    steps = np.arange(pred_h.shape[0])

    s_err = np.linalg.norm(pred_s - target_s, axis=(1, 2))
    f_err = np.linalg.norm(pred_F - target_F, axis=(1, 2))
    h_err = np.abs(pred_h - target_h)

    plt.figure(figsize=(7.2, 4.2))
    plt.plot(steps, target_h, label="Target", color="#222222", linewidth=2)
    plt.plot(steps, pred_h, label="Predicted", color="#1f8a70",
             linewidth=2, linestyle="--")
    plt.xlabel("Timestep")
    plt.ylabel("Mean height")
    plt.title("Target vs NN+AD Predicted Height")
    plt.legend()
    plt.grid(True, alpha=0.28)
    savefig("height_target_vs_predicted.png")

    plt.figure(figsize=(7.2, 4.2))
    plt.semilogy(steps, np.maximum(h_err, 1e-16), label="|h error|",
                 color="#2458a6", linewidth=2)
    plt.semilogy(steps, np.maximum(s_err, 1e-16), label="||s error||_F",
                 color="#c44536", linewidth=2)
    plt.semilogy(steps, np.maximum(f_err, 1e-16), label="||F_mean error||_F",
                 color="#6a4c93", linewidth=2)
    plt.xlabel("Timestep")
    plt.ylabel("Error (log scale)")
    plt.title("Observable Errors Over Time")
    plt.legend()
    plt.grid(True, which="both", alpha=0.28)
    savefig("observable_errors.png")

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    axes[0, 0].semilogy(steps, np.maximum(h_err, 1e-16),
                        color="#2458a6", linewidth=2)
    axes[0, 0].set_title("Height Error")
    axes[0, 0].set_xlabel("Timestep")
    axes[0, 0].set_ylabel("|h error|")

    axes[0, 1].semilogy(steps, np.maximum(s_err, 1e-16),
                        color="#c44536", linewidth=2)
    axes[0, 1].set_title("Covariance Error")
    axes[0, 1].set_xlabel("Timestep")
    axes[0, 1].set_ylabel("||s error||_F")

    axes[1, 0].semilogy(steps, np.maximum(f_err, 1e-16),
                        color="#6a4c93", linewidth=2)
    axes[1, 0].set_title("Mean Deformation Error")
    axes[1, 0].set_xlabel("Timestep")
    axes[1, 0].set_ylabel("||F_mean error||_F")

    axes[1, 1].plot(steps, target_h, label="Target",
                    color="#222222", linewidth=2)
    axes[1, 1].plot(steps, pred_h, label="Predicted",
                    color="#1f8a70", linewidth=2, linestyle="--")
    axes[1, 1].set_title("Height Trajectory")
    axes[1, 1].set_xlabel("Timestep")
    axes[1, 1].set_ylabel("Mean height")
    axes[1, 1].legend()

    for ax in axes.flat:
        ax.grid(True, which="both", alpha=0.28)
    savefig("inference_summary.png")


def plot_forward_scan():
    scan_path = os.path.join(DATA_DIR, "E_loss_scan.npy")
    if not os.path.exists(scan_path):
        print("Skip E-loss scan plot: run `python code\\scan_E_loss.py`.")
        return

    scan = np.load(scan_path)
    e_vals = scan[:, 0]
    losses = scan[:, 4] if scan.shape[1] >= 5 else scan[:, 1]
    order = np.argsort(e_vals)

    plt.figure(figsize=(7.2, 4.2))
    plt.semilogy(e_vals[order], np.maximum(losses[order], 1e-16),
                 marker="o", color="#2458a6", linewidth=1.8)
    plt.xlabel("Young's modulus E")
    plt.ylabel("Weighted loss (log scale)")
    plt.title("Forward E-Loss Scan")
    plt.grid(True, which="both", alpha=0.28)
    savefig("forward_E_loss_scan.png")


def main():
    ensure_plot_dir()
    plot_training_curves()
    plot_inference_curves()
    plot_forward_scan()
    print(f"All plots are in {PLOT_DIR}")


if __name__ == "__main__":
    main()
