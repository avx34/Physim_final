"""Create presentation-ready plots from inverse-simulation outputs.

Examples:

    python code/plot_results.py --method nn
    python code/plot_results.py --method baseline
    python code/plot_results.py --method all
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PLOT_DIR = os.path.join(DATA_DIR, "plots")

parser = argparse.ArgumentParser()
parser.add_argument("--method", choices=("nn", "baseline", "all"),
                    default="all",
                    help="which plot group to generate")
args = parser.parse_args()


def ensure_plot_dir():
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(os.path.join(PLOT_DIR, "nn"), exist_ok=True)
    os.makedirs(os.path.join(PLOT_DIR, "baseline"), exist_ok=True)


def savefig(name, group):
    path = os.path.join(PLOT_DIR, group, name)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"Saved {path}")


def load_npz(name):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")
    return np.load(path)


def plot_baseline_search():
    result_path = os.path.join(DATA_DIR, "baseline_search_result.npz")
    if not os.path.exists(result_path):
        result_path = os.path.join(DATA_DIR, "E_search_result.npz")
    if not os.path.exists(result_path):
        print("Skip baseline plots: run `python code\\optimize_E_search.py`.")
        return

    result = np.load(result_path)
    evals = result["evaluations"]
    iters = result["iterations"]
    e_true = float(result["E_true"])
    nu_true = float(result["nu_true"]) if "nu_true" in result else 0.4
    best_e = float(result["best_E"])
    best_nu = float(result["best_nu"]) if "best_nu" in result else nu_true
    best_loss = float(result["best_loss"])
    learn_param = (str(result["learn_param"].item())
                   if "learn_param" in result else "E")
    obs_mode = str(result["obs_mode"].item()) if "obs_mode" in result else "full"
    obs_alpha_h = float(result["obs_alpha_h"]) if "obs_alpha_h" in result else 1.0
    obs_alpha_s = float(result["obs_alpha_s"]) if "obs_alpha_s" in result else None
    obs_alpha_F = float(result["obs_alpha_F"]) if "obs_alpha_F" in result else None
    obs_label = (obs_mode if obs_alpha_s is None else
                 f"{obs_mode}, a_h={obs_alpha_h:g}, "
                 f"a_s={obs_alpha_s:g}, a_F={obs_alpha_F:g}")
    legacy_1d = "learn_param" not in result and evals.shape[1] < 6
    best_e_col = 7 if legacy_1d else 5
    best_nu_col = 6
    best_loss_col = 8 if legacy_1d else 7
    width_col = 9 if legacy_1d else 8

    if evals.shape[1] >= 6:
        e_vals = evals[:, 0]
        nu_vals = evals[:, 1]
        losses = evals[:, 2]
    else:
        e_vals = evals[:, 0]
        nu_vals = np.full_like(e_vals, nu_true)
        losses = evals[:, 1]

    order = np.argsort(e_vals)
    plt.figure(figsize=(7.2, 4.2))
    plt.semilogy(e_vals[order], np.maximum(losses[order], 1e-16),
                 marker="o", color="#2458a6", linewidth=1.8)
    plt.axvline(e_true, label=f"True E = {e_true:.1f}",
                color="#c44536", linestyle="--", linewidth=1.8)
    plt.axvline(best_e, label=f"Best E = {best_e:.2f}",
                color="#1f8a70", linestyle=":", linewidth=2.0)
    plt.xlabel("Young's modulus E")
    plt.ylabel("Weighted loss (log scale)")
    plt.title(f"Derivative-Free Search: E-Loss Samples ({obs_label})")
    plt.legend()
    plt.grid(True, which="both", alpha=0.28)
    savefig("E_loss_samples.png", "baseline")

    if learn_param in ("nu", "both"):
        plt.figure(figsize=(7.2, 4.2))
        plt.scatter(nu_vals, np.maximum(losses, 1e-16), c=e_vals,
                    cmap="viridis", s=42, edgecolors="none")
        plt.yscale("log")
        plt.axvline(nu_true, label=f"True nu = {nu_true:.4f}",
                    color="#c44536", linestyle="--", linewidth=1.8)
        plt.axvline(best_nu, label=f"Best nu = {best_nu:.4f}",
                    color="#1f8a70", linestyle=":", linewidth=2.0)
        plt.xlabel("Poisson ratio nu")
        plt.ylabel("Weighted loss (log scale)")
        plt.title(f"Derivative-Free Search: nu-Loss Samples ({obs_label})")
        plt.colorbar(label="E sample")
        plt.legend()
        plt.grid(True, which="both", alpha=0.28)
        savefig("nu_loss_samples.png", "baseline")

    if learn_param == "both":
        plt.figure(figsize=(7.2, 5.4))
        sc = plt.scatter(e_vals, nu_vals, c=np.log10(np.maximum(losses, 1e-16)),
                         cmap="magma_r", s=54, edgecolors="none")
        plt.scatter([e_true], [nu_true], marker="x", s=90,
                    color="#c44536", label="True")
        plt.scatter([best_e], [best_nu], marker="o", s=70,
                    facecolors="none", edgecolors="#1f8a70",
                    linewidths=2, label="Best")
        plt.xlabel("Young's modulus E")
        plt.ylabel("Poisson ratio nu")
        plt.title(f"2D Baseline Search Samples ({obs_label})")
        plt.colorbar(sc, label="log10 weighted loss")
        plt.legend()
        plt.grid(True, alpha=0.28)
        savefig("E_nu_loss_samples.png", "baseline")

    plt.figure(figsize=(7.2, 4.2))
    plt.plot(iters[:, 0], iters[:, best_e_col], color="#1f8a70", linewidth=2,
             label="Best E so far")
    if learn_param == "E":
        plt.fill_between(iters[:, 0], iters[:, 1], iters[:, 2],
                         color="#2458a6", alpha=0.16, label="Search bracket")
    plt.axhline(e_true, label=f"True E = {e_true:.1f}",
                color="#c44536", linestyle="--", linewidth=1.8)
    plt.xlabel("Iteration")
    plt.ylabel("Young's modulus E")
    plt.title(f"Derivative-Free Search Convergence ({obs_label})")
    plt.legend()
    plt.grid(True, alpha=0.28)
    savefig("E_convergence.png", "baseline")

    if learn_param in ("nu", "both"):
        plt.figure(figsize=(7.2, 4.2))
        plt.plot(iters[:, 0], iters[:, best_nu_col], color="#6a4c93", linewidth=2,
                 label="Best nu so far")
        plt.axhline(nu_true, label=f"True nu = {nu_true:.4f}",
                    color="#c44536", linestyle="--", linewidth=1.8)
        plt.xlabel("Iteration")
        plt.ylabel("Poisson ratio nu")
        plt.title(f"Derivative-Free Search: nu Convergence ({obs_label})")
        plt.legend()
        plt.grid(True, alpha=0.28)
        savefig("nu_convergence.png", "baseline")

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True)
    axes[0].semilogy(iters[:, 0], np.maximum(iters[:, best_loss_col], 1e-16),
                     color="#6a4c93", linewidth=2)
    axes[0].set_ylabel("Best loss")
    axes[0].set_title(f"Baseline Loss and Bracket Width ({obs_label})")
    axes[0].grid(True, which="both", alpha=0.28)
    width = (iters[:, width_col] if legacy_1d
             else np.maximum(iters[:, 8], iters[:, 9]))
    axes[1].semilogy(iters[:, 0], np.maximum(width, 1e-12),
                     color="#dd7f20", linewidth=2)
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Search width")
    axes[1].grid(True, which="both", alpha=0.28)
    savefig("search_diagnostics.png", "baseline")

    pred_h = result["pred_h"]
    target_h = result["target_h"][:pred_h.shape[0]]
    steps = np.arange(pred_h.shape[0])
    pred_s = result["pred_s"]
    target_s = result["target_s"][:pred_s.shape[0]]
    pred_f = result["pred_F_mean"]
    target_f = result["target_F_mean"][:pred_f.shape[0]]
    s_err = np.linalg.norm(pred_s - target_s, axis=(1, 2))
    f_err = np.linalg.norm(pred_f - target_f, axis=(1, 2))
    h_err = np.abs(pred_h - target_h)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    axes[0, 0].plot(steps, target_h, label="Target",
                    color="#222222", linewidth=2)
    axes[0, 0].plot(steps, pred_h, label="Baseline",
                    color="#1f8a70", linewidth=2, linestyle="--")
    axes[0, 0].set_title("Height Trajectory")
    axes[0, 0].set_xlabel("Timestep")
    axes[0, 0].set_ylabel("Mean height")
    axes[0, 0].legend()
    axes[0, 1].semilogy(steps, np.maximum(h_err, 1e-16),
                        color="#2458a6", linewidth=2)
    axes[0, 1].set_title("Height Error")
    axes[0, 1].set_xlabel("Timestep")
    axes[0, 1].set_ylabel("|h error|")
    axes[1, 0].semilogy(steps, np.maximum(s_err, 1e-16),
                        color="#c44536", linewidth=2)
    axes[1, 0].set_title("Covariance Error")
    axes[1, 0].set_xlabel("Timestep")
    axes[1, 0].set_ylabel("||s error||_F")
    axes[1, 1].semilogy(steps, np.maximum(f_err, 1e-16),
                        color="#6a4c93", linewidth=2)
    axes[1, 1].set_title("Mean Deformation Error")
    axes[1, 1].set_xlabel("Timestep")
    axes[1, 1].set_ylabel("||F_mean error||_F")
    for ax in axes.flat:
        ax.grid(True, which="both", alpha=0.28)
    fig.suptitle(f"Baseline Inference Summary: E={best_e:.3f}, "
                 f"nu={best_nu:.4f}, loss={best_loss:.3e}")
    savefig("inference_summary.png", "baseline")


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
        plt.title("Training Loss")
        plt.grid(True, which="both", alpha=0.28)
        savefig("loss_curve.png", "nn")
        print("Only loss_history.npy was found; rerun training to get E and "
              "gradient diagnostic plots.")
        return

    log = load_npz("training_log.npz")
    epoch = log["epoch"]
    loss = log["loss"]
    if "param_pred" in log:
        learn_param = str(log["learn_param"].item())
        param_pred = log["param_pred"]
        param_abs_error = log["param_abs_error"]
        param_true = float(log["E_true"] if learn_param == "E"
                           else log["nu_true"])
    else:
        learn_param = "E"
        param_pred = log["E_pred"]
        param_abs_error = log["E_abs_error"]
        param_true = float(log["E_true"])
    max_grad = log["max_grad"]
    epoch_time = log["epoch_time"]
    e_pred = log["E_pred"] if "E_pred" in log else None
    nu_pred = log["nu_pred"] if "nu_pred" in log else None
    e_true = float(log["E_true"]) if "E_true" in log else None
    nu_true = float(log["nu_true"]) if "nu_true" in log else None
    lr = float(log["lr"]) if "lr" in log else None
    fd_eps_E = float(log["fd_eps_E"]) if "fd_eps_E" in log else None
    fd_eps_nu = float(log["fd_eps_nu"]) if "fd_eps_nu" in log else None
    obs_mode = str(log["obs_mode"].item()) if "obs_mode" in log else "full"
    obs_alpha_h = float(log["obs_alpha_h"]) if "obs_alpha_h" in log else 1.0
    obs_alpha_s = float(log["obs_alpha_s"]) if "obs_alpha_s" in log else None
    obs_alpha_F = float(log["obs_alpha_F"]) if "obs_alpha_F" in log else None
    obs_label = (obs_mode if obs_alpha_s is None else
                 f"{obs_mode}, a_h={obs_alpha_h:g}, "
                 f"a_s={obs_alpha_s:g}, a_F={obs_alpha_F:g}")
    if lr is None:
        run_label = f"learn_param={learn_param}, obs={obs_label}"
    elif learn_param == "both":
        run_label = (f"learn_param=both, obs={obs_label}, lr={lr:g}, "
                     f"eps_E={fd_eps_E:g}, eps_nu={fd_eps_nu:g}")
    else:
        eps = fd_eps_E if learn_param == "E" else fd_eps_nu
        run_label = (f"learn_param={learn_param}, obs={obs_label}, "
                     f"lr={lr:g}, eps={eps:g}")

    plt.figure(figsize=(7.2, 4.2))
    plt.semilogy(epoch, np.maximum(loss, 1e-16), color="#2458a6", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.title(f"Training Loss ({run_label})")
    plt.grid(True, which="both", alpha=0.28)
    savefig("loss_curve.png", "nn")

    if learn_param == "both" and e_pred is not None and nu_pred is not None:
        fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.2), sharex=True)
        axes[0].plot(epoch, e_pred, label="Predicted E",
                     color="#1f8a70", linewidth=2)
        axes[0].axhline(e_true, label=f"True E = {e_true:.4g}",
                        color="#c44536", linestyle="--", linewidth=1.8)
        axes[0].set_ylabel("E")
        axes[0].set_title(f"Material Parameter Convergence ({run_label})")
        axes[0].legend()
        axes[0].grid(True, alpha=0.28)
        axes[1].plot(epoch, nu_pred, label="Predicted nu",
                     color="#6a4c93", linewidth=2)
        axes[1].axhline(nu_true, label=f"True nu = {nu_true:.4g}",
                        color="#c44536", linestyle="--", linewidth=1.8)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("nu")
        axes[1].legend()
        axes[1].grid(True, alpha=0.28)
        savefig("parameter_prediction_curve.png", "nn")

        fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.5), sharex=True)
        axes[0].semilogy(epoch, np.maximum(np.abs(e_pred - e_true), 1e-8),
                         color="#c44536", linewidth=2)
        axes[0].set_ylabel("|E error|")
        axes[0].set_title(f"Parameter Errors and Gradient Diagnostic ({run_label})")
        axes[0].grid(True, which="both", alpha=0.28)
        axes[1].semilogy(epoch, np.maximum(np.abs(nu_pred - nu_true), 1e-8),
                         color="#1f8a70", linewidth=2)
        axes[1].set_ylabel("|nu error|")
        axes[1].grid(True, which="both", alpha=0.28)
        axes[2].plot(epoch, max_grad, color="#6a4c93", linewidth=2)
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("max |gradient|")
        axes[2].grid(True, alpha=0.28)
        savefig("training_diagnostics.png", "nn")
    else:
        plt.figure(figsize=(7.2, 4.2))
        plt.plot(epoch, param_pred, label=f"Predicted {learn_param}",
                 color="#1f8a70", linewidth=2)
        plt.axhline(param_true, label=f"True {learn_param} = {param_true:.4g}",
                    color="#c44536", linestyle="--", linewidth=1.8)
        plt.xlabel("Epoch")
        plt.ylabel(learn_param)
        plt.title(f"Material Parameter Convergence ({run_label})")
        plt.legend()
        plt.grid(True, alpha=0.28)
        savefig("parameter_prediction_curve.png", "nn")

        fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True)
        axes[0].semilogy(epoch, np.maximum(param_abs_error, 1e-8),
                         color="#c44536", linewidth=2)
        axes[0].set_ylabel(f"|{learn_param}_pred - {learn_param}_true|")
        axes[0].set_title(f"Parameter Error ({run_label})")
        axes[0].grid(True, which="both", alpha=0.28)
        axes[1].plot(epoch, max_grad, color="#6a4c93", linewidth=2)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("max |gradient|")
        axes[1].set_title("Gradient Diagnostic")
        axes[1].grid(True, alpha=0.28)
        savefig("training_diagnostics.png", "nn")

    plt.figure(figsize=(7.2, 4.2))
    plt.plot(epoch, epoch_time, color="#dd7f20", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Seconds")
    plt.title(f"Epoch Runtime ({run_label})")
    plt.grid(True, alpha=0.28)
    savefig("epoch_runtime.png", "nn")


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
    target_h = (pred["target_h"][:pred_h.shape[0]]
                if "target_h" in pred else target["h"][:pred_h.shape[0]])
    target_s = (pred["target_s"][:pred_s.shape[0]]
                if "target_s" in pred else target["s"][:pred_s.shape[0]])
    target_F = (pred["target_F_mean"][:pred_F.shape[0]]
                if "target_F_mean" in pred else target["F_mean"][:pred_F.shape[0]])
    clean_h = (target["h_clean"][:pred_h.shape[0]]
               if "h_clean" in target else None)
    clean_s = (target["s_clean"][:pred_s.shape[0]]
               if "s_clean" in target else None)
    clean_F = (target["F_mean_clean"][:pred_F.shape[0]]
               if "F_mean_clean" in target else None)
    steps = np.arange(pred_h.shape[0])

    s_err = np.linalg.norm(pred_s - target_s, axis=(1, 2))
    f_err = np.linalg.norm(pred_F - target_F, axis=(1, 2))
    h_err = np.abs(pred_h - target_h)

    plt.figure(figsize=(7.2, 4.2))
    if clean_h is not None:
        plt.plot(steps, clean_h, label="Clean target",
                 color="#777777", linewidth=1.8)
        plt.plot(steps, target_h, label="Noisy target",
                 color="#222222", linewidth=2, alpha=0.78)
    else:
        plt.plot(steps, target_h, label="Target", color="#222222", linewidth=2)
    plt.plot(steps, pred_h, label="Predicted", color="#1f8a70",
             linewidth=2, linestyle="--")
    plt.xlabel("Timestep")
    plt.ylabel("Mean height")
    plt.title("Target vs Predicted Height Trajectory")
    plt.legend()
    plt.grid(True, alpha=0.28)
    savefig("height_target_vs_predicted.png", "nn")

    if clean_h is not None and clean_s is not None and clean_F is not None:
        noise_h = np.abs(target_h - clean_h)
        noise_s = np.linalg.norm(target_s - clean_s, axis=(1, 2))
        noise_F = np.linalg.norm(target_F - clean_F, axis=(1, 2))
        plt.figure(figsize=(7.2, 4.2))
        plt.semilogy(steps, np.maximum(noise_h, 1e-16), label="|h noise|",
                     color="#2458a6", linewidth=2)
        plt.semilogy(steps, np.maximum(noise_s, 1e-16),
                     label="||s noise||_F", color="#c44536", linewidth=2)
        plt.semilogy(steps, np.maximum(noise_F, 1e-16),
                     label="||F_mean noise||_F", color="#6a4c93", linewidth=2)
        plt.xlabel("Timestep")
        plt.ylabel("Noise magnitude (log scale)")
        plt.title("Injected Target Observation Noise")
        plt.legend()
        plt.grid(True, which="both", alpha=0.28)
        savefig("target_noise_profile.png", "nn")

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
    savefig("observable_errors.png", "nn")

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

    if clean_h is not None:
        axes[1, 1].plot(steps, clean_h, label="Clean target",
                        color="#777777", linewidth=1.8)
        axes[1, 1].plot(steps, target_h, label="Noisy target",
                        color="#222222", linewidth=2, alpha=0.78)
    else:
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
    savefig("inference_summary.png", "nn")


def plot_method_comparison():
    pred_path = os.path.join(DATA_DIR, "predicted_trajectory.npz")
    baseline_path = os.path.join(DATA_DIR, "baseline_search_result.npz")
    if not os.path.exists(pred_path) or not os.path.exists(baseline_path):
        print("Skip method comparison: need NN inference and baseline result.")
        return

    nn = load_npz("predicted_trajectory.npz")
    baseline = load_npz("baseline_search_result.npz")
    train_log = (load_npz("training_log.npz")
                 if os.path.exists(os.path.join(DATA_DIR, "training_log.npz"))
                 else None)
    e_true = float(nn["E_true"])
    nu_true = float(nn["nu_true"])
    nn_e = float(nn["E_pred"])
    nn_nu = float(nn["nu_pred"])
    base_e = float(baseline["best_E"])
    base_nu = float(baseline["best_nu"])
    nn_loss = (float(nn["weighted_loss"]) if "weighted_loss" in nn
               else float(nn["mse_h"])
               + 10.0 * float(nn["mse_s"])
               + 5.0 * float(nn["mse_F"]))
    base_loss = float(baseline["best_loss"])
    nn_obs = str(nn["obs_mode"].item()) if "obs_mode" in nn else "full"
    base_obs = (str(baseline["obs_mode"].item())
                if "obs_mode" in baseline else "full")
    if train_log is not None:
        nn_runtime = (float(train_log["train_runtime_sec"])
                      if "train_runtime_sec" in train_log
                      else float(np.sum(train_log["epoch_time"])))
        nn_evals = (int(train_log["n_forward_evals"])
                    if "n_forward_evals" in train_log else None)
    else:
        nn_runtime = np.nan
        nn_evals = None
    base_runtime = (float(baseline["runtime_sec"])
                    if "runtime_sec" in baseline else
                    float(np.sum(baseline["iterations"][:, -1])))
    base_evals = (int(baseline["n_forward_evals"])
                  if "n_forward_evals" in baseline else None)

    methods = ["NN+FD", "Baseline"]
    e_errors = [abs(nn_e - e_true), abs(base_e - e_true)]
    nu_errors = [abs(nn_nu - nu_true), abs(base_nu - nu_true)]
    losses = [nn_loss, base_loss]
    runtimes = [nn_runtime, base_runtime]

    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.8))
    axes[0].bar(methods, e_errors, color=["#1f8a70", "#2458a6"])
    axes[0].set_title("E Error")
    axes[0].set_ylabel("|E_pred - E_true|")
    axes[1].bar(methods, nu_errors, color=["#1f8a70", "#2458a6"])
    axes[1].set_title("nu Error")
    axes[1].set_ylabel("|nu_pred - nu_true|")
    axes[2].bar(methods, np.maximum(losses, 1e-16),
                color=["#1f8a70", "#2458a6"])
    axes[2].set_yscale("log")
    axes[2].set_title("Weighted Observable Loss")
    axes[2].set_ylabel("Loss")
    bars = axes[3].bar(methods, runtimes, color=["#1f8a70", "#2458a6"])
    axes[3].set_title("Runtime")
    axes[3].set_ylabel("Seconds")
    for bar, evals in zip(bars, [nn_evals, base_evals]):
        if evals is not None:
            axes[3].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height(), f"{evals} evals",
                         ha="center", va="bottom", fontsize=8)
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.28)
    fig.suptitle(f"NN+FD vs Derivative-Free Baseline "
                 f"(NN obs={nn_obs}, baseline obs={base_obs})")
    savefig("method_comparison.png", "comparison")


def main():
    ensure_plot_dir()
    if args.method in ("nn", "all"):
        plot_training_curves()
        plot_inference_curves()
    if args.method in ("baseline", "all"):
        plot_baseline_search()
    if args.method == "all":
        os.makedirs(os.path.join(PLOT_DIR, "comparison"), exist_ok=True)
        plot_method_comparison()
    print(f"All plots are in {PLOT_DIR}")


if __name__ == "__main__":
    main()
