"""Generate a side-by-side matplotlib comparison chart from eval outputs.

Reads results from the report directory (produced by eval_compare.py) and
creates a multi-panel figure comparing all three inverse-physics methods.

Usage:
  python code/plot_comparison.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(PROJECT_ROOT, "report")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATA_DIR_AD = os.path.join(PROJECT_ROOT, "data_ad")
PLOT_DIR = os.path.join(REPORT_DIR, "plots")

# Method display names and colours
METHOD_NAMES = {
    "baseline": "Baseline\n(golden-section)",
    "nn_fd": "NN+FD\n(finite diff)",
    "nn_ad": "NN+AD\n(autodiff)",
}
METHOD_COLORS = {
    "baseline": "#dd7f20",   # orange
    "nn_fd": "#2458a6",      # blue
    "nn_ad": "#1f8a70",      # green
}
METHOD_MARKERS = {
    "baseline": "s",
    "nn_fd": "o",
    "nn_ad": "^",
}


def ensure_dirs():
    os.makedirs(PLOT_DIR, exist_ok=True)


def load_comparison_results():
    """Load the JSON summary from eval_compare.py."""
    import json
    json_path = os.path.join(REPORT_DIR, "comparison_summary.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    return None


def load_training_logs():
    """Load training logs for NN+FD and NN+AD."""
    logs = {}
    for method, data_dir in [("nn_fd", DATA_DIR), ("nn_ad", DATA_DIR_AD)]:
        path = os.path.join(data_dir, "training_log.npz")
        if os.path.exists(path):
            logs[method] = dict(np.load(path))
    return logs


def load_baseline_data():
    """Load baseline search results."""
    path = os.path.join(DATA_DIR, "E_search_result.npz")
    if os.path.exists(path):
        return dict(np.load(path))
    return None


def load_trajectory_data():
    """Load predicted trajectories for all methods."""
    trajectories = {}

    # NN+FD
    path_fd = os.path.join(DATA_DIR, "predicted_trajectory.npz")
    if os.path.exists(path_fd):
        trajectories["nn_fd"] = dict(np.load(path_fd))

    # NN+AD
    path_ad = os.path.join(DATA_DIR_AD, "predicted_trajectory.npz")
    if os.path.exists(path_ad):
        trajectories["nn_ad"] = dict(np.load(path_ad))

    # Baseline
    path_bl = os.path.join(DATA_DIR, "E_search_result.npz")
    if os.path.exists(path_bl):
        data = dict(np.load(path_bl))
        if "pred_h" in data:
            trajectories["baseline"] = {
                "h": data["pred_h"],
                "s": data["pred_s"],
                "F_mean": data["pred_F_mean"],
                "target_h": data["target_h"][:data["pred_h"].shape[0]],
                "target_s": data["target_s"][:data["pred_s"].shape[0]],
                "target_F_mean": data["target_F_mean"][:data["pred_F_mean"].shape[0]],
            }

    return trajectories


def plot_1_e_recovery(ax, results):
    """Bar chart: |E_pred - E_true| for each method."""
    if results is None:
        ax.text(0.5, 0.5, "No comparison data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        return

    methods = [r["method"] for r in results if r.get("success")
               and "E_error" in r]
    errors = [r["E_error"] for r in results if r.get("success")
              and "E_error" in r]
    colors = [METHOD_COLORS.get(m, "#888888") for m in methods]
    labels = [METHOD_NAMES.get(m, m) for m in methods]

    bars = ax.bar(range(len(methods)), errors, color=colors, edgecolor="white",
                  linewidth=1.2, width=0.55)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("|E_pred − E_true|", fontsize=11)
    ax.set_title("E Recovery Accuracy (lower is better)", fontsize=12,
                 fontweight="bold")

    # Annotate bars with values
    for bar, err in zip(bars, errors):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(errors) * 0.03,
                f"{err:.2f}", ha="center", va="bottom", fontsize=10,
                fontweight="bold")

    ax.grid(axis="y", alpha=0.28)
    ax.set_ylim(bottom=0)


def plot_2_loss_comparison(ax, results):
    """Bar chart: final loss for each method."""
    if results is None:
        ax.text(0.5, 0.5, "No comparison data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        return

    methods = [r["method"] for r in results if r.get("success")
               and "loss" in r]
    losses = [r["loss"] for r in results if r.get("success")
              and "loss" in r]
    colors = [METHOD_COLORS.get(m, "#888888") for m in methods]
    labels = [METHOD_NAMES.get(m, m) for m in methods]

    bars = ax.bar(range(len(methods)), losses, color=colors, edgecolor="white",
                  linewidth=1.2, width=0.55)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Weighted Loss", fontsize=11)
    ax.set_title("Final Loss (lower is better)", fontsize=12, fontweight="bold")
    ax.set_yscale("log")

    for bar, loss_val in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.5,
                f"{loss_val:.2e}", ha="center", va="bottom", fontsize=9,
                fontweight="bold")

    ax.grid(axis="y", alpha=0.28, which="both")


def plot_3_runtime_comparison(ax, results):
    """Bar chart: runtime for each method."""
    if results is None:
        ax.text(0.5, 0.5, "No comparison data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        return

    methods = [r["method"] for r in results if r.get("success")]
    times = [r.get("train_time", r.get("time", 0)) for r in results
             if r.get("success")]
    colors = [METHOD_COLORS.get(m, "#888888") for m in methods]
    labels = [METHOD_NAMES.get(m, m) for m in methods]

    bars = ax.bar(range(len(methods)), times, color=colors, edgecolor="white",
                  linewidth=1.2, width=0.55)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Runtime (seconds)", fontsize=11)
    ax.set_title("Total Runtime (lower is better)", fontsize=12,
                 fontweight="bold")

    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(times) * 0.03,
                f"{t:.1f}s", ha="center", va="bottom", fontsize=10,
                fontweight="bold")

    ax.grid(axis="y", alpha=0.28)


def plot_4_trajectory_errors(ax, results):
    """Bar chart: MSE in observables (h, s, F)."""
    if results is None:
        ax.text(0.5, 0.5, "No comparison data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        return

    methods = [r["method"] for r in results if r.get("success")
               and "mse_h" in r]
    if not methods:
        ax.text(0.5, 0.5, "No trajectory error data", ha="center",
                va="center", transform=ax.transAxes, fontsize=12, color="gray")
        return

    x = np.arange(len(methods))
    width = 0.25

    mse_h_vals = [r["mse_h"] for r in results if r.get("success")
                  and "mse_h" in r]
    mse_s_vals = [r["mse_s"] for r in results if r.get("success")
                  and "mse_s" in r]
    mse_f_vals = [r["mse_F"] for r in results if r.get("success")
                  and "mse_F" in r]

    bars_h = ax.bar(x - width, mse_h_vals, width, label="MSE(h)",
                    color="#2458a6", edgecolor="white", linewidth=0.8)
    bars_s = ax.bar(x, mse_s_vals, width, label="MSE(s)",
                    color="#c44536", edgecolor="white", linewidth=0.8)
    bars_f = ax.bar(x + width, mse_f_vals, width, label="MSE(F)",
                    color="#6a4c93", edgecolor="white", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_NAMES.get(m, m) for m in methods], fontsize=9)
    ax.set_ylabel("MSE (log scale)", fontsize=11)
    ax.set_title("Trajectory Observable Errors", fontsize=12, fontweight="bold")
    ax.set_yscale("log")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.28, which="both")


def plot_5_training_curves(fig, gs, training_logs):
    """NN training convergence curves (loss + E_pred)."""
    if not training_logs:
        return

    # Loss curve
    ax_loss = fig.add_subplot(gs[0])
    for method, log in training_logs.items():
        color = METHOD_COLORS.get(method, "#888888")
        label = METHOD_NAMES.get(method, method).replace("\n", " ")
        ax_loss.semilogy(log["epoch"], np.maximum(log["loss"], 1e-16),
                         color=color, linewidth=2, label=label,
                         marker=METHOD_MARKERS.get(method, "o"),
                         markersize=3, markevery=max(1, len(log["epoch"]) // 8))
    ax_loss.set_xlabel("Epoch", fontsize=10)
    ax_loss.set_ylabel("Loss (log scale)", fontsize=10)
    ax_loss.set_title("Training Loss Convergence", fontsize=11,
                      fontweight="bold")
    ax_loss.legend(fontsize=8)
    ax_loss.grid(True, which="both", alpha=0.28)

    # E prediction curve
    ax_E = fig.add_subplot(gs[1])
    for method, log in training_logs.items():
        color = METHOD_COLORS.get(method, "#888888")
        label = METHOD_NAMES.get(method, method).replace("\n", " ")
        ax_E.plot(log["epoch"], log["E_pred"], color=color, linewidth=2,
                  label=label, marker=METHOD_MARKERS.get(method, "o"),
                  markersize=3, markevery=max(1, len(log["epoch"]) // 8))
    if training_logs:
        first_log = next(iter(training_logs.values()))
        e_true = float(first_log["E_true"])
        ax_E.axhline(e_true, color="#222222", linestyle="--", linewidth=1.5,
                     label=f"True E = {e_true:.1f}")
    ax_E.set_xlabel("Epoch", fontsize=10)
    ax_E.set_ylabel("Predicted E", fontsize=10)
    ax_E.set_title("E Parameter Convergence", fontsize=11, fontweight="bold")
    ax_E.legend(fontsize=8)
    ax_E.grid(True, alpha=0.28)


def plot_6_height_trajectory(ax, trajectories):
    """Height trajectory: target vs each method's prediction."""
    if not trajectories:
        ax.text(0.5, 0.5, "No trajectory data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        return

    # Use first available target as reference
    target_h = None
    for method, data in trajectories.items():
        if "target_h" in data:
            target_h = data["target_h"]
            break

    if target_h is not None:
        steps = np.arange(len(target_h))
        ax.plot(steps, target_h, color="#222222", linewidth=2.5,
                label="Target", zorder=10)

    for method in ["baseline", "nn_fd", "nn_ad"]:
        if method not in trajectories:
            continue
        data = trajectories[method]
        pred_h = data["h"]
        steps = np.arange(len(pred_h))
        color = METHOD_COLORS.get(method, "#888888")
        label = METHOD_NAMES.get(method, method).replace("\n", " ")
        ax.plot(steps, pred_h, color=color, linewidth=1.8, linestyle="--",
                label=label)

    ax.set_xlabel("Timestep", fontsize=10)
    ax.set_ylabel("Mean Height", fontsize=10)
    ax.set_title("Predicted vs Target Height Trajectory", fontsize=11,
                 fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.28)


def plot_7_baseline_search(ax, baseline_data):
    """Baseline golden-section search landscape."""
    if baseline_data is None:
        ax.text(0.5, 0.5, "No baseline search data", ha="center",
                va="center", transform=ax.transAxes, fontsize=12, color="gray")
        return

    evals = baseline_data["evaluations"]
    e_true = float(baseline_data["E_true"])
    best_e = float(baseline_data["best_E"])

    order = np.argsort(evals[:, 0])
    ax.semilogy(evals[order, 0], np.maximum(evals[order, 1], 1e-16),
                marker="o", color=METHOD_COLORS["baseline"],
                linewidth=1.8, markersize=5)
    ax.axvline(e_true, label=f"True E = {e_true:.1f}",
               color="#c44536", linestyle="--", linewidth=1.8)
    ax.axvline(best_e, label=f"Best E = {best_e:.2f}",
               color="#1f8a70", linestyle=":", linewidth=2.0)
    ax.set_xlabel("Young's Modulus E", fontsize=10)
    ax.set_ylabel("Weighted Loss (log scale)", fontsize=10)
    ax.set_title("Baseline: Golden-Section Search Landscape", fontsize=11,
                 fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.28)


def build_single_comparison_figure(results, training_logs, trajectories,
                                   baseline_data):
    """Build one comprehensive comparison figure with all subplots."""
    fig = plt.figure(figsize=(18, 22))

    # Title area
    fig.suptitle("Inverse-Physics Method Comparison\n"
                 "NN+AD (Autodiff)  vs  NN+FD (Finite Diff)  vs  "
                 "Baseline (Golden-Section Search)",
                 fontsize=15, fontweight="bold", y=0.995)

    gs_master = fig.add_gridspec(4, 3, hspace=0.45, wspace=0.35,
                                 top=0.94, bottom=0.04,
                                 left=0.06, right=0.97)

    # Row 1: bar charts
    ax_recovery = fig.add_subplot(gs_master[0, 0])
    plot_1_e_recovery(ax_recovery, results)

    ax_loss = fig.add_subplot(gs_master[0, 1])
    plot_2_loss_comparison(ax_loss, results)

    ax_time = fig.add_subplot(gs_master[0, 2])
    plot_3_runtime_comparison(ax_time, results)

    # Row 2: trajectory errors + height trajectory
    ax_traj_err = fig.add_subplot(gs_master[1, 0])
    plot_4_trajectory_errors(ax_traj_err, results)

    ax_height = fig.add_subplot(gs_master[1, 1:])
    plot_6_height_trajectory(ax_height, trajectories)

    # Row 3: training curves + baseline search
    if training_logs:
        gs_train = gs_master[2, :2].subgridspec(2, 1, hspace=0.35)
        plot_5_training_curves(fig, gs_train, training_logs)

    if baseline_data is not None:
        ax_search = fig.add_subplot(gs_master[2, 2])
        plot_7_baseline_search(ax_search, baseline_data)
    elif training_logs:
        # Put a summary note
        ax_note = fig.add_subplot(gs_master[2, 2])
        ax_note.axis("off")
        summary_lines = ["Summary of results:"]
        if results:
            for r in results:
                if r.get("success"):
                    name = METHOD_NAMES.get(r["method"], r["method"]).replace(
                        "\n", " ")
                    e_err = r.get("E_error", "N/A")
                    summary_lines.append(
                        f"  {name}:  |E error| = {e_err:.4f}")
        ax_note.text(0.05, 0.95, "\n".join(summary_lines),
                     transform=ax_note.transAxes, fontsize=10,
                     verticalalignment="top", fontfamily="monospace")
        ax_note.set_title("Summary", fontsize=11, fontweight="bold")

    # Row 4: text summary
    ax_text = fig.add_subplot(gs_master[3, :])
    ax_text.axis("off")

    text_lines = ["Method Descriptions:"]
    text_lines.append(
        "  • Baseline — Golden-section search on the 1D loss landscape "
        "L(E). No gradient, no neural network."
    )
    text_lines.append(
        "  • NN+FD — A neural network predicts E from trajectory features. "
        "Physics gradient dL/dE is estimated via finite difference."
    )
    text_lines.append(
        "  • NN+AD — Same NN architecture, but dL/dE is computed via "
        "reverse-mode automatic differentiation through the full MPM simulation."
    )

    if results:
        text_lines.append("")
        text_lines.append("Key Metrics:")
        for r in results:
            if not r.get("success"):
                continue
            name = METHOD_NAMES.get(r["method"], r["method"]).replace(
                "\n", " ")
            e_err = r.get("E_error", float("nan"))
            loss_val = r.get("loss", float("nan"))
            t = r.get("train_time", r.get("time", 0))
            text_lines.append(
                f"  {name:25s}  |E error| = {e_err:8.4f}  "
                f"loss = {loss_val:12.6e}  time = {t:7.1f}s"
            )

    ax_text.text(0.02, 0.98, "\n".join(text_lines),
                 transform=ax_text.transAxes, fontsize=9.5,
                 verticalalignment="top", fontfamily="monospace")

    return fig


def main():
    ensure_dirs()

    print("Loading comparison results ...")
    results = load_comparison_results()
    training_logs = load_training_logs()
    baseline_data = load_baseline_data()
    trajectories = load_trajectory_data()

    if results is None:
        print("No comparison_summary.json found. Run eval_compare.py first.")
    else:
        print(f"Found {len(results)} method results: "
              f"{[r['method'] for r in results]}")
    if training_logs:
        print(f"Found training logs for: {list(training_logs.keys())}")
    if baseline_data is not None:
        print("Found baseline search data")
    if trajectories:
        print(f"Found trajectory data for: {list(trajectories.keys())}")

    # ---------- Comprehensive comparison figure ----------
    print("\nBuilding comprehensive comparison chart ...")
    fig = build_single_comparison_figure(
        results, training_logs, trajectories, baseline_data)

    comp_path = os.path.join(PLOT_DIR, "method_comparison.png")
    fig.savefig(comp_path, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved {comp_path}")

    # ---------- Additional individual plots ----------
    # E-error-only bar chart (for presentations)
    if results:
        fig2, ax2 = plt.subplots(figsize=(7, 4.5))
        plot_1_e_recovery(ax2, results)
        ax2.set_title("E Recovery Accuracy by Method", fontsize=13,
                      fontweight="bold")
        path2 = os.path.join(PLOT_DIR, "e_recovery_comparison.png")
        fig2.savefig(path2, dpi=200, bbox_inches="tight",
                     facecolor="white")
        plt.close(fig2)
        print(f"Saved {path2}")

    # Height trajectory overlay (for presentations)
    if trajectories:
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        plot_6_height_trajectory(ax3, trajectories)
        ax3.set_title("Height Trajectory: All Methods vs Target",
                      fontsize=13, fontweight="bold")
        path3 = os.path.join(PLOT_DIR, "height_trajectory_comparison.png")
        fig3.savefig(path3, dpi=200, bbox_inches="tight",
                     facecolor="white")
        plt.close(fig3)
        print(f"Saved {path3}")

    print(f"\nAll comparison plots saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
