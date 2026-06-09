"""Render target and inverse-predicted particle trajectories side by side.

Run after:

    python code/gen_target_data.py
    python code/inverse_train.py --infer

Examples:

    python code/render_trajectory_compare.py --format mp4
    python code/render_trajectory_compare.py --format frames --stride 2
"""
import argparse
import os

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEFAULT_OUT_DIR = os.path.join(DATA_DIR, "renders")


parser = argparse.ArgumentParser()
parser.add_argument("--target", default=os.path.join(DATA_DIR,
                                                    "target_trajectory.npz"))
parser.add_argument("--pred", default=os.path.join(DATA_DIR,
                                                  "predicted_trajectory.npz"))
parser.add_argument("--output", default="trajectory_compare.mp4",
                    help="output file name; placed under data/renders by default")
parser.add_argument("--format", choices=("mp4", "gif", "frames"),
                    default="mp4")
parser.add_argument("--fps", type=int, default=12)
parser.add_argument("--stride", type=int, default=1,
                    help="render every Nth simulation frame")
parser.add_argument("--sample", type=int, default=1800,
                    help="maximum particles drawn per panel")
parser.add_argument("--dpi", type=int, default=160)
args = parser.parse_args()


def load_positions(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")
    data = np.load(path)
    if "x" not in data:
        raise KeyError(
            f"{label} file has no particle positions key `x`: {path}\n"
            "Rerun target generation and inference with the updated scripts.")
    return data["x"].astype(np.float32), data


def output_path():
    os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)
    if os.path.isabs(args.output):
        return args.output
    return os.path.join(DEFAULT_OUT_DIR, args.output)


def set_axes_equalish(ax, mins, maxs):
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[2] - radius, center[2] + radius)
    ax.set_zlim(max(0.0, center[1] - radius), center[1] + radius)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    ax.view_init(elev=18, azim=-62)


def scatter_positions(ax, positions, title, mins, maxs):
    colors = positions[:, 1]
    sc = ax.scatter(positions[:, 0], positions[:, 2], positions[:, 1],
                    c=colors, cmap="viridis", s=4, alpha=0.72,
                    vmin=mins[1], vmax=maxs[1], linewidths=0)
    ax.set_title(title)
    set_axes_equalish(ax, mins, maxs)
    return sc


def main():
    target_x, target_data = load_positions(args.target, "target")
    pred_x, pred_data = load_positions(args.pred, "prediction")
    n_frames = min(target_x.shape[0], pred_x.shape[0])
    frame_ids = np.arange(0, n_frames, max(args.stride, 1), dtype=np.int32)

    n_particles = min(target_x.shape[1], pred_x.shape[1])
    if args.sample > 0 and n_particles > args.sample:
        sample_ids = np.linspace(0, n_particles - 1, args.sample).astype(np.int32)
    else:
        sample_ids = np.arange(n_particles, dtype=np.int32)

    target_x = target_x[:n_frames, sample_ids]
    pred_x = pred_x[:n_frames, sample_ids]
    all_pos = np.concatenate([target_x.reshape(-1, 3),
                              pred_x.reshape(-1, 3)], axis=0)
    mins = all_pos.min(axis=0)
    maxs = all_pos.max(axis=0)

    e_true = float(target_data["E_true"]) if "E_true" in target_data else np.nan
    e_pred = float(pred_data["E_pred"]) if "E_pred" in pred_data else np.nan

    fig = plt.figure(figsize=(11.0, 5.2))
    ax_target = fig.add_subplot(1, 2, 1, projection="3d")
    ax_pred = fig.add_subplot(1, 2, 2, projection="3d")

    def draw(frame_index):
        step = int(frame_ids[frame_index])
        ax_target.cla()
        ax_pred.cla()
        scatter_positions(ax_target, target_x[step],
                          f"Target, E={e_true:.2f}", mins, maxs)
        scatter_positions(ax_pred, pred_x[step],
                          f"Inverse prediction, E={e_pred:.2f}", mins, maxs)
        fig.suptitle(f"MPM soft-body trajectory comparison | step {step}")
        return []

    if args.format == "frames":
        out_dir = output_path()
        if os.path.splitext(out_dir)[1]:
            out_dir = os.path.splitext(out_dir)[0] + "_frames"
        os.makedirs(out_dir, exist_ok=True)
        for i in range(len(frame_ids)):
            draw(i)
            path = os.path.join(out_dir, f"frame_{i:04d}.png")
            plt.tight_layout()
            fig.savefig(path, dpi=args.dpi)
        print(f"Saved {len(frame_ids)} frames to {out_dir}")
        return

    anim = FuncAnimation(fig, draw, frames=len(frame_ids), interval=1000 / args.fps)
    path = output_path()
    if args.format == "gif":
        if not path.lower().endswith(".gif"):
            path = os.path.splitext(path)[0] + ".gif"
        writer = PillowWriter(fps=args.fps)
    else:
        if not path.lower().endswith(".mp4"):
            path = os.path.splitext(path)[0] + ".mp4"
        writer = FFMpegWriter(fps=args.fps, bitrate=1800)

    anim.save(path, writer=writer, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
