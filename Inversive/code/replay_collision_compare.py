"""Replay target and inverse trajectories in a Taichi collision scene.

Unlike render_trajectory_compare.py, this uses Taichi's 3D scene renderer and
shows the same ground/stair collision geometry as the forward demo. The two
rollouts are placed side by side in one world: target on the left, prediction
on the right.

Run after regenerating target/inference data with particle positions:

    python code/gen_target_data.py
    python code/inverse_train.py --infer

Examples:

    python code/replay_collision_compare.py
    python code/replay_collision_compare.py --record --format mp4
"""
import argparse
import os
import shutil
import subprocess

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEFAULT_OUT_DIR = os.path.join(DATA_DIR, "renders", "collision_compare")


parser = argparse.ArgumentParser()
parser.add_argument("--target", default=os.path.join(DATA_DIR,
                                                    "target_trajectory.npz"))
parser.add_argument("--pred", default=os.path.join(DATA_DIR,
                                                  "predicted_trajectory.npz"))
parser.add_argument("--record", action="store_true",
                    help="save frames/video instead of only opening the viewer")
parser.add_argument("--format", choices=("frames", "mp4"), default="frames")
parser.add_argument("--output", default="collision_compare.mp4",
                    help="output mp4 name; frame mode writes a directory")
parser.add_argument("--fps", type=int, default=12)
parser.add_argument("--stride", type=int, default=1)
parser.add_argument("--sample", type=int, default=4096,
                    help="maximum particles drawn for each rollout")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--particle_radius", type=float, default=0.006)
parser.add_argument("--gap", type=float, default=1.65,
                    help="world-space distance between the two scene copies")
args = parser.parse_args()

import taichi as ti


def load_positions(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")
    data = np.load(path)
    if "x" not in data:
        raise KeyError(
            f"{label} file has no particle positions key `x`: {path}\n"
            "Rerun `python code\\gen_target_data.py` and "
            "`python code\\inverse_train.py --infer` with the updated scripts.")
    return data["x"].astype(np.float32), data


def get_box_mesh(center, extent):
    verts = []
    for i in [0, 1]:
        for j in [0, 1]:
            for k in [0, 1]:
                verts.append([
                    center[0] + (2 * i - 1) * extent[0],
                    center[1] + (2 * j - 1) * extent[1],
                    center[2] + (2 * k - 1) * extent[2],
                ])
    inds = [
        0, 1, 3, 0, 3, 2,
        4, 6, 7, 4, 7, 5,
        0, 4, 5, 0, 5, 1,
        2, 3, 7, 2, 7, 6,
        0, 2, 6, 0, 6, 4,
        1, 5, 7, 1, 7, 3,
    ]
    return np.array(verts, dtype=np.float32), np.array(inds, dtype=np.int32)


def make_collision_mesh(offset_x):
    boxes = [
        ([0.5 + offset_x, 0.05, 0.5], [1.0, 0.05, 1.0]),
        ([0.5 + offset_x, 0.15, 0.7], [0.3, 0.05, 0.1]),
        ([0.5 + offset_x, 0.25, 0.5], [0.3, 0.05, 0.1]),
    ]
    verts_list = []
    inds_list = []
    offset = 0
    for center, extent in boxes:
        verts, inds = get_box_mesh(center, extent)
        verts_list.append(verts)
        inds_list.append(inds + offset)
        offset += verts.shape[0]
    return np.concatenate(verts_list), np.concatenate(inds_list)


def make_dual_collision_mesh(left_offset, right_offset):
    left_v, left_i = make_collision_mesh(left_offset)
    right_v, right_i = make_collision_mesh(right_offset)
    verts = np.concatenate([left_v, right_v], axis=0)
    inds = np.concatenate([left_i, right_i + left_v.shape[0]], axis=0)
    return verts.astype(np.float32), inds.astype(np.int32)


def maybe_encode_mp4(frame_dir, output_path, fps):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("ffmpeg was not found. Kept PNG frames instead:")
        print(f"  {frame_dir}")
        return
    cmd = [
        ffmpeg, "-y",
        "-framerate", str(fps),
        "-i", os.path.join(frame_dir, "frame_%04d.png"),
        "-pix_fmt", "yuv420p",
        output_path,
    ]
    subprocess.run(cmd, check=True)
    print(f"Saved {output_path}")


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
    n_draw = target_x.shape[1]

    ti.init(arch=ti.gpu)

    target_pos = ti.Vector.field(3, dtype=ti.f32, shape=n_draw)
    pred_pos = ti.Vector.field(3, dtype=ti.f32, shape=n_draw)

    left_offset = -0.5 * args.gap
    right_offset = 0.5 * args.gap
    mesh_verts_np, mesh_inds_np = make_dual_collision_mesh(left_offset,
                                                           right_offset)
    mesh_verts = ti.Vector.field(3, dtype=ti.f32, shape=mesh_verts_np.shape[0])
    mesh_inds = ti.field(dtype=ti.i32, shape=mesh_inds_np.shape[0])
    mesh_verts.from_numpy(mesh_verts_np)
    mesh_inds.from_numpy(mesh_inds_np)

    window = ti.ui.Window("Target vs Inverse Collision Replay",
                          (args.width, args.height), vsync=False)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(0.5, 1.05, 3.15)
    camera.lookat(0.5, 0.18, 0.55)
    camera.fov(42)

    e_true = float(target_data["E_true"]) if "E_true" in target_data else np.nan
    e_pred = float(pred_data["E_pred"]) if "E_pred" in pred_data else np.nan

    out_dir = DEFAULT_OUT_DIR
    if args.record:
        os.makedirs(out_dir, exist_ok=True)

    def draw(frame_id):
        target_frame = target_x[frame_id].copy()
        pred_frame = pred_x[frame_id].copy()
        target_frame[:, 0] += left_offset
        pred_frame[:, 0] += right_offset
        target_pos.from_numpy(target_frame.astype(np.float32))
        pred_pos.from_numpy(pred_frame.astype(np.float32))

        scene.set_camera(camera)
        scene.mesh(mesh_verts, indices=mesh_inds, color=(0.38, 0.38, 0.43))
        scene.point_light(pos=(0.5, 1.7, 1.2), color=(1.0, 1.0, 1.0))
        scene.ambient_light((0.45, 0.45, 0.45))
        scene.particles(target_pos, radius=args.particle_radius,
                        color=(0.08, 0.80, 0.95))
        scene.particles(pred_pos, radius=args.particle_radius,
                        color=(1.00, 0.48, 0.14))
        canvas.scene(scene)

    print("\nTarget is cyan on the left; prediction is orange on the right.")
    print(f"Target E={e_true:.3f}, predicted E={e_pred:.3f}")
    print("Hold right mouse button in viewer mode to adjust the camera.")

    if args.record:
        for out_index, frame_id in enumerate(frame_ids):
            draw(int(frame_id))
            frame_path = os.path.join(out_dir, f"frame_{out_index:04d}.png")
            window.save_image(frame_path)
            window.show()
            print(f"Saved {frame_path}")
        if args.format == "mp4":
            output = args.output
            if not os.path.isabs(output):
                output = os.path.join(os.path.dirname(out_dir), output)
            maybe_encode_mp4(out_dir, output, args.fps)
        else:
            print(f"Saved frames to {out_dir}")
        return

    frame_cursor = 0
    while window.running:
        camera.track_user_inputs(window, movement_speed=0.035,
                                 hold_key=ti.ui.RMB)
        draw(int(frame_ids[frame_cursor]))
        window.show()
        frame_cursor = (frame_cursor + 1) % len(frame_ids)


if __name__ == "__main__":
    main()
