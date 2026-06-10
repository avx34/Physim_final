"""Record a Taichi collision-scene replay of target vs prediction.

This is the default presentation video: it replays the saved particle
trajectories in the same ground/stair scene used by the simulator. Target is
shown on the left, inverse prediction on the right.

Run after:

    python code/gen_target_data.py
    python code/inverse_train.py --infer

Examples:

    python code/replay_collision_compare.py
    python code/replay_collision_compare.py --record
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
                    help="record an MP4 video instead of opening the viewer")
parser.add_argument("--output", default="collision_compare.mp4",
                    help="mp4 output name; placed under data/renders by default")
parser.add_argument("--fps", type=int, default=12)
parser.add_argument("--stride", type=int, default=1)
parser.add_argument("--sample", type=int, default=8192,
                    help="maximum particles drawn for each rollout")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--particle_radius", type=float, default=0.008)
parser.add_argument("--gap", type=float, default=1.65,
                    help="world-space distance between the two scene copies")
parser.add_argument("--mesh_y_offset", type=float, default=0.0,
                    help="visual-only collision mesh offset")
parser.add_argument("--no_mesh", action="store_true",
                    help="hide collision mesh and render only particles")
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


def make_collision_mesh(offset_x, offset_y):
    boxes = [
        ([0.5 + offset_x, 0.05 + offset_y, 0.5], [1.0, 0.05, 1.0]),
        ([0.5 + offset_x, 0.15 + offset_y, 0.7], [0.3, 0.05, 0.1]),
        ([0.5 + offset_x, 0.25 + offset_y, 0.5], [0.3, 0.05, 0.1]),
    ]
    verts_list = []
    inds_list = []
    index_offset = 0
    for center, extent in boxes:
        verts, inds = get_box_mesh(center, extent)
        verts_list.append(verts)
        inds_list.append(inds + index_offset)
        index_offset += verts.shape[0]
    return np.concatenate(verts_list), np.concatenate(inds_list)


def make_dual_collision_mesh(left_offset, right_offset, offset_y):
    left_v, left_i = make_collision_mesh(left_offset, offset_y)
    right_v, right_i = make_collision_mesh(right_offset, offset_y)
    verts = np.concatenate([left_v, right_v], axis=0)
    inds = np.concatenate([left_i, right_i + left_v.shape[0]], axis=0)
    return verts.astype(np.float32), inds.astype(np.int32)


def init_taichi():
    for name, arch in (("cuda", ti.cuda), ("cpu", ti.cpu)):
        try:
            ti.init(arch=arch)
            print(f"[ARCH] Using {name} backend")
            return
        except Exception as exc:
            print(f"[ARCH] {name} not available: {exc}")
            try:
                ti.reset()
            except Exception:
                pass
    raise RuntimeError("No Taichi backend available")


def require_ffmpeg():
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg was not found. Install it before recording video, e.g. "
            "`conda install -c conda-forge ffmpeg`."
        )
    return ffmpeg


def encode_mp4(frame_dir, output_path, fps, ffmpeg):
    cmd = [
        ffmpeg, "-y",
        "-framerate", str(fps),
        "-i", os.path.join(frame_dir, "frame_%04d.png"),
        "-pix_fmt", "yuv420p",
        output_path,
    ]
    subprocess.run(cmd, check=True)
    print(f"Saved {output_path}")


def output_mp4_path():
    os.makedirs(os.path.dirname(DEFAULT_OUT_DIR), exist_ok=True)
    if os.path.isabs(args.output):
        return args.output
    return os.path.join(os.path.dirname(DEFAULT_OUT_DIR), args.output)


def clear_old_frames(frame_dir):
    if not os.path.isdir(frame_dir):
        return
    for name in os.listdir(frame_dir):
        if name.startswith("frame_") and name.endswith(".png"):
            os.remove(os.path.join(frame_dir, name))


def remove_frame_dir(frame_dir):
    if os.path.isdir(frame_dir):
        shutil.rmtree(frame_dir)


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

    raw_min = np.minimum(target_x.reshape(-1, 3).min(axis=0),
                         pred_x.reshape(-1, 3).min(axis=0))
    raw_max = np.maximum(target_x.reshape(-1, 3).max(axis=0),
                         pred_x.reshape(-1, 3).max(axis=0))

    init_taichi()

    target_pos = ti.Vector.field(3, dtype=ti.f32, shape=n_draw)
    pred_pos = ti.Vector.field(3, dtype=ti.f32, shape=n_draw)

    left_offset = -0.5 * args.gap
    right_offset = 0.5 * args.gap
    mesh_verts_np, mesh_inds_np = make_dual_collision_mesh(
        left_offset, right_offset, args.mesh_y_offset)
    mesh_verts = ti.Vector.field(3, dtype=ti.f32, shape=mesh_verts_np.shape[0])
    mesh_inds = ti.field(dtype=ti.i32, shape=mesh_inds_np.shape[0])
    mesh_verts.from_numpy(mesh_verts_np)
    mesh_inds.from_numpy(mesh_inds_np)

    window = ti.ui.Window("Target vs Inverse Collision Replay",
                          (args.width, args.height), vsync=False)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(0.5, 1.05, 3.0)
    camera.lookat(0.5, 0.28, 0.55)
    camera.fov(45)

    e_true = float(target_data["E_true"]) if "E_true" in target_data else np.nan
    e_pred = float(pred_data["E_pred"]) if "E_pred" in pred_data else np.nan
    nu_true = float(target_data["nu_true"]) if "nu_true" in target_data else np.nan
    nu_pred = float(pred_data["nu_pred"]) if "nu_pred" in pred_data else nu_true
    warmup = int(target_data["warmup_steps"]) if "warmup_steps" in target_data else -1

    def draw(frame_id):
        target_frame = target_x[frame_id].copy()
        pred_frame = pred_x[frame_id].copy()
        target_frame[:, 0] += left_offset
        pred_frame[:, 0] += right_offset
        target_pos.from_numpy(target_frame.astype(np.float32))
        pred_pos.from_numpy(pred_frame.astype(np.float32))

        scene.set_camera(camera)
        if not args.no_mesh:
            scene.mesh(mesh_verts, indices=mesh_inds, color=(0.36, 0.36, 0.40))
        scene.point_light(pos=(0.5, 1.8, 1.3), color=(1.0, 1.0, 1.0))
        scene.ambient_light((0.55, 0.55, 0.55))
        scene.particles(target_pos, radius=args.particle_radius,
                        color=(0.08, 0.80, 0.95))
        scene.particles(pred_pos, radius=args.particle_radius,
                        color=(1.00, 0.48, 0.14))
        canvas.scene(scene)

    print("\nTarget is cyan on the left; prediction is orange on the right.")
    print(f"Target E={e_true:.3f}, nu={nu_true:.4f}; "
          f"predicted E={e_pred:.3f}, nu={nu_pred:.4f}; "
          f"warmup_steps={warmup}")
    print(f"Loaded particle bounds: min={raw_min}, max={raw_max}")

    if not args.record:
        frame_cursor = 0
        print("Viewer mode. Hold right mouse button to adjust the camera.")
        while window.running:
            camera.track_user_inputs(window, movement_speed=0.035,
                                     hold_key=ti.ui.RMB)
            draw(int(frame_ids[frame_cursor]))
            window.show()
            frame_cursor = (frame_cursor + 1) % len(frame_ids)
        return

    ffmpeg = require_ffmpeg()
    os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)
    clear_old_frames(DEFAULT_OUT_DIR)
    for out_index, frame_id in enumerate(frame_ids):
        draw(int(frame_id))
        frame_path = os.path.join(DEFAULT_OUT_DIR, f"frame_{out_index:04d}.png")
        window.save_image(frame_path)
        window.show()
        print(f"Saved {frame_path}")

    encode_mp4(DEFAULT_OUT_DIR, output_mp4_path(), args.fps, ffmpeg)
    remove_frame_dir(DEFAULT_OUT_DIR)


if __name__ == "__main__":
    main()
