import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


DOF = 6


def load_trajectory(path):
    rows = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            text = line.strip()
            if not text:
                continue

            parts = text.replace(",", " ").split()
            try:
                values = [float(part) for part in parts]
            except ValueError:
                continue

            rows.append(values)

    if not rows:
        raise ValueError(f"No numeric trajectory data found: {path}")

    min_cols = DOF + 1
    valid_rows = [row for row in rows if len(row) >= min_cols]
    if not valid_rows:
        raise ValueError(
            f"Trajectory must contain at least {min_cols} numeric columns: "
            "time + q1..q6."
        )

    data = np.asarray(valid_rows, dtype=np.float64)
    time_s = data[:, 0]
    q = data[:, 1 : 1 + DOF]

    if data.shape[1] >= 1 + 2 * DOF:
        dq = data[:, 1 + DOF : 1 + 2 * DOF]
    else:
        dq = np.zeros_like(q)

    order = np.argsort(time_s)
    time_s = time_s[order]
    q = q[order]
    dq = dq[order]

    unique_mask = np.concatenate(([True], np.diff(time_s) > 1e-12))
    time_s = time_s[unique_mask]
    q = q[unique_mask]
    dq = dq[unique_mask]

    if len(time_s) < 2:
        raise ValueError("Trajectory must contain at least two time samples.")

    return time_s, q, dq


def sample(time_s, q, dq, t):
    t = float(np.clip(t, time_s[0], time_s[-1]))
    q_ref = np.array([np.interp(t, time_s, q[:, i]) for i in range(DOF)])
    dq_ref = np.array([np.interp(t, time_s, dq[:, i]) for i in range(DOF)])
    return q_ref, dq_ref


def run(args):
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent

    xml_path = args.xml or project_dir / "xml" / "ROKAE_SR4.XML"
    # 导入文件的位置
    # trajectory_path = args.trajectory or project_dir / "Trajectory_txt" / "joint_trajectory.txt"
    # 圆弧轨迹
    trajectory_path = args.trajectory or project_dir / "Trajectory_txt" / "circle_R200.txt"

    xml_path = xml_path.resolve()
    trajectory_path = trajectory_path.resolve()

    if not xml_path.exists():
        raise FileNotFoundError(f"XML model not found: {xml_path}")
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {trajectory_path}")

    time_s, q_all, dq_all = load_trajectory(trajectory_path)

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    if model.nq < DOF or model.nv < DOF:
        raise ValueError(f"Model nq/nv must be at least {DOF}, got nq={model.nq}, nv={model.nv}")

    data.qpos[:DOF] = q_all[0]
    data.qvel[:DOF] = dq_all[0]
    mujoco.mj_forward(model, data)

    duration = time_s[-1] - time_s[0]
    print(f"XML: {xml_path}")
    print(f"Trajectory: {trajectory_path}")
    print(f"Samples: {len(time_s)}")
    print(f"Duration: {duration:.6f} s")
    print("Trajectory will hold the final pose after playback. Close the viewer manually.")

    render_interval = 1.0 / max(args.render_fps, 1.0)
    last_render = 0.0
    wall_start = time.time()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = args.azimuth
        viewer.cam.elevation = args.elevation
        viewer.cam.distance = args.distance
        viewer.cam.lookat[:] = np.asarray(args.lookat, dtype=np.float64)

        while viewer.is_running():
            elapsed = (time.time() - wall_start) * args.speed
            traj_t = min(time_s[0] + elapsed, time_s[-1])

            q_ref, dq_ref = sample(time_s, q_all, dq_all, traj_t)
            data.time = traj_t
            data.qpos[:DOF] = q_ref
            data.qvel[:DOF] = dq_ref
            mujoco.mj_forward(model, data)

            now = time.time()
            if now - last_render >= render_interval:
                viewer.sync()
                last_render = now

            time.sleep(0.001)


def parse_args():
    parser = argparse.ArgumentParser(description="Show ROKAE joint trajectory in MuJoCo.")
    parser.add_argument("--xml", type=Path, default=None)
    parser.add_argument("--trajectory", type=Path, default=None)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--render-fps", type=float, default=60.0)
    parser.add_argument("--azimuth", type=float, default=135.0)
    parser.add_argument("--elevation", type=float, default=-25.0)
    parser.add_argument("--distance", type=float, default=2.2)
    parser.add_argument("--lookat", type=float, nargs=3, default=[0.2, 0.0, 0.45])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
