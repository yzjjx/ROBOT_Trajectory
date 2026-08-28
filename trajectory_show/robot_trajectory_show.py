import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


DOF = 6
DEFAULT_TCP_SITE = "tool_site"


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


def compute_tcp_positions(model, q_all, site_name):
    """根据关节轨迹计算TCP站点在世界坐标系中的位置。"""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"TCP site not found in XML model: {site_name}")

    forward_data = mujoco.MjData(model)
    tcp_positions = np.empty((len(q_all), 3), dtype=np.float64)
    for index, q in enumerate(q_all):
        forward_data.qpos[:DOF] = q
        mujoco.mj_forward(model, forward_data)
        tcp_positions[index] = forward_data.site_xpos[site_id]

    return tcp_positions


def add_tcp_trajectory_to_scene(scene, tcp_positions, max_segments, line_width, rgba):
    """把显示降采样后的TCP折线加入MuJoCo用户场景。"""
    if max_segments < 1:
        raise ValueError("tcp-segments must be at least 1.")
    if line_width <= 0.0:
        raise ValueError("tcp-line-width must be greater than 0.")

    available_segments = min(max_segments, scene.maxgeom)
    point_count = min(len(tcp_positions), available_segments + 1)
    indices = np.linspace(0, len(tcp_positions) - 1, point_count, dtype=int)
    indices = np.unique(indices)
    display_points = tcp_positions[indices]

    scene.ngeom = 0
    identity = np.eye(3, dtype=np.float64).reshape(-1)
    color = np.asarray(rgba, dtype=np.float32)
    for start, end in zip(display_points[:-1], display_points[1:]):
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_LINE,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            identity,
            color,
        )
        mujoco.mjv_connector(
            geom, mujoco.mjtGeom.mjGEOM_LINE, line_width, start, end
        )
        scene.ngeom += 1

    return scene.ngeom


def run(args):
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent

    # xml_path = args.xml or project_dir / "xml" / "ROKAE_SR4.XML"
    xml_path = args.xml or project_dir / "xml" / "NB4-R475-04.xml"
    # 直线轨迹
    # NB4直线轨迹，(1为直线，2为斜线)速度50mm/s
    trajectory_path = args.trajectory or project_dir / "Trajectory_txt" / "circle_R200_joint_trajectory_NB4.txt"

    # 圆弧轨迹
    # NB4圆弧轨迹半径200与半径20，速度50mm/s
    # trajectory_path = args.trajectory or project_dir / "Trajectory_txt"/"circle_R200_NB4.txt"
    # SR4圆弧轨迹半径200与半径20，速度50mm/s
    # trajectory_path = args.trajectory or project_dir / "Trajectory_txt"/"circle_R20_SR4_V25.txt"

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

    tcp_positions = compute_tcp_positions(model, q_all, args.tcp_site)

    data.qpos[:DOF] = q_all[0]
    data.qvel[:DOF] = dq_all[0]
    mujoco.mj_forward(model, data)

    duration = time_s[-1] - time_s[0]
    print(f"XML: {xml_path}")
    print(f"Trajectory: {trajectory_path}")
    print(f"Samples: {len(time_s)}")
    print(f"Duration: {duration:.6f} s")
    print(f"TCP site: {args.tcp_site}")
    print("Trajectory will hold the final pose after playback. Close the viewer manually.")

    render_interval = 1.0 / max(args.render_fps, 1.0)
    last_render = 0.0
    wall_start = time.time()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = args.azimuth
        viewer.cam.elevation = args.elevation
        viewer.cam.distance = args.distance
        viewer.cam.lookat[:] = np.asarray(args.lookat, dtype=np.float64)

        with viewer.lock():
            tcp_segment_count = add_tcp_trajectory_to_scene(
                viewer.user_scn,
                tcp_positions,
                args.tcp_segments,
                args.tcp_line_width,
                args.tcp_color,
            )
        print(f"TCP trajectory display segments: {tcp_segment_count}")

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
    parser.add_argument("--tcp-site", default=DEFAULT_TCP_SITE)
    parser.add_argument("--tcp-segments", type=int, default=600)
    parser.add_argument("--tcp-line-width", type=float, default=0.003)
    parser.add_argument(
        "--tcp-color", type=float, nargs=4, default=[0.0, 0.35, 1.0, 1.0],
        metavar=("R", "G", "B", "A"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
