"""在MuJoCo中调用C++ RNEA力矩前馈PD控制器跟踪关节轨迹。"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

DOF = 6

def read_trajectory(file_path):
    """读取time、六关节位置和六关节速度，并计算期望加速度。"""
    table = np.loadtxt(file_path, skiprows=1, dtype=np.float64)
    table = np.atleast_2d(table)

    if table.shape[1] < 13:
        raise ValueError(
            "Trajectory file needs 13 columns: time, 6 q values and 6 dq values."
        )
    if len(table) < 2:
        raise ValueError("Trajectory file needs at least two data rows.")
    if not np.isfinite(table[:, :13]).all():
        raise ValueError("Trajectory contains NaN or Inf values.")

    time_s = table[:, 0]
    q_desired = table[:, 1:7]
    dq_desired = table[:, 7:13]
    time_step = np.diff(time_s)
    if np.any(time_step <= 0.0):
        raise ValueError("The trajectory time column must be strictly increasing.")

    trajectory_dt = float(np.median(time_step))
    edge_order = 2 if len(time_s) >= 3 else 1
    ddq_desired = np.gradient(
        dq_desired, time_s, axis=0, edge_order=edge_order
    )
    return time_s, q_desired, dq_desired, ddq_desired, trajectory_dt


def add_dll_directory(folder, handles):
    """在导入pybind11模块前注册其依赖DLL目录。"""
    if not folder.exists():
        return
    if hasattr(os, "add_dll_directory"):
        handles.append(os.add_dll_directory(str(folder)))
    else:
        os.environ["PATH"] = str(folder) + os.pathsep + os.environ.get("PATH", "")


def import_cpp_controller(project_dir):
    """导入由RNEA_PD.cpp和bindcode.cpp编译得到的Python扩展模块。"""
    dll_handles = []
    python_dir = Path(sys.executable).resolve().parent
    for folder in (
        python_dir,
        python_dir / "Library" / "bin",
        Path("D:/GCC_WIN64/mingw64/bin"),
    ):
        add_dll_directory(folder, dll_handles)

    for folder in (
        project_dir / "build_msvc" / "Release",
        project_dir / "build_msvc",
        project_dir / "build" / "Release",
        project_dir / "build",
    ):
        if folder.exists():
            sys.path.insert(0, str(folder))

    try:
        import RNEA_PD
    except ImportError as error:
        raise ImportError(
            "Cannot import the compiled RNEA_PD module. Build RNEA_PD.cpp and "
            "bindcode.cpp with the same Python/Pinocchio environment used here."
        ) from error

    # os.add_dll_directory返回的句柄必须保持存活，否则目录会被立即注销。
    RNEA_PD._dll_directory_handles = dll_handles
    return RNEA_PD


def parse_numbers(text, count, name):
    values = [float(value.strip()) for value in text.split(",")]
    if len(values) != count or not np.isfinite(values).all():
        raise ValueError(
            f"{name} must contain exactly {count} finite comma-separated numbers."
        )
    return np.asarray(values, dtype=np.float64)


def tcp_position(model, data, site_id):
    """返回当前关节状态下TCP站点的世界坐标。"""
    mujoco.mj_forward(model, data)
    return data.site_xpos[site_id].copy()


def add_line(scene, start, end, color, width):
    if scene.ngeom >= scene.maxgeom:
        return False

    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_LINE,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).reshape(-1),
        np.asarray(color, dtype=np.float32),
    )
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, width, start, end)
    scene.ngeom += 1
    return True


def draw_tcp_trajectories(viewer, reference_points, actual_points):
    """期望TCP轨迹为绿色，实际TCP轨迹为红色。"""
    scene = viewer.user_scn
    scene.ngeom = 0
    for points, color in (
        (reference_points, (0.0, 1.0, 0.0, 1.0)),
        (actual_points, (1.0, 0.0, 0.0, 1.0)),
    ):
        for start, end in zip(points[:-1], points[1:]):
            if not add_line(scene, start, end, color, 3.0):
                return


def save_log(file_path, rows):
    header = (
        ["time_s"]
        + [f"q_ref_{i}" for i in range(1, 7)]
        + [f"q_{i}" for i in range(1, 7)]
        + [f"dq_ref_{i}" for i in range(1, 7)]
        + [f"dq_{i}" for i in range(1, 7)]
        + [f"ddq_ref_{i}" for i in range(1, 7)]
        + [f"tau_{i}" for i in range(1, 7)]
        + ["tcp_ref_x", "tcp_ref_y", "tcp_ref_z"]
        + ["tcp_x", "tcp_y", "tcp_z"]
    )
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def run(args):
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    xml_path = (args.xml or project_dir / "xml" / "ROKAE_SR4.XML").resolve()
    urdf_path = (args.urdf or project_dir / "urdf" / "ROKAE_SR4.urdf").resolve()
    trajectory_path = (
        args.trajectory or project_dir / "Trajectory_txt" / "circle_R200_SR4.txt"
    ).resolve()
    log_path = (
        args.log
        or project_dir / "RNEA_PD_control_out" / "RNEA_PD_control_SR4_mujoco.csv"
    ).resolve()

    if args.trace_points < 2:
        raise ValueError("trace-points must be at least 2.")
    for path, description in (
        (xml_path, "XML"),
        (urdf_path, "URDF"),
        (trajectory_path, "trajectory"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{description} file not found: {path}")

    time_s, q_desired, dq_desired, ddq_desired, trajectory_dt = read_trajectory(
        trajectory_path
    )
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    if model.nq < DOF or model.nv < DOF or model.nu < DOF:
        raise ValueError("This program needs a six-joint, six-actuator robot model.")

    # URDF/Pinocchio默认使用-z方向9.81 m/s^2重力，MuJoCo必须保持一致。
    gravity = parse_numbers(args.gravity, 3, "gravity")
    model.opt.gravity[:] = gravity
    model.opt.timestep = trajectory_dt
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0

    controller_module = import_cpp_controller(project_dir)
    kp = parse_numbers(args.kp, DOF, "kp")
    kd = parse_numbers(args.kd, DOF, "kd")
    torque_limit = parse_numbers(args.torque_limit, DOF, "torque-limit")
    if np.any(kp < 0.0) or np.any(kd < 0.0):
        raise ValueError("kp and kd must be non-negative.")
    if np.any(torque_limit <= 0.0):
        raise ValueError("torque-limit must be positive.")

    controller = controller_module.RNEA_PD(
        kp.tolist(), kd.tolist(), str(urdf_path)
    )
    data = mujoco.MjData(model)
    reference_data = mujoco.MjData(model)
    tcp_site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, args.tcp_site
    )
    if tcp_site_id < 0:
        raise ValueError(f"TCP site not found: {args.tcp_site}")

    data.qpos[:DOF] = q_desired[0]
    data.qvel[:DOF] = dq_desired[0]
    mujoco.mj_forward(model, data)

    trace_stride = max(1, int(np.ceil(len(time_s) / args.trace_points)))
    reference_trace = []
    actual_trace = []
    log_rows = []
    tcp_errors = []

    print(f"XML: {xml_path}")
    print(f"URDF: {urdf_path}")
    print(f"Trajectory: {trajectory_path}")
    print(f"Trajectory samples: {len(time_s)}, dt: {trajectory_dt:.6f} s")
    print(f"C++ controller: {controller_module.__file__}")
    print(f"Kp: {kp.tolist()}")
    print(f"Kd: {kd.tolist()}")
    print(f"Gravity (m/s^2): {gravity.tolist()}")
    print(f"Torque limits (N*m): {torque_limit.tolist()}")
    print("Collision is disabled. TCP reference is green; actual TCP is red.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        viewer.cam.distance = 2.2
        viewer.cam.lookat[:] = np.array([0.2, 0.0, 0.45])

        for index in range(len(time_s)):
            if not viewer.is_running():
                break
            step_start = time.time()

            q_ref = q_desired[index]
            dq_ref = dq_desired[index]
            ddq_ref = ddq_desired[index]
            q = data.qpos[:DOF].copy()
            dq = data.qvel[:DOF].copy()

            # tau = RNEA(q_ref, dq_ref, ddq_ref)
            #       + Kp*(q_ref-q) + Kd*(dq_ref-dq)
            torque = np.asarray(
                controller.RNEA_PD_control(
                    q_ref.tolist(),
                    dq_ref.tolist(),
                    ddq_ref.tolist(),
                    q.tolist(),
                    dq.tolist(),
                ),
                dtype=np.float64,
            )
            torque = np.clip(torque, -torque_limit, torque_limit)
            data.ctrl[:DOF] = torque

            reference_data.qpos[:DOF] = q_ref
            reference_data.qvel[:DOF] = dq_ref
            tcp_ref = tcp_position(model, reference_data, tcp_site_id)
            tcp_actual = tcp_position(model, data, tcp_site_id)
            tcp_errors.append(float(np.linalg.norm(tcp_ref - tcp_actual)))

            if index % trace_stride == 0 or index == len(time_s) - 1:
                reference_trace.append(tcp_ref)
                actual_trace.append(tcp_actual)

            if args.save_log:
                log_rows.append(
                    [time_s[index]]
                    + q_ref.tolist()
                    + q.tolist()
                    + dq_ref.tolist()
                    + dq.tolist()
                    + ddq_ref.tolist()
                    + torque.tolist()
                    + tcp_ref.tolist()
                    + tcp_actual.tolist()
                )

            mujoco.mj_step(model, data)
            draw_tcp_trajectories(viewer, reference_trace, actual_trace)
            viewer.sync()

            if args.real_time:
                remaining_time = trajectory_dt - (time.time() - step_start)
                if remaining_time > 0.0:
                    time.sleep(remaining_time)

        while viewer.is_running():
            draw_tcp_trajectories(viewer, reference_trace, actual_trace)
            viewer.sync()
            time.sleep(0.02)

    if args.save_log:
        save_log(log_path, log_rows)
        print(f"Log saved: {log_path}")
    if tcp_errors:
        print(f"TCP final error: {tcp_errors[-1]:.6f} m")
        print(f"TCP max error: {max(tcp_errors):.6f} m")


def parse_args():
    parser = argparse.ArgumentParser(
        description="ROKAE MuJoCo RNEA feedforward plus PD trajectory tracking."
    )
    parser.add_argument("--xml", type=Path)
    parser.add_argument("--urdf", type=Path)
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--kp", default="100,100,80,20,10,3")
    parser.add_argument("--kd", default="20,20,16,2,0.5,0.08")
    parser.add_argument("--torque-limit", default="300,300,300,300,300,300")
    parser.add_argument("--gravity", default="0,0,-9.81")
    parser.add_argument("--tcp-site", default="tool_site")
    parser.add_argument("--trace-points", type=int, default=400)
    parser.add_argument("--real-time", action="store_true", default=True)
    parser.add_argument("--no-real-time", dest="real_time", action="store_false")
    parser.add_argument("--save-log", action="store_true", default=True)
    parser.add_argument("--no-save-log", dest="save_log", action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
