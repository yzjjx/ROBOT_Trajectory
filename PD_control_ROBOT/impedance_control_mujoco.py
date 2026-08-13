# 该代码使用C++生成动态链接库的控制器，实现基于mujoco环境下的阻抗控制

import argparse
import csv
import os
import sys
import time
from pathlib import Path # 提供面向对象的文件路径操作

import mujoco
import mujoco.viewer
import numpy as np


DOF = 6


def read_trajectory(file_path):
    """Read the fixed 13-column trajectory format described above."""
    table = np.loadtxt(file_path, skiprows=1, dtype=np.float64)
    table = np.atleast_2d(table)

    if table.shape[1] < 13:
        raise ValueError("Trajectory file needs 13 columns: time, 6 q values and 6 dq values.")
    if len(table) < 2:
        raise ValueError("Trajectory file needs at least two data rows.")

    time_s = table[:, 0]
    q_desired = table[:, 1:7]
    dq_desired = table[:, 7:13]

    dt = float(np.median(np.diff(time_s)))
    if dt <= 0.0:
        raise ValueError("The time column must increase.")

    return time_s, q_desired, dq_desired, dt


def add_dll_directory(folder):
    """Make dependent DLLs visible before importing the pybind11 module."""
    if not folder.exists():
        return
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(folder))
    else:
        os.environ["PATH"] = str(folder) + os.pathsep + os.environ.get("PATH", "")


def import_cpp_pd_controller(project_dir):
    """Import build/PD_ROBOT.cp310-win_amd64.pyd (or its matching variant)."""
    python_dir = Path(sys.executable).resolve().parent
    add_dll_directory(python_dir)
    add_dll_directory(python_dir / "Library" / "bin")
    add_dll_directory(Path("D:/GCC_WIN64/mingw64/bin"))

    for folder in (project_dir / "build", project_dir / "build" / "Release"):
        if folder.exists():
            sys.path.insert(0, str(folder))

    try:
        import PD_ROBOT
    except ImportError as error:
        raise ImportError(
            "Cannot import the compiled PD_ROBOT module. "
            "Build it using the same Python version that runs this script."
        ) from error
    return PD_ROBOT


def parse_six_numbers(text, name):
    values = [float(value.strip()) for value in text.split(",")]
    if len(values) != DOF:
        raise ValueError(f"{name} must contain exactly six comma-separated numbers.")
    return np.asarray(values, dtype=np.float64)


def tcp_position(model, data, site_id):
    """Return the world position of the TCP site for the current joint state."""
    mujoco.mj_forward(model, data)
    return data.site_xpos[site_id].copy()


def add_line(scene, start, end, color, width):
    """Add one line segment to MuJoCo's user drawing scene."""
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
    """Draw desired TCP path in green and actual TCP path in red."""
    scene = viewer.user_scn
    scene.ngeom = 0  # Clear the old drawing before drawing the complete paths again.

    for points, color in (
        (reference_points, (0.0, 1.0, 0.0, 1.0)),
        (actual_points, (1.0, 0.0, 0.0, 1.0)),
    ):
        for start, end in zip(points[:-1], points[1:]):
            if not add_line(scene, start, end, color, 3.0):
                return


def save_log(file_path, rows):
    """Save q, dq and TCP positions for later checking."""
    header = (
        ["time_s"]
        + [f"q_ref_{i}" for i in range(1, 7)]
        + [f"q_{i}" for i in range(1, 7)]
        + [f"dq_ref_{i}" for i in range(1, 7)]
        + [f"dq_{i}" for i in range(1, 7)]
        + [f"tau_{i}" for i in range(1, 7)]
        + ["tcp_ref_x", "tcp_ref_y", "tcp_ref_z"]
        + ["tcp_x", "tcp_y", "tcp_z"]
    )
    with open(file_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def run(args):
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    xml_path = (args.xml or project_dir / "xml" / "ROKAE_SR4.XML").resolve()
    trajectory_path = (args.trajectory or project_dir / "Trajectory_txt" / "joint_trajectory.txt").resolve()
    log_path = (args.log or script_dir / ".." /"PD_control_out"/"PD_control_SR4_mujoco.csv").resolve()

    if args.trace_points < 2:
        raise ValueError("trace-points must be at least 2.")
    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {trajectory_path}")

    time_s, q_desired, dq_desired, trajectory_dt = read_trajectory(trajectory_path)
    model = mujoco.MjModel.from_xml_path(str(xml_path))

    # The XML uses visual meshes as collision meshes. They initially overlap,
    # so disable collisions for this free-space trajectory-tracking simulation.
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    model.opt.timestep = trajectory_dt

    if model.nq < DOF or model.nv < DOF or model.nu < DOF:
        raise ValueError("This program needs a six-joint, six-actuator robot model.")

    pd_module = import_cpp_pd_controller(project_dir)
    kp = parse_six_numbers(args.kp, "kp")
    kd = parse_six_numbers(args.kd, "kd")
    torque_limit = parse_six_numbers(args.torque_limit, "torque-limit")
    controller = pd_module.PD_ROBOT(kp.tolist(), kd.tolist())

    data = mujoco.MjData(model)
    reference_data = mujoco.MjData(model)
    tcp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, args.tcp_site)
    if tcp_site_id < 0:
        raise ValueError(f"TCP site not found: {args.tcp_site}")

    # Start exactly at the first reference point.
    data.qpos[:DOF] = q_desired[0]
    data.qvel[:DOF] = dq_desired[0]
    mujoco.mj_forward(model, data)

    # Keep the number of displayed line segments manageable.
    trace_stride = max(1, int(np.ceil(len(time_s) / args.trace_points)))
    reference_trace = []
    actual_trace = []
    log_rows = []
    tcp_errors = []

    print(f"XML: {xml_path}")
    print(f"Trajectory: {trajectory_path}")
    print(f"Trajectory samples: {len(time_s)}, dt: {trajectory_dt:.6f} s")
    print(f"C++ controller: {pd_module.__file__}")
    print(f"Kp: {kp.tolist()}")
    print(f"Kd: {kd.tolist()}")
    print(f"Torque limits (Nm): {torque_limit.tolist()}")
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

            # These two values are read directly from columns 2-7 and 8-13.
            q_ref = q_desired[index]
            dq_ref = dq_desired[index]
            q = data.qpos[:DOF].copy()
            dq = data.qvel[:DOF].copy()

            # The C++ function implements:
            # tau = Kp * (q_ref - q) + Kd * (dq_ref - dq)
            torque = np.asarray(
                controller.PD_control(q_ref.tolist(), q.tolist(), dq_ref.tolist(), dq.tolist()),
                dtype=np.float64,
            )
            torque = np.clip(torque, -torque_limit, torque_limit)

            # In ROKAE_SR4.XML every motor has gear="1", therefore ctrl=tau.
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
                    + q_ref.tolist() + q.tolist()
                    + dq_ref.tolist() + dq.tolist()
                    + torque.tolist() + tcp_ref.tolist() + tcp_actual.tolist()
                )

            mujoco.mj_step(model, data)
            draw_tcp_trajectories(viewer, reference_trace, actual_trace)
            viewer.sync()

            if args.real_time:
                remaining_time = trajectory_dt - (time.time() - step_start)
                if remaining_time > 0.0:
                    time.sleep(remaining_time)         

        # Keep the final robot pose and the two TCP paths visible.
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
    parser = argparse.ArgumentParser(description="ROKAE MuJoCo joint-space PD tracking.")
    parser.add_argument("--xml", type=Path)
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--kp", default="100,100,80,20,10,3")
    parser.add_argument("--kd", default="20,20,16,2,0.5,0.08")
    parser.add_argument("--torque-limit", default="100,100,60,15,5,1")
    parser.add_argument("--tcp-site", default="tool_site")
    parser.add_argument("--trace-points", type=int, default=400)
    parser.add_argument("--real-time", action="store_true", default=True)
    parser.add_argument("--no-real-time", dest="real_time", action="store_false")
    parser.add_argument("--save-log", action="store_true", default=True)
    parser.add_argument("--no-save-log", dest="save_log", action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
