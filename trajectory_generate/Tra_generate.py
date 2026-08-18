"""使用 Pinocchio 将 TCP 齐次位姿轨迹转换为 SR4 关节轨迹。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import pinocchio as pin
except ImportError as exc:  # pragma: no cover - 取决于本机Python环境
    raise SystemExit(
        "未安装机器人学 Pinocchio，请先在当前Python环境安装 pinocchio。"
    ) from exc


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_POSE_FILE = PROJECT_DIR / "MATLAB_code" / "circle_R200_TCP_poses.txt"
DEFAULT_URDF_FILE = PROJECT_DIR / "urdf" / "ROKAE_SR4.urdf"
DEFAULT_OUTPUT_FILE = PROJECT_DIR / "Trajectory_txt" / "circle_R200.txt"
DEFAULT_END_FRAME = "xMateSR4C_link6"
# 该初值对应当前圆轨迹的一条可连续运行整圈、且不触碰关节限位的SR4 IK分支。
DEFAULT_INITIAL_Q = np.array(
    [-0.212847607, 0.276685742, -1.629032023,
     -0.581638679, 0.394670354, 0.545525568]
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="利用Pinocchio对4x4 TCP位姿序列逐点进行逆运动学求解。"
    )
    parser.add_argument("--poses", type=Path, default=DEFAULT_POSE_FILE,
                        help="输入TCP位姿TXT；每四个非空行是一个4x4矩阵。")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF_FILE,
                        help="机器人URDF文件。")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE,
                        help="输出关节轨迹TXT。")
    parser.add_argument("--end-frame", default=DEFAULT_END_FRAME,
                        help="与输入TCP位姿对应的URDF末端帧名称。")
    parser.add_argument("--frequency", type=float, default=1000.0,
                        help="轨迹采样频率，单位Hz，默认1000。")
    parser.add_argument("--translation-scale", type=float, default=1e-3,
                        help="输入平移到URDF米制单位的比例，毫米输入默认1e-3。")
    parser.add_argument("--max-iterations", type=int, default=80,
                        help="除第一点外，每个轨迹点的最大迭代次数。")
    parser.add_argument("--tolerance", type=float, default=1e-7,
                        help="SE(3)误差范数收敛阈值。")
    parser.add_argument(
        "--initial-q", type=float, nargs="+", default=None,
        help="可选初始关节角（rad），数量必须等于机器人自由度。",
    )
    return parser.parse_args()


def load_tcp_poses(path: Path, translation_scale: float) -> np.ndarray:
    """读取连续4x4矩阵，并检查其是否为有效刚体位姿。"""
    if not path.is_file():
        raise FileNotFoundError(f"找不到TCP位姿文件：{path}")
    if translation_scale <= 0.0:
        raise ValueError("translation_scale必须大于0。")

    values = np.loadtxt(path, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4 or values.shape[0] % 4 != 0:
        raise ValueError("TCP位姿TXT必须由若干个4x4数值矩阵组成。")

    poses = values.reshape(-1, 4, 4)
    if not np.isfinite(poses).all():
        raise ValueError("TCP位姿中包含NaN或Inf。")

    expected_last_row = np.array([0.0, 0.0, 0.0, 1.0])
    if not np.allclose(poses[:, 3, :], expected_last_row, atol=1e-9):
        raise ValueError("所有TCP位姿的最后一行必须为[0, 0, 0, 1]。")

    rotations = poses[:, :3, :3]
    orthogonality = np.matmul(rotations.transpose(0, 2, 1), rotations)
    if not np.allclose(orthogonality, np.eye(3), atol=1e-7):
        raise ValueError("输入TCP位姿包含非正交旋转矩阵。")
    if not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-7):
        raise ValueError("输入TCP位姿包含行列式不为1的旋转矩阵。")

    poses = poses.copy()
    poses[:, :3, 3] *= translation_scale
    return poses


def pose_error_and_jacobian(
    model: pin.Model,
    data: pin.Data,
    q: np.ndarray,
    frame_id: int,
    target: pin.SE3,
) -> tuple[np.ndarray, np.ndarray]:
    """返回当前末端到目标位姿的局部SE(3)误差及其雅可比。"""
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacement(model, data, frame_id)
    frame_to_target = data.oMf[frame_id].actInv(target)
    error = pin.log6(frame_to_target).vector
    jacobian = pin.computeFrameJacobian(
        model, data, q, frame_id, pin.ReferenceFrame.LOCAL
    )
    jacobian = -pin.Jlog6(frame_to_target.inverse()) @ jacobian
    return error, jacobian


def clamp_to_limits(model: pin.Model, q: np.ndarray) -> np.ndarray:
    lower = model.lowerPositionLimit
    upper = model.upperPositionLimit
    finite_lower = np.where(np.isfinite(lower), lower, -np.inf)
    finite_upper = np.where(np.isfinite(upper), upper, np.inf)
    return np.clip(q, finite_lower, finite_upper)


def solve_one_pose(
    model: pin.Model,
    data: pin.Data,
    frame_id: int,
    target: pin.SE3,
    q_start: np.ndarray,
    max_iterations: int,
    tolerance: float,
) -> tuple[np.ndarray, float, bool]:
    """采用带阻尼和回溯线搜索的Levenberg-Marquardt方法求一个位姿。"""
    q = clamp_to_limits(model, np.asarray(q_start, dtype=float).copy())
    damping = 1e-6

    for _ in range(max_iterations):
        error, jacobian = pose_error_and_jacobian(model, data, q, frame_id, target)
        error_norm = float(np.linalg.norm(error))
        if error_norm < tolerance:
            return q, error_norm, True

        normal_matrix = jacobian.T @ jacobian + damping * np.eye(model.nv)
        try:
            step = np.linalg.solve(normal_matrix, -jacobian.T @ error)
        except np.linalg.LinAlgError:
            step = -np.linalg.pinv(jacobian) @ error

        step_norm = np.linalg.norm(step)
        if step_norm > 0.5:
            step *= 0.5 / step_norm

        accepted = False
        step_scale = 1.0
        for _ in range(10):
            candidate = pin.integrate(model, q, step_scale * step)
            candidate = clamp_to_limits(model, candidate)
            candidate_error, _ = pose_error_and_jacobian(
                model, data, candidate, frame_id, target
            )
            if np.linalg.norm(candidate_error) < error_norm:
                q = candidate
                damping = max(1e-10, damping * 0.5)
                accepted = True
                break
            step_scale *= 0.5

        if not accepted:
            damping = min(1e6, damping * 10.0)

    final_error, _ = pose_error_and_jacobian(model, data, q, frame_id, target)
    final_norm = float(np.linalg.norm(final_error))
    return q, final_norm, final_norm < tolerance


def initial_candidates(model: pin.Model, user_initial_q: list[float] | None) -> list[np.ndarray]:
    """为第一个位姿构造确定性的多初值，降低落入错误IK分支的概率。"""
    if user_initial_q is not None:
        q = np.asarray(user_initial_q, dtype=float)
        if q.shape != (model.nq,):
            raise ValueError(f"--initial-q需要{model.nq}个数，实际得到{q.size}个。")
        return [clamp_to_limits(model, q)]

    candidates = []
    if model.nq == DEFAULT_INITIAL_Q.size:
        candidates.append(clamp_to_limits(model, DEFAULT_INITIAL_Q))
    candidates.append(pin.neutral(model))
    lower = model.lowerPositionLimit
    upper = model.upperPositionLimit
    if np.isfinite(lower).all() and np.isfinite(upper).all():
        candidates.append((lower + upper) / 2.0)
        rng = np.random.default_rng(20260818)
        candidates.extend(rng.uniform(lower, upper) for _ in range(12))
    return candidates


def solve_trajectory(
    model: pin.Model,
    poses: np.ndarray,
    frame_id: int,
    initial_q: list[float] | None,
    max_iterations: int,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    if max_iterations <= 0:
        raise ValueError("max_iterations必须大于0。")
    if tolerance <= 0.0:
        raise ValueError("tolerance必须大于0。")

    targets = [pin.SE3(pose[:3, :3], pose[:3, 3]) for pose in poses]
    best_q = None
    best_error = np.inf
    for candidate in initial_candidates(model, initial_q):
        data = model.createData()
        q, error, converged = solve_one_pose(
            model, data, frame_id, targets[0], candidate,
            max(max_iterations * 10, 500), tolerance,
        )
        if error < best_error:
            best_q, best_error = q, error
        if converged:
            break

    if best_q is None or best_error >= tolerance:
        raise RuntimeError(
            f"第1个TCP位姿IK未收敛，最小SE(3)误差范数为{best_error:.3e}。"
        )

    point_count = len(targets)
    joint_positions = np.empty((point_count, model.nq))
    residuals = np.empty(point_count)
    joint_positions[0] = best_q
    residuals[0] = best_error

    data = model.createData()
    q = best_q
    for index in range(1, point_count):
        q, error, converged = solve_one_pose(
            model, data, frame_id, targets[index], q, max_iterations, tolerance
        )
        if not converged:
            raise RuntimeError(
                f"第{index + 1}个TCP位姿IK未收敛，SE(3)误差范数为{error:.3e}。"
            )
        joint_positions[index] = q
        residuals[index] = error
        if (index + 1) % 1000 == 0 or index + 1 == point_count:
            print(f"已求解 {index + 1}/{point_count} 个TCP位姿")

    return joint_positions, residuals


def save_joint_trajectory(
    output_path: Path,
    joint_positions: np.ndarray,
    frequency: float,
) -> None:
    if frequency <= 0.0:
        raise ValueError("frequency必须大于0。")

    dt = 1.0 / frequency
    times = np.arange(len(joint_positions), dtype=float) * dt
    edge_order = 2 if len(joint_positions) >= 3 else 1
    joint_velocities = np.gradient(joint_positions, dt, axis=0, edge_order=edge_order)
    output_data = np.column_stack((times, joint_positions, joint_velocities))

    joint_count = joint_positions.shape[1]
    header_fields = ["time_s"]
    header_fields.extend(f"q{i}_rad" for i in range(1, joint_count + 1))
    header_fields.extend(f"dq{i}_rad_s" for i in range(1, joint_count + 1))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    np.savetxt(
        temporary_path,
        output_data,
        fmt="%.9f",
        delimiter="\t",
        header="\t".join(header_fields),
        comments="",
    )
    temporary_path.replace(output_path)


def main() -> int:
    args = parse_arguments()
    if not hasattr(pin, "buildModelFromUrdf"):
        raise RuntimeError(
            "当前导入的是同名的nose测试插件pinocchio，而不是机器人学Pinocchio。"
            "请切换到正确环境后运行，例如：conda run -n pinocchio-env python "
            "trajectory_generate/Tra_generate.py"
        )

    pose_path = args.poses.resolve()
    urdf_path = args.urdf.resolve()
    output_path = args.output.resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"找不到URDF文件：{urdf_path}")

    poses = load_tcp_poses(pose_path, args.translation_scale)
    model = pin.buildModelFromUrdf(str(urdf_path))
    if model.nq != model.nv:
        raise ValueError("当前脚本只支持nq等于nv的固定基座机械臂。")

    frame_id = model.getFrameId(args.end_frame)
    if frame_id >= len(model.frames):
        raise ValueError(f"URDF中不存在末端帧：{args.end_frame}")

    print(f"URDF：{urdf_path}")
    print(f"末端帧：{args.end_frame}")
    print(f"TCP位姿数：{len(poses)}")
    joint_positions, residuals = solve_trajectory(
        model, poses, frame_id, args.initial_q,
        args.max_iterations, args.tolerance,
    )
    save_joint_trajectory(output_path, joint_positions, args.frequency)

    print(f"最大SE(3)误差范数：{residuals.max():.3e}")
    print(f"关节轨迹已写入：{output_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        sys.exit(1)
