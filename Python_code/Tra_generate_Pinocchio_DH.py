"""使用 URDF 和 Pinocchio 将 TCP 位姿轨迹转换为关节轨迹。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import solve as solve_linear_system

try:
    import pinocchio as pin
except ImportError as exc:
    raise SystemExit("未安装机器人学 Pinocchio，请使用 pinocchio-env 运行。") from exc

if not hasattr(pin, "buildModelFromUrdf"):
    raise SystemExit("当前导入的不是机器人学 Pinocchio。\n请运行：conda activate pinocchio-env")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF_FILE = PROJECT_ROOT / "urdf" / "ROKAE_CR20.urdf"
DEFAULT_POSE_FILE = PROJECT_ROOT / "Trajectory_TCP" / "circle_R500_TCP_poses_CR20_V50.txt"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "Trajectory_txt" / "circle_R500_joint_trajectory_CR20_V50.txt"

# link6 坐标系位于第 6 轴法兰处，本脚本将它作为 TCP 坐标系。
DEFAULT_TCP_FRAME = "XMC20-R1650-W7S3B1_link6"
DEFAULT_INITIAL_Q = np.deg2rad([-133.471137, -31.522699, -50.446545, -48.115254, 102.892501, 166.028089])
JOINT_COUNT = len(DEFAULT_INITIAL_Q)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 URDF + Pinocchio 将 TCP 位姿转换为 q/dq。")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF_FILE)
    parser.add_argument("--tcp-frame", default=DEFAULT_TCP_FRAME)
    parser.add_argument("--poses", type=Path, default=DEFAULT_POSE_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--frequency", type=float, default=1000.0)
    parser.add_argument("--translation-scale", type=float, default=1e-3)
    parser.add_argument("--max-iterations", type=int, default=80)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--speed-profile", choices=("s-curve", "uniform"), default="s-curve")
    parser.add_argument("--ramp-time", type=float, default=1.0)
    parser.add_argument(
        "--joint-velocity-scale", type=float, default=0.8,
        help="URDF 关节限速的使用比例，默认 0.8。",
    )
    parser.add_argument(
        "--joint-velocity-limits", type=float, nargs=JOINT_COUNT,
        help="可选：各关节最大速度 rad/s；覆盖 URDF 中的 velocity。",
    )
    parser.add_argument(
        "--initial-q", type=float, nargs=JOINT_COUNT,
        help="首个位姿的 IK 初值，单位 rad。",
    )
    return parser.parse_args()


def build_model(
    urdf_path: Path = DEFAULT_URDF_FILE,
    tcp_frame: str = DEFAULT_TCP_FRAME,
) -> tuple[pin.Model, int, np.ndarray, np.ndarray, np.ndarray]:
    """从 URDF 读取模型、TCP、关节限位和速度上限。"""
    urdf_path = Path(urdf_path).resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"找不到 URDF：{urdf_path}")

    model = pin.buildModelFromUrdf(str(urdf_path))
    if model.nq != JOINT_COUNT or model.nv != JOINT_COUNT:
        raise ValueError(
            f"期望 {JOINT_COUNT} 个单自由度关节，URDF 实际为 nq={model.nq}, nv={model.nv}。"
        )
    if not model.existFrame(tcp_frame):
        raise ValueError(f"URDF 中找不到 TCP frame：{tcp_frame}")

    lower = np.asarray(model.lowerPositionLimit, dtype=float).copy()
    upper = np.asarray(model.upperPositionLimit, dtype=float).copy()
    velocity_limits = np.asarray(model.velocityLimit, dtype=float).copy()
    if np.any(lower >= upper):
        raise ValueError("URDF 关节限位无效：必须满足 lower < upper。")
    if np.any(velocity_limits <= 0.0):
        raise ValueError("URDF 关节速度上限必须大于 0。")

    return model, model.getFrameId(tcp_frame), lower, upper, velocity_limits


def load_poses(path: Path, translation_scale: float) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"找不到 TCP 位姿文件：{path}")
    if translation_scale <= 0.0:
        raise ValueError("translation-scale 必须大于 0。")

    values = np.loadtxt(path, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4 or values.shape[0] % 4:
        raise ValueError("TCP 文件必须由连续的 4×4 数值矩阵组成。")

    poses = values.reshape(-1, 4, 4)
    rotations = poses[:, :3, :3]
    if not np.isfinite(poses).all():
        raise ValueError("TCP 位姿含 NaN 或 Inf。")
    if not np.allclose(poses[:, 3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError("TCP 位姿的最后一行必须为 [0, 0, 0, 1]。")
    # einsum/cross 避免部分 Windows 环境中 Pinocchio 与 NumPy 批量 linalg 的冲突。
    orthogonality = np.einsum("nji,njk->nik", rotations, rotations)
    determinants = np.einsum(
        "ij,ij->i", rotations[:, 0], np.cross(rotations[:, 1], rotations[:, 2])
    )
    if not np.allclose(orthogonality, np.eye(3), atol=1e-7):
        raise ValueError("TCP 位姿含非正交旋转矩阵。")
    if not np.allclose(determinants, 1.0, atol=1e-7):
        raise ValueError("TCP 旋转矩阵的行列式必须为 1。")

    poses = poses.copy()
    poses[:, :3, 3] *= translation_scale
    return poses


def error_and_jacobian(
    model: pin.Model, data: pin.Data, frame_id: int,
    q: np.ndarray, target: pin.SE3,
) -> tuple[np.ndarray, np.ndarray]:
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacement(model, data, frame_id)
    delta = data.oMf[frame_id].actInv(target)
    error = pin.log6(delta).vector
    jacobian = pin.computeFrameJacobian(
        model, data, q, frame_id, pin.ReferenceFrame.LOCAL
    )
    return error, -np.einsum("ij,jk->ik", pin.Jlog6(delta.inverse()), jacobian)


def solve_pose(
    model: pin.Model, data: pin.Data, frame_id: int, target: pin.SE3,
    q_start: np.ndarray, lower: np.ndarray, upper: np.ndarray,
    max_iterations: int, tolerance: float,
) -> tuple[np.ndarray, float, bool]:
    """用阻尼最小二乘法求一个 TCP 位姿的 IK。"""
    q = np.clip(np.asarray(q_start, dtype=float), lower, upper).copy()
    damping = 1e-6

    for _ in range(max_iterations):
        error, jacobian = error_and_jacobian(model, data, frame_id, q, target)
        error_norm = float(np.linalg.norm(error))
        if error_norm < tolerance:
            return q, error_norm, True

        hessian = np.einsum("ki,kj->ij", jacobian, jacobian)
        hessian += damping * np.eye(model.nv)
        gradient = np.einsum("ki,k->i", jacobian, error)
        step = solve_linear_system(hessian, -gradient, assume_a="pos")
        step_norm = np.linalg.norm(step)
        if step_norm > 0.5:
            step *= 0.5 / step_norm

        for scale in 0.5 ** np.arange(10):
            candidate = np.clip(pin.integrate(model, q, scale * step), lower, upper)
            candidate_error, _ = error_and_jacobian(model, data, frame_id, candidate, target)
            if np.linalg.norm(candidate_error) < error_norm:
                q = candidate
                damping = max(1e-10, damping * 0.5)
                break
        else:
            damping = min(1e6, damping * 10.0)

    final_error, _ = error_and_jacobian(model, data, frame_id, q, target)
    final_norm = float(np.linalg.norm(final_error))
    return q, final_norm, final_norm < tolerance


def solve_trajectory(
    model: pin.Model, frame_id: int, poses: np.ndarray,
    lower: np.ndarray, upper: np.ndarray, initial_q: list[float] | None,
    max_iterations: int, tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    if max_iterations <= 0 or tolerance <= 0.0:
        raise ValueError("max-iterations 和 tolerance 必须大于 0。")

    targets = [pin.SE3(pose[:3, :3], pose[:3, 3]) for pose in poses]
    rng = np.random.default_rng(20260830)
    seeds = [np.asarray(initial_q)] if initial_q is not None else [
        DEFAULT_INITIAL_Q,
        pin.neutral(model),
        *(rng.uniform(lower, upper) for _ in range(12)),
    ]

    first_results = []
    for seed in seeds:
        result = solve_pose(
            model, model.createData(), frame_id, targets[0], seed, lower, upper,
            max(500, 10 * max_iterations), tolerance,
        )
        first_results.append(result)
        if result[2]:
            break

    q, error, converged = min(first_results, key=lambda result: result[1])
    if not converged:
        raise RuntimeError(
            f"第 1 个位姿 IK 未收敛，最小误差为 {error:.3e}。"
            "请检查 URDF、TCP frame 和目标可达性。"
        )

    positions = np.empty((len(targets), model.nq))
    residuals = np.empty(len(targets))
    positions[0], residuals[0] = q, error
    data = model.createData()

    for i, target in enumerate(targets[1:], start=1):
        q, error, converged = solve_pose(
            model, data, frame_id, target, q, lower, upper,
            max_iterations, tolerance,
        )
        if not converged:
            raise RuntimeError(f"第 {i + 1} 个位姿 IK 未收敛，误差为 {error:.3e}。")
        positions[i], residuals[i] = q, error
        if (i + 1) % 1000 == 0 or i + 1 == len(targets):
            print(f"IK：{i + 1}/{len(targets)}")

    return positions, residuals


def interpolate_path(path: np.ndarray, progress: np.ndarray) -> np.ndarray:
    source = np.linspace(0.0, 1.0, len(path))
    return np.column_stack(
        [np.interp(progress, source, path[:, joint]) for joint in range(path.shape[1])]
    )


def time_parameterize(
    path: np.ndarray, frequency: float, profile: str, ramp_time: float,
    velocity_limits: np.ndarray, velocity_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if frequency <= 0.0 or ramp_time <= 0.0:
        raise ValueError("frequency 和 ramp-time 必须大于 0。")
    if len(path) < 2:
        raise ValueError("轨迹至少需要两个点。")
    if not 0.0 < velocity_scale <= 1.0:
        raise ValueError("joint-velocity-scale 必须位于 (0, 1]。")

    dt = 1.0 / frequency
    nominal_duration = (len(path) - 1) * dt
    source_progress = np.linspace(0.0, 1.0, len(path))
    edge_order = 2 if len(path) >= 3 else 1
    dq_ds = np.gradient(path, source_progress, axis=0, edge_order=edge_order)

    path_slopes = np.max(np.abs(dq_ds), axis=0)
    constrained = np.isfinite(velocity_limits) & (path_slopes > 1e-12)
    max_progress_rate = 1.0 / nominal_duration
    if np.any(constrained):
        allowed_rates = velocity_scale * velocity_limits[constrained] / path_slopes[constrained]
        max_progress_rate = min(max_progress_rate, float(np.min(allowed_rates)))

    if profile == "uniform":
        segments = max(1, int(np.ceil(frequency / max_progress_rate)))
        times = np.arange(segments + 1) * dt
        progress = times / times[-1]
        progress_rate = np.full_like(times, 1.0 / times[-1])
    else:
        constant_speed_duration = 1.0 / max_progress_rate
        ramp = min(ramp_time, constant_speed_duration)
        segments = max(2, int(np.ceil((constant_speed_duration + ramp) * frequency)))
        times = np.arange(segments + 1) * dt
        total = times[-1]
        max_rate = 1.0 / (total - ramp)

        progress = np.empty_like(times)
        progress_rate = np.empty_like(times)
        accelerating = times <= ramp
        decelerating = times >= total - ramp
        cruising = ~(accelerating | decelerating)

        u = times[accelerating] / ramp
        progress[accelerating] = max_rate * ramp * (u**3 - 0.5 * u**4)
        progress_rate[accelerating] = max_rate * (3.0 * u**2 - 2.0 * u**3)
        progress[cruising] = max_rate * (times[cruising] - 0.5 * ramp)
        progress_rate[cruising] = max_rate
        u = (total - times[decelerating]) / ramp
        progress[decelerating] = 1.0 - max_rate * ramp * (u**3 - 0.5 * u**4)
        progress_rate[decelerating] = max_rate * (3.0 * u**2 - 2.0 * u**3)
        progress[0], progress[-1] = 0.0, 1.0
        progress_rate[0], progress_rate[-1] = 0.0, 0.0

    progress = np.clip(progress, 0.0, 1.0)
    positions = interpolate_path(path, progress)
    velocities = interpolate_path(dq_ds, progress) * progress_rate[:, None]
    return times, positions, velocities


def save_trajectory(
    path: Path, times: np.ndarray, positions: np.ndarray, velocities: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joint_count = positions.shape[1]
    header = ["time_s"]
    header += [f"q{i}_rad" for i in range(1, joint_count + 1)]
    header += [f"dq{i}_rad_s" for i in range(1, joint_count + 1)]
    np.savetxt(
        path, np.column_stack((times, positions, velocities)),
        fmt="%.9f", delimiter="\t", header="\t".join(header), comments="",
    )


def main() -> int:
    args = parse_args()
    model, frame_id, lower, upper, velocity_limits = build_model(args.urdf, args.tcp_frame)
    if args.joint_velocity_limits is not None:
        velocity_limits = np.asarray(args.joint_velocity_limits, dtype=float)
        if not np.isfinite(velocity_limits).all() or np.any(velocity_limits <= 0.0):
            raise ValueError("joint-velocity-limits 必须全部为有限正数。")

    poses = load_poses(args.poses.resolve(), args.translation_scale)
    positions, residuals = solve_trajectory(
        model, frame_id, poses, lower, upper, args.initial_q,
        args.max_iterations, args.tolerance,
    )
    times, positions, velocities = time_parameterize(
        positions, args.frequency, args.speed_profile, args.ramp_time,
        velocity_limits, args.joint_velocity_scale,
    )
    output_path = args.output.resolve()
    save_trajectory(output_path, times, positions, velocities)

    peak_ratio = np.max(np.abs(velocities) / velocity_limits)
    print(f"模型：{model.name}（{args.urdf.resolve()}）")
    print(f"TCP frame：{args.tcp_frame}，位姿数：{len(poses)}")
    print(f"最大 SE(3) 误差：{residuals.max():.3e}")
    print(f"最大关节速度/限速：{peak_ratio:.2%}")
    print(f"输出：{output_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        sys.exit(1)
