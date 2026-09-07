"""使用 URDF 和 Pinocchio 搜索机器人的不同数值 IK 分支。

多初值数值搜索不能从数学上保证找到全部解析解，但增加网格密度和随机初值
数量可以降低漏解概率。
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

from Tra_generate_Pinocchio_DH import build_model, pin, solve_pose


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 更换机器人时，只需修改下面两项；也可以用 --urdf 和 --tcp-frame 临时指定。
DEFAULT_URDF_FILE = PROJECT_ROOT / "urdf" / "ER400_mdh_verified.urdf"
DEFAULT_TCP_FRAME = "ER_link6"


# 目标 TCP 位姿：平移单位 m。
TARGET_TCP = np.array(
    [
        [0.0, 0.0, 1.0, 2.276],
        [0.0, 1.0, 0.0, 0.9],
        [-1.0, 0.0, 0.0, 1.880],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 URDF + Pinocchio 搜索机器人的不同 IK 分支。"
    )
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF_FILE)
    parser.add_argument("--tcp-frame", default=DEFAULT_TCP_FRAME)
    parser.add_argument(
        "--initial-q", type=float, nargs="+",
        help="可选的优先 IK 初值，单位 rad；数量必须等于 URDF 关节数。",
    )
    parser.add_argument(
        "--grid-points",
        type=int,
        default=3,
        help="每个关节的网格初值数；六轴机器人取 3 时共有 729 个初值。",
    )
    parser.add_argument(
        "--random-starts",
        type=int,
        default=500,
        help="每轮随机初值数，默认 500。",
    )
    parser.add_argument("--random-rounds", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument(
        "--cluster-tolerance-deg",
        type=float,
        default=0.05,
        help="分支去重的单关节角度阈值。",
    )
    parser.add_argument("--seed", type=int, default=20260830)
    return parser.parse_args()


def validate_settings(model, args: argparse.Namespace) -> None:
    if TARGET_TCP.shape != (4, 4):
        raise ValueError("TARGET_TCP 必须是 4×4 矩阵。")
    if not np.allclose(TARGET_TCP[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError("TARGET_TCP 最后一行无效。")
    rotation = TARGET_TCP[:3, :3]
    if not np.allclose(np.einsum("ji,jk->ik", rotation, rotation), np.eye(3), atol=1e-9):
        raise ValueError("TARGET_TCP 的旋转矩阵不正交。")
    if args.grid_points < 1 or args.random_starts < 0 or args.random_rounds < 0:
        raise ValueError("搜索次数参数无效。")
    if args.max_iterations <= 0 or args.tolerance <= 0.0:
        raise ValueError("max-iterations 和 tolerance 必须大于 0。")
    if args.cluster_tolerance_deg <= 0.0:
        raise ValueError("cluster-tolerance-deg 必须大于 0。")
    if args.initial_q is not None and len(args.initial_q) != model.nq:
        raise ValueError(f"initial-q 需要 {model.nq} 个数。")


def wrapped_difference(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """转动关节按 2π 等价计算最小角距离。"""
    return (q1 - q2 + np.pi) % (2.0 * np.pi) - np.pi


def canonical_angles(q: np.ndarray) -> np.ndarray:
    """将解映射到 [-π, π)，便于打印和去重。"""
    return (q + np.pi) % (2.0 * np.pi) - np.pi


def add_if_new(solutions: list[np.ndarray], q: np.ndarray,
               cluster_tolerance: float) -> bool:
    q = canonical_angles(q)
    if any(np.max(np.abs(wrapped_difference(q, old))) < cluster_tolerance
           for old in solutions):
        return False
    solutions.append(q)
    return True


def grid_seeds(points_per_joint: int, lower: np.ndarray, upper: np.ndarray):
    """避开限位端点，在搜索区间内部生成笛卡尔网格。"""
    axes = [
        np.linspace(lower[i], upper[i], points_per_joint + 2)[1:-1]
        for i in range(len(lower))
    ]
    return itertools.product(*axes)


def search_solutions(
    model, frame_id: int, lower: np.ndarray, upper: np.ndarray,
    args: argparse.Namespace,
) -> list[np.ndarray]:
    target = pin.SE3(TARGET_TCP[:3, :3], TARGET_TCP[:3, 3])
    data = model.createData()
    solutions: list[np.ndarray] = []
    cluster_tolerance = np.deg2rad(args.cluster_tolerance_deg)

    def try_seed(seed: np.ndarray) -> bool:
        q, _, converged = solve_pose(
            model,
            data,
            frame_id,
            target,
            np.asarray(seed, dtype=float),
            lower,
            upper,
            args.max_iterations,
            args.tolerance,
        )
        return converged and add_if_new(solutions, q, cluster_tolerance)

    # 先试用户初值和中位姿态，再做确定性网格搜索。
    if args.initial_q is not None:
        try_seed(np.asarray(args.initial_q))
    try_seed(np.zeros(model.nq))

    total_grid = args.grid_points ** model.nq
    for index, seed in enumerate(grid_seeds(args.grid_points, lower, upper), start=1):
        try_seed(np.asarray(seed))
        if index % 100 == 0 or index == total_grid:
            print(f"网格搜索：{index}/{total_grid}，已找到 {len(solutions)} 个分支")

    # 随机搜索连续两轮无新增时提前停止。
    rng = np.random.default_rng(args.seed)
    no_new_rounds = 0
    for round_index in range(1, args.random_rounds + 1):
        before = len(solutions)
        for seed in rng.uniform(lower, upper, size=(args.random_starts, model.nq)):
            try_seed(seed)
        added = len(solutions) - before
        print(f"随机搜索第 {round_index} 轮：新增 {added}，累计 {len(solutions)}")
        no_new_rounds = no_new_rounds + 1 if added == 0 else 0
        if no_new_rounds >= 2:
            break

    solutions.sort(key=lambda q: tuple(q.tolist()))
    return solutions


def solution_errors(model, frame_id: int, q: np.ndarray) -> tuple[float, float]:
    data = model.createData()
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacement(model, data, frame_id)
    actual = data.oMf[frame_id].homogeneous
    position_error_mm = 1e3 * np.linalg.norm(actual[:3, 3] - TARGET_TCP[:3, 3])
    rotation_error = np.einsum(
        "ji,jk->ik", actual[:3, :3], TARGET_TCP[:3, :3]
    )
    cosine = np.clip((np.trace(rotation_error) - 1.0) / 2.0, -1.0, 1.0)
    rotation_error_deg = np.rad2deg(np.arccos(cosine))
    return float(position_error_mm), float(rotation_error_deg)


def print_solutions(model, frame_id: int, solutions: list[np.ndarray]) -> None:
    print(f"\n找到 {len(solutions)} 个不同数值 IK 分支。")
    for index, q in enumerate(solutions, start=1):
        position_error, rotation_error = solution_errors(model, frame_id, q)
        q_rad = ", ".join(f"{value:.9f}" for value in q)
        q_deg = ", ".join(f"{value:.6f}" for value in np.rad2deg(q))
        print(f"\n解 {index}")
        print(f"q [rad] = [{q_rad}]")
        print(f"q [deg] = [{q_deg}]")
        print(f"位置误差 = {position_error:.3e} mm")
        print(f"姿态误差 = {rotation_error:.3e} deg")


def main() -> int:
    args = parse_args()
    model, frame_id, lower, upper, _ = build_model(args.urdf, args.tcp_frame)
    validate_settings(model, args)

    print(f"模型：{model.name}（{args.urdf.resolve()}）")
    print(f"TCP frame：{args.tcp_frame}")
    print(f"目标位置 [m]：{TARGET_TCP[:3, 3]}")
    print(f"URDF 关节下限 [deg]：{np.rad2deg(lower)}")
    print(f"URDF 关节上限 [deg]：{np.rad2deg(upper)}")

    solutions = search_solutions(model, frame_id, lower, upper, args)
    print_solutions(model, frame_id, solutions)
    if not solutions:
        raise RuntimeError("没有找到 IK 解，请检查目标位姿、URDF 和 TCP frame。")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        sys.exit(1)
