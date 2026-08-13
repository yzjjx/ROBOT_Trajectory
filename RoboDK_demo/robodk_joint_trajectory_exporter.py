# -*- coding: utf-8 -*-
"""
RoboDK 关节轨迹采样导出工具
功能：
1. 连接当前运行的 RoboDK
2. 读取当前工作站中的 RoboDK Program
3. 按用户指定采样频率进行时间采样
4. 导出 time + q1...qN + dq1...dqN 到 TXT
   dq 由导出的关节角 q 对时间 t 进行数值微分得到

依赖：
    pip install robodk

注意：
- 采样对象必须是 RoboDK 的“普通 Program”（MoveJ/MoveL/MoveC 等指令组成），
  不是 Python Program。
- RoboDK 的 InstructionListJoints 时间采样依赖程序中的速度/加速度设置。
"""

import math
import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    from robodk.robolink import (
        Robolink,
        ITEM_TYPE_PROGRAM,
        ITEM_TYPE_ROBOT,
        COLLISION_OFF,
        InstructionListJointsFlags,
    )
except Exception as exc:
    raise SystemExit(
        "无法导入 RoboDK Python API。\n"
        "请先执行：pip install robodk\n"
        f"原始错误：{exc}"
    )


class RoboDKTrajectoryExporter(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RoboDK 关节轨迹采样导出工具")
        self.geometry("760x520")
        self.minsize(720, 480)

        self.RDK = None
        self.program_map = {}

        self.freq_var = tk.StringVar(value="1000")
        self.unit_var = tk.StringVar(value="deg")
        self.program_var = tk.StringVar()
        self.output_var = tk.StringVar(
            value=os.path.join(os.getcwd(), "joint_trajectory.txt")
        )
        self.status_var = tk.StringVar(value="尚未连接 RoboDK")
        self.header_var = tk.BooleanVar(value=True)

        self._build_ui()
        self.after(300, self.connect_robodk)

    def _build_ui(self):
        main = ttk.Frame(self, padding=16)
        main.pack(fill="both", expand=True)

        title = ttk.Label(
            main,
            text="RoboDK 关节轨迹采样导出",
            font=("Microsoft YaHei UI", 16, "bold"),
        )
        title.pack(anchor="w", pady=(0, 14))

        conn = ttk.LabelFrame(main, text="1. RoboDK 连接与程序选择", padding=12)
        conn.pack(fill="x", pady=(0, 12))

        row = ttk.Frame(conn)
        row.pack(fill="x")
        ttk.Button(row, text="连接 RoboDK", command=self.connect_robodk).pack(side="left")
        ttk.Button(row, text="刷新程序", command=self.refresh_programs).pack(side="left", padx=8)
        ttk.Label(row, textvariable=self.status_var).pack(side="left", padx=12)

        row2 = ttk.Frame(conn)
        row2.pack(fill="x", pady=(10, 0))
        ttk.Label(row2, text="RoboDK Program：", width=18).pack(side="left")
        self.program_combo = ttk.Combobox(
            row2,
            textvariable=self.program_var,
            state="readonly",
            width=50,
        )
        self.program_combo.pack(side="left", fill="x", expand=True)

        params = ttk.LabelFrame(main, text="2. 采样参数", padding=12)
        params.pack(fill="x", pady=(0, 12))

        p1 = ttk.Frame(params)
        p1.pack(fill="x")
        ttk.Label(p1, text="采样频率：", width=18).pack(side="left")
        ttk.Entry(p1, textvariable=self.freq_var, width=14).pack(side="left")
        ttk.Label(p1, text="Hz").pack(side="left", padx=(5, 20))
        ttk.Label(p1, text="例如：1000 Hz = 1 ms/点").pack(side="left")

        p2 = ttk.Frame(params)
        p2.pack(fill="x", pady=(10, 0))
        ttk.Label(p2, text="关节角单位：", width=18).pack(side="left")
        ttk.Radiobutton(p2, text="degree (°)", variable=self.unit_var, value="deg").pack(side="left")
        ttk.Radiobutton(p2, text="radian (rad)", variable=self.unit_var, value="rad").pack(side="left", padx=15)
        ttk.Checkbutton(p2, text="TXT 写入表头", variable=self.header_var).pack(side="left", padx=20)

        output = ttk.LabelFrame(main, text="3. 输出文件", padding=12)
        output.pack(fill="x", pady=(0, 12))

        o1 = ttk.Frame(output)
        o1.pack(fill="x")
        ttk.Entry(o1, textvariable=self.output_var).pack(side="left", fill="x", expand=True)
        ttk.Button(o1, text="浏览...", command=self.choose_output).pack(side="left", padx=(8, 0))

        action = ttk.Frame(main)
        action.pack(fill="x", pady=(4, 10))
        self.export_btn = ttk.Button(
            action,
            text="开始采样并导出 TXT",
            command=self.start_export,
        )
        self.export_btn.pack(side="left")

        self.progress = ttk.Progressbar(action, mode="indeterminate", length=260)
        self.progress.pack(side="left", padx=16)

        log_frame = ttk.LabelFrame(main, text="运行日志", padding=8)
        log_frame.pack(fill="both", expand=True)
        self.log = tk.Text(log_frame, height=9, wrap="word")
        self.log.pack(fill="both", expand=True)
        self.log.configure(state="disabled")

    def log_msg(self, msg):
        def _write():
            self.log.configure(state="normal")
            self.log.insert("end", msg.rstrip() + "\n")
            self.log.see("end")
            self.log.configure(state="disabled")
        self.after(0, _write)

    def connect_robodk(self):
        try:
            self.status_var.set("正在连接...")
            self.update_idletasks()
            self.RDK = Robolink()
            self.status_var.set("已连接 RoboDK")
            self.log_msg("已连接 RoboDK。")
            self.refresh_programs()
        except Exception as exc:
            self.RDK = None
            self.status_var.set("连接失败")
            self.log_msg(f"连接失败：{exc}")

    def refresh_programs(self):
        if self.RDK is None:
            self.connect_robodk()
            return

        try:
            programs = self.RDK.ItemList(ITEM_TYPE_PROGRAM, False)
            self.program_map.clear()

            names = []
            for prog in programs:
                if prog.Valid():
                    name = prog.Name()
                    names.append(name)
                    self.program_map[name] = prog

            self.program_combo["values"] = names

            if names:
                if self.program_var.get() not in names:
                    self.program_var.set(names[0])
                self.log_msg(f"找到 {len(names)} 个 RoboDK Program。")
            else:
                self.program_var.set("")
                self.log_msg("当前工作站中未找到 RoboDK Program。")
        except Exception as exc:
            self.log_msg(f"刷新程序失败：{exc}")

    @staticmethod
    def calculate_joint_velocity(rows):
        """
        根据 (time, q) 轨迹计算关节速度 dq/dt。

        内点使用中心差分：
            dq[k] = (q[k+1] - q[k-1]) / (t[k+1] - t[k-1])

        首点使用前向差分，末点使用后向差分。

        单位保持与 q 一致：
            q=deg -> dq=deg/s
            q=rad -> dq=rad/s
        """
        n = len(rows)
        if n == 0:
            return []

        if n == 1:
            return [(rows[0][0], [0.0] * len(rows[0][1]))]

        velocities = []

        # 首点：前向差分
        t0, q0 = rows[0]
        t1, q1 = rows[1]
        dt = t1 - t0
        if dt <= 0:
            raise RuntimeError("轨迹时间不是严格递增，无法计算关节速度 dq。")
        dq0 = [(q1[j] - q0[j]) / dt for j in range(len(q0))]
        velocities.append((t0, dq0))

        # 中间点：中心差分
        for k in range(1, n - 1):
            tm, qm = rows[k - 1]
            tc, qc = rows[k]
            tp, qp = rows[k + 1]

            dt_center = tp - tm
            if dt_center <= 0:
                raise RuntimeError("轨迹时间不是严格递增，无法计算关节速度 dq。")

            dq = [
                (qp[j] - qm[j]) / dt_center
                for j in range(len(qc))
            ]
            velocities.append((tc, dq))

        # 末点：后向差分
        tn, qn = rows[-1]
        tp, qp = rows[-2]
        dt = tn - tp
        if dt <= 0:
            raise RuntimeError("轨迹时间不是严格递增，无法计算关节速度 dq。")
        dqn = [(qn[j] - qp[j]) / dt for j in range(len(qn))]
        velocities.append((tn, dqn))

        return velocities

    def choose_output(self):
        path = filedialog.asksaveasfilename(
            title="保存关节轨迹 TXT",
            defaultextension=".txt",
            filetypes=[("Text file", "*.txt"), ("All files", "*.*")],
            initialfile="joint_trajectory.txt",
        )
        if path:
            self.output_var.set(path)

    def start_export(self):
        try:
            freq = float(self.freq_var.get())
            if not math.isfinite(freq) or freq <= 0:
                raise ValueError
            if freq > 10000:
                if not messagebox.askyesno(
                    "高采样频率",
                    f"你设置了 {freq:g} Hz，轨迹点数量可能非常大。\n是否继续？",
                ):
                    return
        except Exception:
            messagebox.showerror("参数错误", "采样频率必须是大于 0 的有效数字。")
            return

        if self.RDK is None:
            messagebox.showerror("未连接", "请先连接 RoboDK。")
            return

        program_name = self.program_var.get().strip()
        if not program_name or program_name not in self.program_map:
            messagebox.showerror("未选择程序", "请选择一个 RoboDK Program。")
            return

        output_path = self.output_var.get().strip()
        if not output_path:
            messagebox.showerror("输出路径错误", "请选择 TXT 输出路径。")
            return

        if not output_path.lower().endswith(".txt"):
            output_path += ".txt"
            self.output_var.set(output_path)

        self.export_btn.configure(state="disabled")
        self.progress.start(10)

        th = threading.Thread(
            target=self.export_worker,
            args=(program_name, freq, output_path, self.unit_var.get(), self.header_var.get()),
            daemon=True,
        )
        th.start()

    def export_worker(self, program_name, freq, output_path, unit, write_header):
        try:
            prog = self.program_map[program_name]
            if not prog.Valid():
                raise RuntimeError("所选 Program 已失效，请刷新后重新选择。")

            # 获取该 Program 关联的机器人
            robot = prog.getLink(ITEM_TYPE_ROBOT)
            if not robot.Valid():
                raise RuntimeError("无法获取该 Program 对应的机器人。")

            n_dof = len(robot.Joints().list())
            if n_dof <= 0:
                raise RuntimeError("无法识别机器人自由度。")

            dt = 1.0 / freq

            self.log_msg(f"程序：{program_name}")
            self.log_msg(f"机器人：{robot.Name()}，自由度：{n_dof}")
            self.log_msg(f"采样频率：{freq:g} Hz，时间步长：{dt:.9f} s")
            self.log_msg("正在调用 RoboDK InstructionListJoints 进行时间采样...")

            # TimeBased = 4
            # mm_step、deg_step在时间采样模式下主要用于内部路径计算精度；
            # 这里使用较小的 1 mm / 1 deg。
            try:
                flag = int(InstructionListJointsFlags.TimeBased)
            except Exception:
                flag = 4

            status_msg, joint_list, status_code = prog.InstructionListJoints(
                mm_step=1.0,
                deg_step=1.0,
                save_to_file=None,
                collision_check=COLLISION_OFF,
                flags=flag,
                time_step=dt,
            )

            if status_code < 0:
                raise RuntimeError(
                    "RoboDK 轨迹计算失败。\n"
                    f"status_code = {status_code}\n"
                    f"message = {status_msg}"
                )

            if joint_list is None or len(joint_list) == 0:
                raise RuntimeError("RoboDK 没有返回轨迹点。")

            # 官方格式在 TimeBased 模式下：
            # [J1..Jn, ERROR, MM_STEP, DEG_STEP, MOVE_ID, TIME, ...]
            # 每个 jnts 对应一个采样点（矩阵的一列）
            rows = []
            cumulative_time = 0.0

            for index, jnts in enumerate(joint_list):
                vals = list(jnts)
                if len(vals) < n_dof:
                    continue

                q_deg = [float(v) for v in vals[:n_dof]]

                # TimeBased 模式在 nDOF+4 位置提供时间信息。
                # RoboDK 示例将其作为相邻采样点的时间增量累加。
                if len(vals) > n_dof + 4:
                    step_time = float(vals[n_dof + 4])
                    if math.isfinite(step_time) and step_time >= 0:
                        cumulative_time += step_time
                    else:
                        cumulative_time = index * dt
                else:
                    cumulative_time = index * dt

                if unit == "rad":
                    q = [math.radians(v) for v in q_deg]
                else:
                    q = q_deg

                rows.append((cumulative_time, q))

            if not rows:
                raise RuntimeError("没有获得有效关节轨迹点。")

            # 为了让第一行严格从 t=0 开始，将所有时间整体平移。
            t0 = rows[0][0]
            rows = [(t - t0, q) for t, q in rows]

            # 根据 q(t) 数值微分计算关节速度 dq(t)
            velocity_rows = self.calculate_joint_velocity(rows)

            if len(velocity_rows) != len(rows):
                raise RuntimeError("关节速度 dq 计算失败：数据长度不一致。")

            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

            with open(output_path, "w", encoding="utf-8", newline="\n") as f:
                if write_header:
                    unit_text = "deg" if unit == "deg" else "rad"
                    velocity_unit_text = "deg_s" if unit == "deg" else "rad_s"

                    header = (
                        ["time_s"]
                        + [f"q{i+1}_{unit_text}" for i in range(n_dof)]
                        + [f"dq{i+1}_{velocity_unit_text}" for i in range(n_dof)]
                    )
                    f.write("\t".join(header) + "\n")

                for (t, q), (_, dq) in zip(rows, velocity_rows):
                    values = (
                        [f"{t:.9f}"]
                        + [f"{v:.9f}" for v in q]
                        + [f"{v:.9f}" for v in dq]
                    )
                    f.write("\t".join(values) + "\n")

            duration = rows[-1][0] if rows else 0.0
            self.log_msg(f"导出完成：{len(rows)} 个轨迹点")
            self.log_msg(f"轨迹时长：{duration:.6f} s")
            self.log_msg("已计算关节速度 dq：由 q(t) 数值微分得到")
            self.log_msg(f"文件：{output_path}")

            self.after(
                0,
                lambda: messagebox.showinfo(
                    "导出完成",
                    f"成功导出 {len(rows)} 个轨迹点。\n\n"
                    f"采样频率：{freq:g} Hz\n"
                    f"轨迹时长：{duration:.6f} s\n\n"
                    f"保存位置：\n{output_path}",
                ),
            )

        except Exception as exc:
            self.log_msg(f"导出失败：{exc}")
            self.after(0, lambda e=str(exc): messagebox.showerror("导出失败", e))
        finally:
            self.after(0, self._finish_export)

    def _finish_export(self):
        self.progress.stop()
        self.export_btn.configure(state="normal")


if __name__ == "__main__":
    app = RoboDKTrajectoryExporter()
    app.mainloop()
