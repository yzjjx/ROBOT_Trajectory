% 理想 TCP 直线轨迹（机器人基坐标系）
% 单位：mm

clear; clc; close all;

% 1. RoboDK 中定义的直线起点和终点，均已转换到机器人基坐标系。
% 若直线点位不同，只需修改这两个 3 x 1 列向量。
P_start = [600; -250; 500];
P_end   = [600; 250; 500];

% 2. 在起点和终点之间均匀生成理想直线点。
point_num = 10501
s = linspace(0, 1, point_num);
ideal_xyz = P_start + (P_end - P_start) * s;

% Ideal_traj 为 N x 3，符合 Trajectory_com.m 的输入格式。
Ideal_traj = ideal_xyz.';

% 3. 导出理想直线轨迹。
script_folder = fileparts(mfilename('fullpath'));
save(fullfile(script_folder, 'Ideal_traj.mat'), 'Ideal_traj');

% 4. 三维显示理想直线轨迹。
figure('Color', 'w');
plot3(Ideal_traj(:, 1), Ideal_traj(:, 2), Ideal_traj(:, 3), ...
      'b-', 'LineWidth', 2); hold on;
plot3(P_start(1), P_start(2), P_start(3), ...
      'go', 'MarkerFaceColor', 'g', 'MarkerSize', 8);
plot3(P_end(1), P_end(2), P_end(3), ...
      'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 8);

grid on; axis equal; view(3);
xlabel('X / mm'); ylabel('Y / mm'); zlabel('Z / mm');
title('Robot Base Frame: Ideal TCP Line Trajectory');
legend('Ideal TCP trajectory', 'Start point', 'End point', ...
       'Location', 'best');

fprintf('Start point: [%.2f, %.2f, %.2f] mm\n', P_start);
fprintf('End point:   [%.2f, %.2f, %.2f] mm\n', P_end);
fprintf('Ideal_traj exported: %s\n', ...
        fullfile(script_folder, 'Ideal_traj.mat'));
