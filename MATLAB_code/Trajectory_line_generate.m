% 根据两个空间点生成机器人的TCP直线轨迹。
% 输出TXT中每四个非空行对应一个4x4齐次位姿矩阵，可直接作为
% trajectory_generate/Tra_generate.py的--poses输入。

%% 可配置参数
% 起点和终点，单位：mm。
% NB4直线
P_start = [331.063710; -8.558820; 319.977830];
P_end   = [331.063710;  31.441180; 319.977830];
% % SR4直线
% P_start = [500; -20; 400];
% P_end   = [500;  20; 400];
% 期望TCP运行速度和轨迹采样频率。
v = 50;       % mm/s
fre = 1000;   % Hz

% TCP相对于基坐标系的固定姿态
posture = [ 0  0  1;
            0  1  0;
           -1  0  0];

output_name = 'line_TCP_poses_NB4_40.txt';

%% 输入检查
if ~isequal(size(P_start), [3, 1]) || ~isequal(size(P_end), [3, 1]) || ...
        any(~isfinite(P_start)) || any(~isfinite(P_end))
    error('P_start和P_end必须是有限数值组成的3x1列向量。');
end

line_vector = P_end - P_start;
line_length = norm(line_vector);
if line_length < 1e-9
    error('P_start和P_end不能是同一个点。');
end
if v <= 0 || fre <= 0
    error('运行速度v和采样频率fre必须大于0。');
end
if norm(posture.' * posture - eye(3), 'fro') > 1e-9 || ...
        abs(det(posture) - 1) > 1e-9
    error('posture必须是有效的3x3旋转矩阵。');
end

%% 按速度和采样频率等距生成完整直线
line_time = line_length / v;
segment_count = max(1, ceil(line_time * fre));
progress = linspace(0, 1, segment_count + 1);
trajectory = P_start + line_vector * progress;
point_count = size(trajectory, 2);
actual_speed = line_length * fre / segment_count;

% 为每个位置生成具有相同TCP姿态的4x4齐次位姿矩阵。
TCP_poses = repmat(eye(4), 1, 1, point_count);
TCP_poses(1:3, 1:3, :) = repmat(posture, 1, 1, point_count);
TCP_poses(1:3, 4, :) = reshape(trajectory, 3, 1, point_count);

%% 写入TXT文件
script_path = mfilename('fullpath');
if isempty(script_path)
    output_dir = pwd;
else
    output_dir = fileparts(script_path);
end
output_file = fullfile(output_dir, output_name);
file_id = fopen(output_file, 'w');
if file_id == -1
    error('无法创建TCP位姿文件：%s', output_file);
end

try
    for i = 1:point_count
        fprintf(file_id, '%.10f %.10f %.10f %.10f\n', TCP_poses(:, :, i).');
        fprintf(file_id, '\n');
    end
    fclose(file_id);
catch ME
    fclose(file_id);
    rethrow(ME);
end

%% 绘制空间直线、端点和TCP坐标系
figure('Color', 'w', 'Name', 'NB4 spatial line and TCP frames');
plot3(trajectory(1, :), trajectory(2, :), trajectory(3, :), ...
      'k-', 'LineWidth', 1.8, 'DisplayName', 'TCP直线轨迹');
hold on;
end_points = [P_start, P_end];
plot3(end_points(1, :), end_points(2, :), end_points(3, :), ...
      'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 7, ...
      'DisplayName', '起点与终点');

axis_length = min(0.15 * line_length, 50);
for i = 1:2
    origin = end_points(:, i);
    quiver3(origin(1), origin(2), origin(3), ...
            posture(1, 1) * axis_length, posture(2, 1) * axis_length, ...
            posture(3, 1) * axis_length, 0, 'r', 'LineWidth', 1.8, ...
            'MaxHeadSize', 0.5, 'HandleVisibility', 'off');
    quiver3(origin(1), origin(2), origin(3), ...
            posture(1, 2) * axis_length, posture(2, 2) * axis_length, ...
            posture(3, 2) * axis_length, 0, 'g', 'LineWidth', 1.8, ...
            'MaxHeadSize', 0.5, 'HandleVisibility', 'off');
    quiver3(origin(1), origin(2), origin(3), ...
            posture(1, 3) * axis_length, posture(2, 3) * axis_length, ...
            posture(3, 3) * axis_length, 0, 'b', 'LineWidth', 1.8, ...
            'MaxHeadSize', 0.5, 'HandleVisibility', 'off');
end
text(P_start(1), P_start(2), P_start(3), '  Start', ...
     'FontSize', 10, 'FontWeight', 'bold');
text(P_end(1), P_end(2), P_end(3), '  End', ...
     'FontSize', 10, 'FontWeight', 'bold');

plot3(nan, nan, nan, 'r-', 'LineWidth', 1.8, 'DisplayName', 'TCP X轴');
plot3(nan, nan, nan, 'g-', 'LineWidth', 1.8, 'DisplayName', 'TCP Y轴');
plot3(nan, nan, nan, 'b-', 'LineWidth', 1.8, 'DisplayName', 'TCP Z轴');

grid on;
axis equal;
view(3);
xlabel('X / mm');
ylabel('Y / mm');
zlabel('Z / mm');
title('NB4两点空间直线轨迹及TCP坐标系');
legend('Location', 'best');

fprintf('起点：[%.4f, %.4f, %.4f] mm\n', P_start);
fprintf('终点：[%.4f, %.4f, %.4f] mm\n', P_end);
fprintf('直线长度：%.4f mm\n', line_length);
fprintf('轨迹点数：%d\n', point_count);
fprintf('实际TCP速度：%.6f mm/s\n', actual_speed);
fprintf('TCP位姿已写入：%s\n', output_file);
