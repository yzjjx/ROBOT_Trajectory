% 三点确定空间圆轨迹，首先输入三点坐标，然后生成空间圆形轨迹
% 可以设置频率与运行速度，方便生成TCP所有点的位置
% 可以设置TCP点相对于基坐标系的姿态（3*3矩阵）

% 三点坐标 SR4 D400
% P1 = [641.44;    0; 541.44];
% P2 = [500.00; -200; 400.00];
% P3 = [500.00;  200; 400.00];
% points = [P1, P2, P3];

% 三点坐标SR4 D40
P1 = [500 + 14.1421;    0; 400 + 14.1421];
P2 = [500; -20; 400]; 
P3 = [500;  20; 400];
points = [P1, P2, P3];

% % NB4斜向空间圆：半径200 mm，圆平面法向量为[1, 0, 1]/sqrt(2)
% P1 = [331.063710; -188.558820; 319.977830];
% P2 = [472.485066237; 11.441180; 178.556473763];
% P3 = [331.063710; 211.441180; 319.977830];
% points = [P1, P2, P3];

% % NB4斜向空间圆：半径20 mm
% % 圆心：[331.063710, 11.441180, 319.977830] mm
% % 圆平面法向量：[1, 0, 1]/sqrt(2)
% P1 = [331.063710;   -8.558820; 319.977830];
% P2 = [345.205845624; 11.441180; 305.835694376];
% P3 = [331.063710;   31.441180; 319.977830];
% points = [P1, P2, P3];

% 采样频率与运行速度
v = 50;% 单位mm/s
fre = 1000;% 单位：Hz

output_txt_name = 'circle_R20_TCP_poses_SR4_V50.txt';

% TCP相对于基坐标系的姿态
posture = [0 0 1;
           0 1 0;
          -1 0 0;];

% 根据三个不共线点计算空间圆的圆心、半径和法向量
a = P2 - P1;
b = P3 - P1;
normal = cross(a, b);
if norm(normal) < 1e-9
    error('P1、P2、P3不能共线。');
end

center = P1 + (cross(normal, a) * dot(b, b) + ...
               cross(b, normal) * dot(a, a)) / (2 * dot(normal, normal));
radius = norm(P1 - center);
normal = normal / norm(normal);

% 在圆平面内建立正交坐标系，使轨迹按P1 -> P2 -> P3的方向运行
ex = (P1 - center) / radius;
ey = cross(normal, ex);
angle_P2 = mod(atan2(ey.' * (P2 - center), ex.' * (P2 - center)), 2*pi);
angle_P3 = mod(atan2(ey.' * (P3 - center), ex.' * (P3 - center)), 2*pi);
if angle_P2 > angle_P3
    ey = -ey;
end

% 根据圆周长度、运行速度和采样频率生成完整圆轨迹
if v <= 0 || fre <= 0
    error('运行速度v和采样频率fre必须大于0。');
end
circle_time = 2*pi*radius / v;
segment_count = max(3, ceil(circle_time * fre));
theta = linspace(0, 2*pi, segment_count + 1);
trajectory = center + radius * (ex * cos(theta) + ey * sin(theta));
Ideal_traj = trajectory.';

% 为圆周每个采样点生成TCP相对于基坐标系的4*4齐次位姿矩阵
if norm(posture.' * posture - eye(3), 'fro') > 1e-9 || ...
        abs(det(posture) - 1) > 1e-9
    error('posture必须是有效的3*3旋转矩阵。');
end

point_count = size(trajectory, 2);
TCP_poses = repmat(eye(4), 1, 1, point_count);
TCP_poses(1:3, 1:3, :) = repmat(posture, 1, 1, point_count);
TCP_poses(1:3, 4, :) = reshape(trajectory, 3, 1, point_count);

% TXT中每四行对应一个采样点的4*4位姿矩阵，矩阵之间用空行分隔
output_dir = 'E:\Y2_1\20260828_Robot_trajectory_tracking_v2\Trajectory_TCP';
if ~exist(output_dir, 'dir')
    [mkdir_ok, mkdir_msg] = mkdir(output_dir);
    if ~mkdir_ok
        error('无法创建输出目录：%s\n%s', output_dir, mkdir_msg);
    end
end
output_file = fullfile(output_dir,output_txt_name);
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

% 在三维坐标系中显示空间圆、三个输入点及其TCP坐标系
figure('Color', 'w', 'Name', 'Circle R200 and TCP frames');
plot3(trajectory(1, :), trajectory(2, :), trajectory(3, :), ...
      'k-', 'LineWidth', 1.8, 'DisplayName', '圆形轨迹');
hold on;
plot3(points(1, :), points(2, :), points(3, :), ...
      'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6, ...
      'DisplayName', 'P1、P2、P3');

axis_length = 0.2 * radius;
for i = 1:size(points, 2)
    origin = points(:, i);
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
    text(origin(1), origin(2), origin(3), ['  P', num2str(i)], ...
         'FontSize', 10, 'FontWeight', 'bold');
end

% 使用不可见对象在图例中标明TCP三轴颜色
plot3(nan, nan, nan, 'r-', 'LineWidth', 1.8, 'DisplayName', 'TCP X轴');
plot3(nan, nan, nan, 'g-', 'LineWidth', 1.8, 'DisplayName', 'TCP Y轴');
plot3(nan, nan, nan, 'b-', 'LineWidth', 1.8, 'DisplayName', 'TCP Z轴');

grid on;
axis equal;
view(3);
xlabel('X / mm');
ylabel('Y / mm');
zlabel('Z / mm');
title('三维圆形轨迹及三个点的TCP坐标系');
legend('Location', 'best');

fprintf('圆心：[%.4f, %.4f, %.4f] mm\n', center);
fprintf('半径：%.4f mm\n', radius);
fprintf('轨迹点数：%d\n', point_count);
fprintf('TCP位姿已写入：%s\n', output_file);
