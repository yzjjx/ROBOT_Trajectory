% 理想 TCP 圆轨迹（机器人基坐标系）
% 单位：mm
% 四个点按照 RoboDK 的运动顺序给出：P1 -> P2 -> P3 -> P4 -> P1。

clear; clc; close all;

% 1. RoboDK 中定义的四个 TCP 点，均已转换到机器人基坐标系。
P1 = [641.44;    0; 541.44];
P2 = [500.00; -200; 400.00];
P3 = [358.56;    0; 258.56];
P4 = [500.00;  200; 400.00];
points = [P1, P2, P3, P4];

% 2. 用前三个不共线点计算圆心、圆半径和圆所在平面的法向量。
a = P2 - P1;
b = P3 - P1;
normal = cross(a, b);

if norm(normal) < 1e-9
    error('P1、P2、P3 不能共线。');
end

center = P1 + (cross(normal, a) * dot(b, b) + ...
               cross(b, normal) * dot(a, a)) / (2 * dot(normal, normal));
radius = norm(P1 - center);
normal = normal / norm(normal);

% 3. 在圆所在平面建立二维坐标系，并求四个点的圆心角。
ex = (P1 - center) / radius;
ey = cross(normal, ex);
angle = atan2(ey.' * (points - center), ex.' * (points - center));

% unwrap 使角度按照 P1 -> P2 -> P3 -> P4 -> P1 连续变化。
angle = unwrap([angle, angle(1)]);

% 4. 每两个点之间生成一段圆弧，模拟连续 MoveC 圆弧运动。
ideal_xyz = [];
point_per_arc = 100;
for i = 1:4
    theta = linspace(angle(i), angle(i + 1), point_per_arc);
    arc = center + radius * (ex * cos(theta) + ey * sin(theta));
    ideal_xyz = [ideal_xyz, arc(:, 1:end-1)]; %#ok<AGROW>
end
ideal_xyz = [ideal_xyz, P1];

% 5. 绘制理想 TCP 圆轨迹。
figure('Color', 'w');
plot3(ideal_xyz(1, :), ideal_xyz(2, :), ideal_xyz(3, :), ...
      'b-', 'LineWidth', 2); hold on;
plot3(points(1, :), points(2, :), points(3, :), ...
      'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 7);
plot3(center(1), center(2), center(3), ...
      'kx', 'LineWidth', 2, 'MarkerSize', 10);

for i = 1:4
    text(points(1, i), points(2, i), points(3, i), ...
         ['  P', num2str(i)], 'FontSize', 11);
end

grid on; axis equal; view(3);
xlabel('X / mm'); ylabel('Y / mm'); zlabel('Z / mm');
title('Robot Base Frame: Ideal TCP Circular Trajectory');
legend('Ideal TCP trajectory', 'RoboDK points', 'Circle center', ...
       'Location', 'best');

fprintf('Circle center: [%.2f, %.2f, %.2f] mm\n', center);
fprintf('Circle radius: %.2f mm\n', radius);
