% 激光跟踪仪坐标系 {L} 到机器人基坐标系 {B} 的最小二乘标定
%
% 每一行数据必须具有相同时间戳：
%   p_LS(i,:)      : 激光仪测得的靶球 S 在 {L} 中的位置，单位 m
%   T_BF(:,:,i)    : 同一时刻机器人法兰 {F} 相对基座 {B} 的齐次矩阵
%
% 待辨识参数：
%   p_FS : 靶球 S 相对机器人法兰 {F} 的位置
%   p_BL : 激光仪原点 {L} 相对机器人基座 {B} 的位置
%   R_BL : 激光仪 {L} 相对机器人基座 {B} 的姿态
%
% 对每一个采样点，有：
% p_BF + R_BF * p_FS = p_BL + R_BL * p_LS
%
% 本程序不需要额外工具箱：外层用 fminsearch 求旋转，
% 对固定旋转时，两个平移量 p_FS 和 p_BL 用最小二乘 A\b 直接求解。

clear; clc;

%% 1. 输入同步后的标定数据
% 激光跟踪仪测得的靶球位置。每行对应一个时间戳，单位 mm。
p_LS = [-868.713 3242.436 226.060;
        -596.038 3318.551 285.840;
        -812.126 3427.623 112.462;
        -1117.109 3372.820 175.357;
        -668.431 3438.188 310.100;
        -851.142 3364.815 486.230]/1000;

% 请将下面的空数组替换为真实机器人位姿。
% T_BF 必须是 4 x 4 x N，N 等于 p_LS 的行数。
% 每一页的格式为：T_BF(:,:,i) = [R_BF, p_BF; 0 0 0 1]。
%
% 例如：
% T_BF(:,:,1) = [1 0 0 0.500;
%                0 1 0 0.100;
%                0 0 1 0.400;
%                0 0 0 1];
T_BF = [];
T_BF(:,:,1) =  [0.00639477,-0.776999,0.629469,0;
                -0.00678059,-0.629501,-0.77697,0;
                0.999957,0.00070037,-0.00929404,0;
                0.438155,-0.0113186,0.806764,1].';
T_BF(:,:,2) =  [0.0232009,-0.441015,0.8972,0;
                -0.0057749,-0.897486,-0.441006,0;
                0.999714,0.00505047,-0.0233693,0;
                0.385939,0.260372,0.840199,1].';
T_BF(:,:,3) =  [0.0204185,-0.913297,0.406783,0;
                -0.025133,-0.407208,-0.91299,0;
                0.999476,0.00841822,-0.0312684,0;
                0.259979,0.0570816,0.709255,1].';
T_BF(:,:,4) =  [0.0183502,-0.87755,0.479133,0;
                -0.032372,-0.479484,-0.876953,0;
                0.999307,0.000581728,-0.0372067,0;
                0.288565,-0.248124,0.765132,1].';
T_BF(:,:,5) =  [0.0330443,0.628708,0.776939,0;
                0.038111,-0.777591,0.627614,0;
                0.998727,0.00887085,-0.0496556,0;
                0.259192,0.14047,0.818704,1].';
T_BF(:,:,6) =  [0.0454844,-0.220182,0.974398,0;
                -0.00483869,-0.975444,-0.220193,0;
                0.998953,0.00530053,-0.0454329,0;
                0.315074,0.00354712,1.02355,1].';

% 检查输入数据是否合理
if isempty(T_BF)
    error('请在程序第 40 行附近填入真实的 T_BF 数据。');
end

N = size(p_LS, 1);
if size(T_BF, 1) ~= 4 || size(T_BF, 2) ~= 4 || size(T_BF, 3) ~= N
    error('T_BF 的尺寸必须是 4 x 4 x N，且 N 等于 p_LS 的行数。');
end

% 检查所有输入位姿是否为有效的刚体齐次变换
for i = 1:N
    R = T_BF(1:3, 1:3, i);
    if norm(R.' * R - eye(3), 'fro') > 1e-4 || abs(det(R)-1) > 1e-4
        error('T_BF(:,:, %d) 的旋转矩阵不是有效旋转矩阵。', i);
    end
    if norm(T_BF(4, :, i) - [0 0 0 1]) > 1e-9
        error('T_BF(:,:, %d) 的最后一行必须为 [0 0 0 1]。', i);
    end
end

%% 2. 最小二乘辨识
% 旋转 R_BL 用 Z-Y-X 欧拉角 [rz, ry, rx] 表示，单位为 rad。
% 使用多个初值可降低局部极小值的影响。
initial_angles = [0   0   0;
                  pi  0   0;
                  0   pi  0;
                  0   0   pi];

best_cost = inf;
best_angles = [];
options = optimset('Display', 'off', 'MaxIter', 5000, 'MaxFunEvals', 10000);

for i = 1:size(initial_angles, 1)
    angles = fminsearch(@(x) calibration_cost(x, T_BF, p_LS), ...
                         initial_angles(i, :), options);
    cost = calibration_cost(angles, T_BF, p_LS);
    if cost < best_cost
        best_cost = cost;
        best_angles = angles;
    end
end

R_BL = rotation_zyx(best_angles);
[p_FS, p_BL] = solve_translations(R_BL, T_BF, p_LS);

%% 3. 计算每个点的坐标转换误差
p_BS_robot = zeros(N, 3);
p_BS_laser = zeros(N, 3);

for i = 1:N
    R_BF = T_BF(1:3, 1:3, i);
    p_BF = T_BF(1:3, 4, i);

    % 从机器人位姿计算靶球在 {B} 中的位置。
    p_BS_robot(i, :) = (p_BF + R_BF * p_FS).';

    % 将激光仪测得的点转换到机器人基坐标系 {B}。
    p_BS_laser(i, :) = (p_BL + R_BL * p_LS(i, :).').';
end

error_xyz = p_BS_laser - p_BS_robot;
error_norm = sqrt(sum(error_xyz.^2, 2));

%% 4. 输出辨识结果
fprintf('\n===== 最小二乘标定结果 =====\n');
fprintf('p_FS (m) = [%.6f, %.6f, %.6f]^T\n', p_FS);
fprintf('p_BL (m) = [%.6f, %.6f, %.6f]^T\n', p_BL);
fprintf('R_BL = \n');
disp(R_BL);
fprintf('RMS position error = %.6f m\n', sqrt(mean(error_norm.^2)));
fprintf('Max position error = %.6f m\n', max(error_norm));

% p_BS_laser 即为所有激光跟踪仪数据转换到机器人基坐标系后的 XYZ。
% 可将它与 MATLAB_code/Trajectory_show.m 生成的理想轨迹进行时间同步比较。

%% ===== 本文件使用的局部函数 =====
function cost = calibration_cost(angles, T_BF, p_LS)
    R_BL = rotation_zyx(angles);
    [p_FS, p_BL] = solve_translations(R_BL, T_BF, p_LS);

    N = size(p_LS, 1);
    residual = zeros(3 * N, 1);
    for k = 1:N
        R_BF = T_BF(1:3, 1:3, k);
        p_BF = T_BF(1:3, 4, k);
        difference = p_BF + R_BF * p_FS - p_BL - R_BL * p_LS(k, :).';
        residual(3*k-2 : 3*k) = difference;
    end
    cost = residual.' * residual;
end


function [p_FS, p_BL] = solve_translations(R_BL, T_BF, p_LS)
    % 固定 R_BL 后：R_BF*p_FS - p_BL = R_BL*p_LS - p_BF。
    % 将所有点堆叠成 A*x=b，再使用最小二乘 x=A\b。
    N = size(p_LS, 1);
    A = zeros(3 * N, 6);
    b = zeros(3 * N, 1);

    for k = 1:N
        row = 3*k-2 : 3*k;
        R_BF = T_BF(1:3, 1:3, k);
        p_BF = T_BF(1:3, 4, k);

        A(row, :) = [R_BF, -eye(3)];
        b(row) = R_BL * p_LS(k, :).'- p_BF;
    end

    x = A \ b;
    p_FS = x(1:3);
    p_BL = x(4:6);
end


function R = rotation_zyx(angle)
    % angle = [rz, ry, rx]，R = Rz(rz) * Ry(ry) * Rx(rx)。
    rz = angle(1); ry = angle(2); rx = angle(3);

    Rz = [cos(rz), -sin(rz), 0;
          sin(rz),  cos(rz), 0;
                0,        0, 1];
    Ry = [ cos(ry), 0, sin(ry);
                 0, 1,       0;
          -sin(ry), 0, cos(ry)];
    Rx = [1,       0,        0;
          0, cos(rx), -sin(rx);
          0, sin(rx),  cos(rx)];

    R = Rz * Ry * Rx;
end
