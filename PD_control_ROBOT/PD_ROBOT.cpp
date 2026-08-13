// 这个代码用来实现机器人的PD控制算法，主要实现控制律即可
#include "PD_ROBOT.h"

// 函数输入：期望位置，实际位置，期望速度，实际速度,函数输出为控制力矩，这个代码用pybind11导入到python中，供python调用
PD_ROBOT::PD_ROBOT(
    const std::array<double, DOF>& kp,
    const std::array<double, DOF>& kd)
    : Kp(kp), Kd(kd)
{
}

std::array<double, PD_ROBOT::DOF> PD_ROBOT::PD_control(
    const std::array<double, DOF>& desired_position,
    const std::array<double, DOF>& current_position,
    const std::array<double, DOF>& desired_velocity,
    const std::array<double, DOF>& current_velocity)
{
    std::array<double, DOF> torque{};

    for (int i = 0; i < DOF; ++i)
    {
        // 位置误差
        double position_error = desired_position[i] - current_position[i];

        // 速度误差
        double velocity_error = desired_velocity[i] - current_velocity[i];

        // PD控制
        torque[i] = Kp[i] * position_error + Kd[i] * velocity_error;
    }

    return torque;
}