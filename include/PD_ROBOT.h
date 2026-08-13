#ifndef PD_ROBOT_H
#define PD_ROBOT_H

#include <array>

class PD_ROBOT
{
public:
    static constexpr int DOF = 6;

    PD_ROBOT(
        const std::array<double, DOF>& kp,
        const std::array<double, DOF>& kd);

    std::array<double, DOF> PD_control(
        const std::array<double, DOF>& desired_position,
        const std::array<double, DOF>& current_position,
        const std::array<double, DOF>& desired_velocity,
        const std::array<double, DOF>& current_velocity);

private:
    std::array<double, DOF> Kp;
    std::array<double, DOF> Kd;
};

#endif