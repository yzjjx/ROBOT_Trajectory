#ifndef RNEA_PD_H
#define RNEA_PD_H

#include <array>
#include <memory>
#include <string>

// ROKAE SR4关节空间力矩前馈PD控制器。
// 控制律：tau = RNEA(q_d, dq_d, ddq_d)
//             + Kp * (q_d - q) + Kd * (dq_d - dq)。
// 创建一个类class，名称为RNEA_PD，包含以下成员函数和成员变量：
class RNEA_PD
{
public:
    static constexpr int DOF = 6;
    using JointVector = std::array<double, DOF>;
    // 构造函数，用于创建函数时的初始化
    RNEA_PD(
        const JointVector& kp,
        const JointVector& kd,
        const std::string& urdf_path =
            "E:/CODE/20260813_ROBOT_tWrajectory_tracking/urdf/ROKAE_SR4.urdf");

    // 析构函数，用于对象被销毁时做清理工作
    ~RNEA_PD();

    // 下面四行有没有无所谓
    RNEA_PD(const RNEA_PD&) = delete;
    RNEA_PD& operator=(const RNEA_PD&) = delete;
    RNEA_PD(RNEA_PD&&) noexcept;
    RNEA_PD& operator=(RNEA_PD&&) noexcept;

    // 输入单位：位置rad，速度rad/s，加速度rad/s^2；输出力矩N*m。
    JointVector RNEA_PD_control(
        const JointVector& desired_position,
        const JointVector& desired_velocity,
        const JointVector& desired_acceleration,
        const JointVector& current_position,
        const JointVector& current_velocity);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    JointVector Kp_;
    JointVector Kd_;
};

#endif
