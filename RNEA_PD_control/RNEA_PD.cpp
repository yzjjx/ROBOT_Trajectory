#include "RNEA_PD.h"

#include <cmath>
#include <stdexcept>
#include <utility>

#include <Eigen/Core>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/parsers/urdf.hpp>

namespace
{
pinocchio::Model load_model(const std::string& urdf_path)
{
    if (urdf_path.empty())
    {
        throw std::invalid_argument("URDF path cannot be empty.");
    }

    pinocchio::Model model;
    try
    {
        pinocchio::urdf::buildModel(urdf_path, model);
    }
    catch (const std::exception& error)
    {
        throw std::runtime_error(
            "Failed to load URDF '" + urdf_path + "': " + error.what());
    }

    if (model.nq != RNEA_PD::DOF || model.nv != RNEA_PD::DOF)
    {
        throw std::runtime_error(
            "ROKAE SR4 model must have nq=nv=6, but the loaded model has nq=" +
            std::to_string(model.nq) + " and nv=" + std::to_string(model.nv) + ".");
    }

    return model;
}

void validate_gains(
    const RNEA_PD::JointVector& kp,
    const RNEA_PD::JointVector& kd)
{
    for (int index = 0; index < RNEA_PD::DOF; ++index)
    {
        if (!std::isfinite(kp[index]) || !std::isfinite(kd[index]))
        {
            throw std::invalid_argument("PD gains must be finite.");
        }
        if (kp[index] < 0.0 || kd[index] < 0.0)
        {
            throw std::invalid_argument("PD gains must be non-negative.");
        }
    }
}

void validate_state(
    const RNEA_PD::JointVector& values,
    const char* state_name)
{
    for (double value : values)
    {
        if (!std::isfinite(value))
        {
            throw std::invalid_argument(
                std::string(state_name) + " must contain only finite values.");
        }
    }
}
} // namespace

struct RNEA_PD::Impl
{
    explicit Impl(const std::string& urdf_path)
        : model(load_model(urdf_path)), data(model)
    {
    }

    pinocchio::Model model;
    pinocchio::Data data;
};

RNEA_PD::RNEA_PD(
    const JointVector& kp,
    const JointVector& kd,
    const std::string& urdf_path)
    : impl_(new Impl(urdf_path)), Kp_(kp), Kd_(kd)
{
    validate_gains(Kp_, Kd_);
}

RNEA_PD::~RNEA_PD() = default;
RNEA_PD::RNEA_PD(RNEA_PD&&) noexcept = default;
RNEA_PD& RNEA_PD::operator=(RNEA_PD&&) noexcept = default;

RNEA_PD::JointVector RNEA_PD::RNEA_PD_control(
    const JointVector& desired_position,
    const JointVector& desired_velocity,
    const JointVector& desired_acceleration,
    const JointVector& current_position,
    const JointVector& current_velocity)
{
    validate_state(desired_position, "desired_position");
    validate_state(desired_velocity, "desired_velocity");
    validate_state(desired_acceleration, "desired_acceleration");
    validate_state(current_position, "current_position");
    validate_state(current_velocity, "current_velocity");

    using Vector6 = Eigen::Matrix<double, DOF, 1>;
    const Eigen::Map<const Vector6> q_desired(desired_position.data());
    const Eigen::Map<const Vector6> dq_desired(desired_velocity.data());
    const Eigen::Map<const Vector6> ddq_desired(desired_acceleration.data());

    // RNEA提供期望轨迹的惯性、科氏/离心和重力力矩前馈。
    const Eigen::VectorXd feedforward_torque = pinocchio::rnea(
        impl_->model,
        impl_->data,
        q_desired,
        dq_desired,
        ddq_desired);

    JointVector commanded_torque{};
    for (int index = 0; index < DOF; ++index)
    {
        const double position_error =
            desired_position[index] - current_position[index];
        const double velocity_error =
            desired_velocity[index] - current_velocity[index];

        commanded_torque[index] =
            feedforward_torque[index] +
            Kp_[index] * position_error +
            Kd_[index] * velocity_error;
    }

    return commanded_torque;
}
