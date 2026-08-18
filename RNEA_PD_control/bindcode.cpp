#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "RNEA_PD.h"

namespace py = pybind11;

using Array6 = std::array<double, RNEA_PD::DOF>;

PYBIND11_MODULE(RNEA_PD, m)
{
    m.doc() = "Pinocchio RNEA feedforward plus PD controller for ROKAE SR4";

    py::class_<RNEA_PD>(m, "RNEA_PD")
        .def(
            py::init<const Array6&, const Array6&, const std::string&>(),
            py::arg("kp"),
            py::arg("kd"),
            py::arg("urdf_path") =
                "E:/CODE/20260813_ROBOT_trajectory_tracking/urdf/ROKAE_SR4.urdf"
        )
        .def(
            "RNEA_PD_control",
            &RNEA_PD::RNEA_PD_control,
            py::arg("desired_position"),
            py::arg("desired_velocity"),
            py::arg("desired_acceleration"),
            py::arg("current_position"),
            py::arg("current_velocity")
        );
}
