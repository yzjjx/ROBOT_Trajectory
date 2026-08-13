#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "PD_ROBOT.h"

namespace py = pybind11;

using Array6 = std::array<double, PD_ROBOT::DOF>;

PYBIND11_MODULE(PD_ROBOT, m)
{
    m.doc() = "C++ PD Controller for ROKAE_SR4";

    py::class_<PD_ROBOT>(m, "PD_ROBOT")
        .def(
            py::init<const Array6&, const Array6&>(),
            py::arg("kp"),
            py::arg("kd")
        )
        .def(
            "PD_control",
            &PD_ROBOT::PD_control,
            py::arg("desired_position"),
            py::arg("current_position"),
            py::arg("desired_velocity"),
            py::arg("current_velocity")
        );
}