cmake_minimum_required(VERSION 3.10)
project(ROKAE_MPC)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==================== 【强行注入缓存，彻底封死自动搜索】 ====================
set(Python3_EXECUTABLE "F:/ProgramData/anaconda3/envs/mpc_rokae/python.exe" CACHE FILEPATH "Force Conda Python3" FORCE)
set(PYTHON_EXECUTABLE "F:/ProgramData/anaconda3/envs/mpc_rokae/python.exe" CACHE FILEPATH "Force Conda Python Fallback" FORCE)
set(Python_EXECUTABLE "F:/ProgramData/anaconda3/envs/mpc_rokae/python.exe" CACHE FILEPATH "Force Conda FindPython" FORCE)
set(Python_ROOT_DIR "F:/ProgramData/anaconda3/envs/mpc_rokae" CACHE PATH "Force Conda Python root" FORCE)
set(PYBIND11_FINDPYTHON ON)
# =========================================================================

list(APPEND CMAKE_PREFIX_PATH "F:/ProgramData/anaconda3/envs/mpc_rokae/Library")

find_package(pybind11 REQUIRED)

find_package(Eigen3 REQUIRED)

# 寻找预编译好的Pinocchio及其依赖
find_package(pinocchio CONFIG REQUIRED)

# add_executable(mpc_node 
#     # test/test_dynamics.cpp
#     src/dynamics_com_tau.cpp
#     src/robot_dynamics.c
#     src/Prediction.cpp
#     src/MPC_Matrices.cpp
#     src/bindcode.cpp
# )

pybind11_add_module(RNEA_PD
    RNEA_PD_control/RNEA_PD.cpp
    RNEA_PD_control/bindcode.cpp
)

# 2. 将之前的 mpc_node 替换为新的目标名 MPC_ROKAE
target_include_directories(RNEA_PD PRIVATE
    "E:/CPP_pac/qpOASES/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

# RNEA控制器需要Eigen和Pinocchio（包含URDF解析器）。
target_link_libraries(RNEA_PD PRIVATE
    Eigen3::Eigen
    pinocchio::pinocchio
)

target_compile_definitions(RNEA_PD PRIVATE
    NOMINMAX
    _WIN32_WINNT=0x0A00
)

if(MSVC)
    target_compile_options(RNEA_PD PRIVATE /utf-8)
endif()

# target_include_directories(mpc_node PRIVATE 
#     "E:/CPP_pac/qpOASES/include"
#     "${CMAKE_CURRENT_SOURCE_DIR}/include"
#     ${CMAKE_CURRENT_SOURCE_DIR}/src
# )

# target_link_libraries(mpc_node PRIVATE 
#     Eigen3::Eigen
#     pinocchio::pinocchio
#     "E:/CPP_pac/qpOASES/build/libs/Release/qpOASES.lib"
# )
