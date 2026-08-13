cmake_minimum_required(VERSION 3.10)
project(ROKAE_MPC)

set(CMAKE_CXX_STANDARD 14)

# ==================== 【强行注入缓存，彻底封死自动搜索】 ====================
set(Python3_EXECUTABLE "F:/ProgramData/anaconda3/envs/mpc_rokae/python.exe" CACHE FILEPATH "Force Conda Python3" FORCE)
set(PYTHON_EXECUTABLE "F:/ProgramData/anaconda3/envs/mpc_rokae/python.exe" CACHE FILEPATH "Force Conda Python Fallback" FORCE)
set(PYBIND11_FINDPYTHON ON)
# =========================================================================

list(APPEND CMAKE_PREFIX_PATH "F:/ProgramData/anaconda3/envs/mpc_rokae/Library")

find_package(pybind11 REQUIRED)

find_package(Eigen3 REQUIRED)

# # 寻找预编译好的 Pinocchio 及其依赖
# find_package(pinocchio CONFIG REQUIRED)

# add_executable(mpc_node 
#     # test/test_dynamics.cpp
#     src/dynamics_com_tau.cpp
#     src/robot_dynamics.c
#     src/Prediction.cpp
#     src/MPC_Matrices.cpp
#     src/bindcode.cpp
# )

pybind11_add_module(PD_ROBOT
    PD_control_ROBOT/PD_ROBOT.cpp
    PD_control_ROBOT/bindcode.cpp
)

# 2. 将之前的 mpc_node 替换为新的目标名 MPC_ROKAE
target_include_directories(PD_ROBOT PRIVATE
    "E:/CPP_pac/qpOASES/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

# # 3. 同样替换链接库的目标名
# target_link_libraries(PD_ROBOT PRIVATE
#     Eigen3::Eigen
#     pinocchio::pinocchio
#     "E:/CPP_pac/qpOASES/build/libs/Release/qpOASES.lib"
# )

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