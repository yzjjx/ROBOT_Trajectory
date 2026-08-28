# 机器人轨迹跟踪
这里对比的是机器人PD控制与机器人加入力矩前馈的PD控制，对比两者控制效果对于轨迹跟踪精度的影响，最终导入刚度模型，实现基于刚度模型的力矩前馈PD控制  
轨迹对比分别为：  
1、位置模式  
2、PD控制  
3、PD控制+力矩前馈  
4、PD控制+力矩前馈+刚度模型  
## 1 轨迹的形状与尺寸
根据国标GB/T 12642—2013《工业机器人 性能规范及其试验方法》实现机器人的指定轨迹跟踪，国标链接可以查看：https://www.doc88.com/p-41571860265921.html， 相关PDF参见：standard_pdf文件夹  
参见国标6.8.6.2，如图1所示，如图2所示，标记了这里所需的全部轨迹

<div align="center">
    <img src="fig\fig_1\1_国标轨迹.png">
    <br>
    图1：国标轨迹相关文字描述
</div>

<div align="center">
    <img src="fig\fig_1\2_国标轨迹.png">
    <br>
    图2：国标真实轨迹执行
</div>

选取工作空间内接正方体，然后选择平面设计轨迹即可，因为珞石SR4最大臂展范围为806.23mm，选择内接正方体的边长为500mm，这样0.8S就是400mm，大圆直径为400mm，小圆直径为40mm，选取对顶点距离的80%，即400*2=800mm

首先是两个直线轨迹

## 2 直线轨迹
### 2.1 直线轨迹2
直线轨迹2为400mm水平直线，在进入轨迹与出去轨迹的时候，需要一定的余量，将余量设置为50mm，一共为500mm的直线，首先要定义机器人的工具中心点  
首先采用的方案是RoboDK的机器人轨迹规划，在RoboDK中先规划机器人的轨迹，如图3所示，然后利用Python程序导出机器人1000Hz的关节点位q，然后将其输入到机器人里面

<div align="center">
    <img src="fig\fig_2\1_RoboDK轨迹规划.png">
    <br>
    图3：RoboDK轨迹规划
</div>

动画如下图所示：

<div align="center">
    <img src="fig\fig_2\3_line_traj_RoboDK.gif">
    <br>
    图3：RoboDK直线轨迹
</div>
  
下一步可以放到mujoco中验证，后期直接在机器人本体验证  
## 3 圆形轨迹
圆形轨迹是直径为400mm的斜向圆，具体轨迹如图4所示

<div align="center">
    <img src="fig\fig_2\2_圆形轨迹.png">
    <br>
    图3：RoboDK轨迹规划
</div>


<div align="center">
    <img src="fig\fig_3\3_traj_circ_roboDK.gif">
    <br>
    图3：RoboDK空间圆形轨迹
</div>

20260818更新：在实际的机器人轨迹运行过程中，运行到两端MoveC的连接位置时，也就是半圆处，因为两段轨迹的不连续，会产生短暂的停顿，因此现在使用MATLAB来进行空间圆形轨迹的生成，空间圆形轨迹的生成需要三个不共线的点、运行速度、提取频率、TCP位姿的3 * 3矩阵，可见MATLAB程序：  MATLAB_code\circle_R200.m，该代码最终生成TCP位姿的4 * 4矩阵，可以通过pinocchio进行IK求解，得到运行这段位姿应该使用的关节角度，输出文件为：Trajectory_txt\circle_R200.txt  
最终运行轨迹见下面图片：

<div align="center">
    <img src="fig\fig_3\1_MATLAB_TRAJECTORY.png">
    <br>
    图3：MATLAB的空间圆轨迹生成
</div>

这个空间圆轨迹生成的小程序不仅仅适用于当前项目，也可以应用于其余空间圆形轨迹的生成  

<div align="center">
    <img src="fig\fig_3\2_Trajectory_cir.gif">
    <br>
    图3：空间圆轨迹生成
</div>

## 4 机器人关节空间PD控制
首先在mujoco中验证机器人PD控制，再放到真实机器人去验证，PD控制为导入一段控制轨迹q与关节速度dq，该部分的控制律为：  

$$
\tau_i=K_{pi}(q_{di}-q{i})+K_{di}(\dot{q}_{di}-\dot{q}_{i})
$$

对每一个关节确定 $K_p$ 和 $K_d$，最终写的控制器也非常简单，参见代码：PD_control_ROBOT\PD_ROBOT.cpp  
实现方式为：为了保证工程的可迁移性，在cpp中设计控制器，然后通过pybind11创建动态链接库，然后用python的mujoco运行  

### 4.1 基于Mujoco的机器人关节空间PD控制

### 4.2 机器人实物关节空间PD控制：ROKAE SR4

## 5 基于力矩前馈的关节空间PD控制
虽然在仿真环境下，控制器效果还可以，但是在实际的工程实现中，碰到的问题如下：  
1、PD控制器参数难以调整，一个关节需要调整两个参数，六个关节需要调整12个参数；  
2、实际收敛效果非常差，末端关节抖动非常厉害，关节速度、关节力矩都容易超限；  

为了避免继续调参浪费时间，下面直接采用基于力矩前馈的关节空间PD控制方案  
力矩前馈就是在通过官方或者辨识得到机器人准确的动力学参数后，根据RNEA先计算一遍机器人动力学力矩，然后用PD控制器进行纠偏，也就是控制律改为：  

$$
\tau_{cmd} = \tau_{RNEA}(q_{d},\dot{q}_{d},\ddot{q}_{d})+K_p(q_d-q)+K_d(\dot{q}_d-\dot{q})
$$

这样做的目的就是减轻PD控制器的压力，使PD控制更容易作用。之前PD控制需要补偿原始重力、外部扰动、摩擦、动力学等，现在去掉了大头的动力学部分，更容易对力矩进行修正  
这里通过Pinocchio实现RNEA，这样只需要更换不同的urdf文档，就可以实现不同机器人的动力学计算  

### 5.1 Mujoco仿真：基于力矩前馈的关节空间PD控制

首先，当前的Pinocchio是存在于conda环境里面，需要选择不同的编译器进行编译，因此需要更改.vscode\settings.json，将编译后的输出文件修改为build_msvc，与之前的编译文件进行区分，后续过程和结果如下面两图所示  

```
{
    "python-envs.defaultEnvManager": "ms-python.python:conda",
    "python-envs.defaultPackageManager": "ms-python.python:conda",
    "cmake.generator": "Visual Studio 17 2022",
    "cmake.buildDirectory": "${workspaceFolder}/build_msvc",
    "cmake.configureArgs": [
        "-A",
        "x64"
    ]
}
```

<div align="center">
    <img src="fig\fig_5\2_build.png">
    <br>
    图3：生成
</div>

<div align="center">
    <img src="fig\fig_5\3_build_output.png">
    <br>
    图3：生成结果
</div>

最终控制结果如图：

<div align="center">
    <img src="fig\fig_5\4_RNEA_PD.gif">
    <br>
    图3：基于力矩前馈的关节空间PD控制
</div>

## 6 基于刚度模型与力矩前馈的机器人PD控制

## 7 机器人理想轨迹获取
将所有轨迹位置全部转换到机器人基坐标系，在基坐标系下，绘制理想圆弧，在MATLAB绘图显示，然后将激光跟踪仪得到的XYZ数据进行坐标转换，然后与理想数据进行对比，最后看轨迹跟踪误差，为了使机器人的理想点位和激光跟踪仪得到的点位对应，需要注意设置时间戳的同步  
机器人理想TCP轨迹这里可以通过MATLAB进行获取，因为所使用的激光跟踪仪不能获得姿态数据，因此只采集位置数据XYZ  
也就是说，以机器人基座坐标系为原点，根据RoboDK定义的点位，在空间中画出来这个圆形或者直线，如图 所示  

<div align="center">
    <img src="fig\fig_5\1_理想圆弧轨迹.png">
    <br>
    图3：理想圆弧轨迹
</div>
  
## 8 激光跟踪仪轨迹获取：获取机器人真实轨迹
通过激光跟踪仪可以获取机器人的真实轨迹，激光跟踪仪获得的坐标系是以激光跟踪仪为原点，得到靶球球心的位置，因此需要将这个坐标转换到机器人的基坐标系。  
由于靶球到机器人基坐标系的位置是不可知的，因此要对其进行辨识得到。  
选择任意八个点，加上最小二乘法就可以辨识出来  
详细代码参见：MATLAB_code\Coordinate_trans.m  
这里选择六个点进行机器人标定，如图所示  

<div align="center">
    <img src="fig\fig_6\1_robot_pos.png">
    <br>
    图3：机器人位姿
</div>
  


20260828更新：
轨迹跟踪需要机器人逆运动学求解理想空间圆，然后得到对应的关节角，然后将关节角输入PD控制器

更新：理想轨迹逆运动学解算需要使用机器人本体的DH参数，而不是辨识出来的DH参数，因为机器人内部解算也是用的机器人本体的DH参数