# 基于关节刚度的机器人轨迹跟踪

理想轨迹逆运动学解算需要使用机器人本体的DH参数，而不是辨识出来的DH参数，因为机器人内部解算也是用的机器人本体的DH参数  

对比TCP轨迹可以分为两种方法：  

1、利用理想TCP轨迹生成关节角，将关节角输入到机器人SDK中，然后获取机器人实际采集到的关节角位置，用“利用理想TCP轨迹生成关节角”这一步使用的IK方法和DH表，求解回去TCP位置  

2、利用理想TCP轨迹生成关节角，将关节角输入到机器人SDK中，然后获取机器人实际采集到的TCP位置，看理想TCP位置和采集的TCP位置差别，但是这种方法获得的TCP位置也是经过内部模型计算出来的    

也就是说，正逆运动学的解算必须使用同一个工具和同一个DH表，这里倾向于使用第一种方法  

## 1 理想轨迹生成
利用MATLAB生成理想TCP位姿，MATLAB是利用三点的齐次变换4*4矩阵生成对应的空间圆，然后通过Pinocchio解算关节角度，然后将关节角输入机器人SDK中  

注意:该阶段生成的机器人轨迹，需要使用S曲线加速、然后恒定速度、然后S曲线减速的顺序，不然在后期加入控制器的时候，会出现解算出的加速度过大等问题  

生成理想空间圆的代码查看：MATLAB_code\Trajectory_circle_generate.m，理论上该代码会生成任意机器人的空间圆，只要不变TCP姿态的3 * 3矩阵  

生成的空间圆如下图所示：  

<div align="center">
    <img src="fig\fig1\1_ideal_circle.png">
    <br>
    图1：理想轨迹生成
</div>
  
之后，将理想TCP轨迹输入到Python_code\Tra_generate_Pinocchio_DH.py中，可以求解IK生成实际机器人的关节角  

可以在mujoco中看机器人轨迹仿真，代码为：Python_code\trajectory_show\robot_trajectory_show.py  

轨迹仿真代码运行效果如下图所示：

<div align="center">
    <img src="fig\fig1\2_mujoco_traj.png">
    <br>
    图2：Mujoco轨迹仿真
</div>
  

