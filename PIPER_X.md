# PiPER-X 仿真适配

本适配将两台独立的 PiPER-X 机械臂接入 RoboTwin，可使用现有任务脚本完成
双臂规划、仿真执行和 HDF5 轨迹采集。

## 组成

- `assets/embodiments/piper_x/piper_x.urdf`：官方 PiPER-X 本体和夹爪模型的
  SAPIEN/cuRobo 扁平化版本，并带默认腕部相机坐标系。
- `assets/embodiments/piper_x_left`、`piper_x_right`：使用 2026-09-08
  工作台标定结果的左右腕相机独立模型。
- `assets/embodiments/piper_x/config.yml`：RoboTwin 关节、夹爪、TCP、相机和
  双臂基座配置。
- `assets/embodiments/piper_x/piper_x.srdf`：相邻链接和固定腕部结构的碰撞白名单。
- `assets/embodiments/piper_x/curobo_tmp.yml`：可迁移的 cuRobo 配置模板。
- `assets/embodiments/piper_x/collision_piper_x.yml`：32 球碰撞近似。
- `env_cfg/task_config/piper_x_demo_clean.yml`：干净背景采集配置。
- `env_cfg/task_config/piper_x_demo_randomized.yml`：随机背景采集配置。

模型来源和上游版本记录在 `assets/embodiments/piper_x/SOURCE.md`，许可证见同目录
`LICENSE.agx_arm_urdf`。

## 采集

干净背景数据：

```bash
conda activate RoboTwin
python scripts/collect_data.py handover_block piper_x_demo_clean
```

双臂由任务配置中的下列三元组启用；第三项是两个基座中心的间距（米）：

```yaml
embodiment: [piper_x_left, piper_x_right, 0.60]
```

默认使用 SAPIEN `default` raster shader，以避免长时间采集时 OIDN 光追去噪器的
GPU 稳定性问题。需要光追时可将任务配置中的 `camera_shader` 改为 `rt`。

输出位于 `data/<task>/<task_config>/`。HDF5 的
`joint_action/vector` 是 14 维：左臂 6 轴、左夹爪、右臂 6 轴、右夹爪；同时保存
左右末端位姿以及 head/left/right 三路相机数据。

## 腕部相机外参

各 `piper_x.urdf` 中的 `camera_joint` 以 `link6` 为父坐标系，使用 SAPIEN 相机
约定：相机局部 `+X` 为光轴向前、`+Y` 向左、`+Z` 向上。使用下面的 embodiment
三元组时，左右臂分别加载独立标定的 URDF 和内参：

```yaml
embodiment: [piper_x_left, piper_x_right, 0.60]
```

标定文件导出的 `T_ee_camera` 使用 OpenCV 光学轴，写入 URDF 前已转换为 SAPIEN
相机轴。左右 `config.yml` 同时保留各自的 640×480 相机矩阵；采集分辨率不同时，
`Camera` 会等比例缩放 `fx/fy/cx/cy`。SAPIEN 输出为无畸变针孔图像，因此标定的
Brown 畸变系数仅作溯源，未用于渲染。

该次标定的左右重投影中位数分别为 `0.266 px` 和 `0.317 px`，但总体几何质量门控
为 `failed`，原因是左右求解器的基座平移结果分歧超过阈值。当前配置按导出的
手眼结果落地，正式大批量采集前应先用若干真实/仿真对应姿态复核目标投影位置。

公共 `piper_x` 模型仍保留由固定结构件 CAD 推导的默认外参；实际双臂数据采集应
使用上述左右独立 embodiment，避免装配误差被一个公共外参覆盖。

## 路径迁移

`curobo_tmp.yml` 使用 `${ASSETS_PATH}`，`curobo.yml` 是当前工作区的展开结果。
移动仓库后，在仓库根目录运行：

```bash
python scripts/update_embodiment_config_path.py
```

这会重新生成所有 embodiment 的绝对路径 cuRobo 配置。
