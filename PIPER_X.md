# PiPER-X 仿真适配

本适配将两台独立的 PiPER-X 机械臂接入 RoboTwin，可使用现有任务脚本完成
双臂规划、仿真执行和 HDF5 轨迹采集。

## 组成

- `assets/embodiments/piper_x/piper_x.urdf`：官方 PiPER-X 本体和夹爪模型的
  SAPIEN/cuRobo 扁平化版本，并带腕部相机坐标系。
- `assets/embodiments/piper_x/config.yml`：RoboTwin 关节、夹爪、TCP、相机和
  双臂基座配置。
- `assets/embodiments/piper_x/piper_x.srdf`：相邻链接和固定腕部结构的碰撞白名单。
- `assets/embodiments/piper_x/curobo_tmp.yml`：可迁移的 cuRobo 配置模板。
- `assets/embodiments/piper_x/collision_piper_x.yml`：32 球碰撞近似。
- `task_config/piper_x_debug.yml`：1 集、双臂、三相机的采集验证配置。
- `task_config/piper_x_demo_clean.yml`：50 集干净背景采集配置。

模型来源和上游版本记录在 `assets/embodiments/piper_x/SOURCE.md`，许可证见同目录
`LICENSE.agx_arm_urdf`。

## 采集

单集验证：

```bash
conda activate RoboTwin
python script/collect_data.py handover_block piper_x_debug
```

50 集干净背景数据：

```bash
python script/collect_data.py handover_block piper_x_demo_clean
```

双臂由任务配置中的下列三元组启用；第三项是两个基座中心的间距（米）：

```yaml
embodiment: [piper_x, piper_x, 0.65]
```

默认使用 SAPIEN `default` raster shader，以避免长时间采集时 OIDN 光追去噪器的
GPU 稳定性问题。需要光追时可将任务配置中的 `camera_shader` 改为 `rt`。

输出位于 `data/<task>/<task_config>/`。HDF5 的
`joint_action/vector` 是 14 维：左臂 6 轴、左夹爪、右臂 6 轴、右夹爪；同时保存
左右末端位姿以及 head/left/right 三路相机数据。

## 腕部相机外参

`piper_x.urdf` 中的 `camera_joint` 以 `link6` 为父坐标系，使用 SAPIEN 相机约定：
相机局部 `+X` 为光轴向前、`+Y` 向左、`+Z` 向上。左右臂加载同一个 URDF，因而
使用相同的局部外参：

```xml
<origin xyz="-0.0347 -0.0687 0.0430"
        rpy="0 -1.1116927 1.5707963"/>
```

这组参数由 `末端相机固定结构件.STEP` 推导。STEP 使用毫米单位，支架包围盒约为
`81.0 x 73.7 x 48.0 mm`，腕部夹持内圆直径约 `57.2 mm`，相机安装板倾角为
`20 deg`，四个安装孔形成约 `45 x 5 mm` 的孔距。支架相对 PiPER-X 夹爪绕
`link6` 旋转 `90 deg` 后，安装孔中心和现有 PiPER D435 光心偏置共同给出上述
平移；姿态保留现有 D435 光学坐标系相对安装板的约 `6.3 deg` 修正。

STEP 只包含固定结构件，不包含相机本体。因此这组外参中的支架尺寸和安装板倾角
来自 CAD，而光心到安装孔的偏置来自仓库现有 D435 模型。若实机使用不同相机、
垫片或孔位，应以实测 `link6 -> optical_center` 手眼标定结果替换该值。

## 路径迁移

`curobo_tmp.yml` 使用 `${ASSETS_PATH}`，`curobo.yml` 是当前工作区的展开结果。
移动仓库后，在仓库根目录运行：

```bash
python script/update_embodiment_config_path.py
```

这会重新生成所有 embodiment 的绝对路径 cuRobo 配置。
