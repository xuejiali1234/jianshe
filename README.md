# 交通流预测与信号控制数字孪生平台

本项目面向城市道路交通运行分析、短时交通流预测与信号放行控制研究，构建了一套基于 `SUMO + TraCI + FastAPI + WebSocket + Web 地图看板` 的一体化数字孪生平台。系统支持实时仿真展示、批量场景生成、movement 级交通状态预测、事故与限速扰动分析，以及单路口/多路口强化学习信号控制实验。

当前工程已经从早期的 edge 级检测口径收敛到 **movement 级预测 + 相位级控制聚合** 这一条主线。预测对象由

```text
tls_id + incoming_edge + turn_type + outgoing_edge
```

共同定义，能更好地区分左转、直行、右转等不同交通流单元，也更适合与信号相位控制做联动。

## 1. 当前正式可用主线

### 1.1 预测主模型

当前正式推荐的预测主模型为：

- `Transformer V1`
- 训练数据：`full_v3 + 精选 lane incident`
- 模型目录：`models/artifacts_full_v3_plus_lane_incident_v1_selected`
- 报告目录：`reports/full_v3_plus_lane_incident_v1_selected`

核心结果：

- `overall MAE = 0.7136`
- `overall RMSE = 1.5602`
- `overall WAPE = 0.2526`

这一版在常规场景和事件场景上都优于此前的 `full_v3` 版本，是当前预测主线的正式口径。

### 1.2 单路口 RL 正式参考结果

当前单路口控制正式参考 checkpoint 为：

- `DQN-pred-v1-v3 @ 9000 steps`
- 文件：
  `models/artifacts_rl/anticipatory_v3_long/checkpoints/dqn_pred_v1_anticipatory_v3_long/dqn_signal_single_tls_dqn_pred_v1_anticipatory_v3_long_9000_steps.zip`
- 正式材料：
  `reports/rl_signal_control/anticipatory_v3_checkpoint_9000_formal`

该版本是单路口预测增强 RL 的综合最优 checkpoint。

### 1.3 多路口联动正式结果

当前多路口正式主结果采用西侧 3 路口强簇：

- `12254641672`（对应 `E23.226`）
- `12254692358`（对应 `E26.232`）
- `J42`（对应 `E31.70`）

正式结果分为两条：

- 多路口主结果：
  `west_multi_no_pred_v1 @ 5000`
- 预测增强对照：
  `west_multi_pred_v1 @ 2000`

对应目录：

- 模型：
  `models/artifacts_rl_multi/west_v1`
- 正式材料：
  `reports/rl_signal_control_multi/west_v1_formal`
- checkpoint 扫描：
  `reports/rl_signal_control_multi/west_v1_checkpoint_scan`

当前结论是：

- `west_multi_no_pred_v1 @ 5000` 为三场景综合最优
- `west_multi_pred_v1 @ 2000` 在事故响应上更敏感，但综合指标仍弱于 `no-pred`

## 2. 平台能力概览

### 2.1 实时仿真与 Web 看板

运行 `python app.py` 后，系统会启动 FastAPI 服务，并驱动 SUMO 仿真。前端通过 WebSocket 持续接收：

- 车辆位置与朝向
- 路网运行状态
- 排队、速度、流量等统计指标
- 信号灯与相位状态
- movement 聚合后的实时预测结果
- 场景扰动下的对比信息

### 2.2 movement 级短时交通流预测

当前默认的预测设定为：

- `observation_level = movement`
- `sample_interval_s = 60`
- `history_steps = 12`
- `horizon_steps = 15`
- `targets = [arrival_flow, mean_speed, queue_veh]`

当前正式训练集下共有：

- `151` 个 movement
- 输入张量形状：`[N, 12, 1663]`
- 输出张量形状：`[N, 15, 453]`

模型族当前包含：

- `ha_baseline`
- `xgboost`
- `lstm`
- `transformer_v1`
- `transformer_v2`

其中当前正式主模型为 `transformer_v1`。

### 2.3 批量场景生成

现有正式场景主集为 `full_v3`，包含：

- `S1_normal`
- `S2_peak`
- `S3_congested`
- `S3_control`
- `S4_vsl`
- `S5_incident`

在此基础上，又新增了更贴近真实事故的 lane 级增量场景：

- `S5_lane_incident_v1`

本轮最终并入正式训练主线的是精选子集：

- `north_j9_e13`
- `north_j9_minus_e13`
- `west_minus_e21_32`

对应合并数据文件为：

- `data/raw/batch_movement_aggregates_full_v3_plus_lane_incident_v1_selected.csv`

### 2.4 信号控制实验

当前控制实验主线包含：

- 规则控制：`Webster`
- 压力控制：`MaxPressure`
- 单路口 RL：`DQN-no-pred`、`DQN-pred-v1`
- 多路口联动 RL：共享策略 `DQN-no-pred`、`DQN-pred-v1`

其中单路口和多路口均采用：

- 主绿灯相位控制
- 黄灯/全红安全过渡
- 相位级聚合状态
- 可接入预测增强状态与前瞻型 reward

## 3. 当前关键数据与配置

### 3.1 预测训练主线

- 数据：
  `data/raw/batch_movement_aggregates_full_v3_plus_lane_incident_v1_selected.csv`
- 数据集目录：
  `data/datasets/full_v3_plus_lane_incident_v1_selected`
- 模型目录：
  `models/artifacts_full_v3_plus_lane_incident_v1_selected`
- 报告目录：
  `reports/full_v3_plus_lane_incident_v1_selected`

### 3.2 多路口西侧强簇配置

- 联动图：
  `data/processed/tls_coordination_graph_west_v1.json`
- 多路口配置：
  `configs/rl_multi_signal_config_west_v1.json`

### 3.3 多路口预测桥配置

为避免直接改动全局默认预测桥，多路口 RL 当前单独使用：

- `configs/prediction_config_full_v3_plus_lane_incident_v1_selected.json`

这使得多路口 RL 能稳定调用新的 `Transformer V1` 预测桥，而不强行改动网页服务的全局默认配置。

## 4. 项目结构

```text
app.py
configs/
data/
models/
prediction/
reports/
rl/
sim/
static/
scripts/
```

重点目录说明：

- `prediction/`
  预测数据集构建、模型定义、训练与加载逻辑
- `rl/`
  单路口和多路口控制环境、奖励函数、训练与评估入口
- `sim/`
  SUMO 路网、场景生成、批量仿真与 movement 采集脚本
- `reports/`
  各阶段实验报告、对比表和图像输出
- `models/`
  训练得到的正式与中间模型产物

## 5. 环境准备

### 5.1 Conda 环境

建议使用 Python 3.10：

```cmd
conda create -n traffic_pred python=3.10 -y
conda activate traffic_pred
```

### 5.2 安装依赖

```cmd
pip install -r requirements_web.txt
```

### 5.3 安装 PyTorch

CPU 版本示例：

```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

如本机有 CUDA，请根据本机驱动和 CUDA 版本安装对应 GPU 版本。

### 5.4 配置 SUMO

如果 SUMO 安装在 `D:\SUMO`，可配置：

```cmd
set SUMO_HOME=D:\SUMO
set PYTHONPATH=%SUMO_HOME%\tools;%PYTHONPATH%
```

验证：

```cmd
sumo --version
python -c "import traci, sumolib; print('SUMO tools OK')"
```

## 6. 快速启动

启动平台：

```cmd
python app.py
```

浏览器访问：

```text
http://127.0.0.1:8000
```

## 7. 常用命令

### 7.1 训练当前正式预测主线

```cmd
python -m prediction.training --csv data/raw/batch_movement_aggregates_full_v3_plus_lane_incident_v1_selected.csv --dataset-dir data/datasets/full_v3_plus_lane_incident_v1_selected --artifact-dir models/artifacts_full_v3_plus_lane_incident_v1_selected --report-dir reports/full_v3_plus_lane_incident_v1_selected
```

### 7.2 训练西侧强簇多路口 no-pred

```cmd
python -m rl.train_multi_dqn --config configs/rl_multi_signal_config_west_v1.json --timesteps 5000 --sim-end 1800 --use-prediction false --out-dir models/artifacts_rl_multi/west_v1 --report-dir reports/rl_signal_control_multi/west_v1 --run-name west_multi_no_pred_v1 --device cuda --checkpoint-every 1000
```

### 7.3 训练西侧强簇多路口 pred-v1

```cmd
python -m rl.train_multi_dqn --config configs/rl_multi_signal_config_west_v1.json --timesteps 5000 --sim-end 1800 --use-prediction true --use-prediction-reward true --reward-mode anticipatory_delta_pressure_v2 --out-dir models/artifacts_rl_multi/west_v1 --report-dir reports/rl_signal_control_multi/west_v1 --run-name west_multi_pred_v1 --device cuda --checkpoint-every 1000
```

## 8. 当前阶段建议

如果现在要做项目展示、论文书写或答辩，建议优先围绕以下四个结果展开：

1. `Transformer V1`（lane incident selected 版）作为预测主模型
2. `DQN-pred-v1-v3 @ 9000` 作为单路口正式参考结果
3. `west_multi_no_pred_v1 @ 5000` 作为多路口联动主结果
4. `west_multi_pred_v1 @ 2000` 作为多路口预测增强对照结果

这样口径最稳，也最能反映当前工程的实际完成度。
