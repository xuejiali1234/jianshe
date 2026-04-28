# 交通流预测与信号控制数字孪生平台

本项目面向城市道路交通运行分析、短时交通流预测与信号放行控制研究，构建了一套基于 `SUMO + TraCI + FastAPI + WebSocket + Web 地图看板` 的一体化数字孪生平台。系统既支持实时仿真展示，也支持批量场景生成、movement 级预测模型训练、事故与限速扰动分析，以及单路口强化学习信号控制实验。

当前工程的默认研究口径已经从早期的 edge 级检测切换为 **movement 级预测链路**。预测对象不再是单条短进口 edge，而是由 `tls_id + incoming_edge + turn_type + outgoing_edge` 构成的转向交通流单元。该设计更适合表达交叉口左转、直行、右转之间的差异，也更便于后续做相位级控制聚合。

## 当前默认链路

当前默认使用的是 `full_v3` 正式链路：

- 训练数据：`data/raw/batch_movement_aggregates_full_v3.csv`
- 场景清单：`data/raw/scenarios_full_v3/manifest.csv`
- 预测配置：`configs/prediction_config.json`
- 模型产物：`models/artifacts_full_v3`
- 指标报告：`reports/full_v3_training/metrics.csv`
- 默认信号网：`data/processed/czq_tls_webster.net.xml`

预测配置中的关键参数如下：

- `observation_level = movement`
- `sample_interval_s = 60`
- `history_steps = 12`
- `horizon_steps = 15`
- `targets = [arrival_flow, mean_speed, queue_veh]`
- `base_demand_factor = 0.25`
- `active_model_from_registry = true`
- `preferred_model = transformer_v2`
- `fallback_model = ha_baseline`

说明：

- 服务启动时优先读取 `models/artifacts_full_v3/model_registry.json`
- 当前 full_v3 训练结果中，registry 的 `active_model` 是 `transformer_v1`
- 因此在线预测服务会默认加载 `transformer_v1`，加载失败或历史窗口不足时回退到 `ha_baseline`

## 项目目标

本项目并不只关注单一预测模型，而是围绕“仿真—采集—预测—控制—展示”构建完整实验闭环，主要目标包括：

- 在 SUMO 中构建可重复的城市路网交通数字孪生平台
- 以 movement 为核心观测单元生成短时交通预测数据
- 训练并评估 `HA / XGBoost / LSTM / Transformer V1 / Transformer V2`
- 将训练后的模型接回同一套 Web 服务，支撑实时预测与场景对比
- 在统一仿真环境中实现 `Webster / MaxPressure / DQN` 信号控制实验
- 分析“无预测控制”与“预测增强控制”之间的差异

## 平台能力概览

### 1. 实时仿真看板

运行 `python app.py` 后，系统会启动 FastAPI 服务，并在后台启动 SUMO 仿真。前端地图看板通过 WebSocket 持续接收：

- 车辆位置与朝向
- 路网运行状态
- 排队、速度、流量等统计指标
- 信号灯与相位状态
- movement 聚合后的实时预测结果
- 事故 / 限速对比结果

### 2. movement 级短时交通流预测

系统默认采用 movement 级建模。每个 movement 由以下四元组定义：

```text
m = (tls_id, incoming_edge, turn_type, outgoing_edge)
```

当前 full_v3 数据中共包含：

- `151` 个 movement
- 输入窗口：`12` 个历史步
- 输出窗口：`15` 个未来步
- 输出维度：`151 × 3 = 453`

预测目标包括：

- `arrival_flow`
- `mean_speed_mps`
- `queue_veh`

当前模型族包括：

- `ha_baseline`
- `xgboost`
- `lstm`
- `transformer_v1`
- `transformer_v2`

### 3. 批量场景生成

批量仿真脚本支持多类需求与扰动场景，当前 full 口径包含：

- `S1_normal`
- `S2_peak`
- `S3_congested`
- `S3_control`
- `S4_vsl`
- `S5_incident`

其中：

- `S3_control` 表示信号配时扰动
- `S4_vsl` 表示可变限速扰动
- `S5_incident` 表示更接近真实事故的封停 / 容量受限场景

### 4. 信号控制实验

项目当前已经具备单路口强化学习信号控制骨架，目标路口默认配置在：

```text
configs/rl_signal_config.json
```

当前控制对照策略包括：

- `webster`
- `max_pressure`
- `dqn_no_pred`
- `dqn_pred_v2`（预测增强 DQN 实验入口）

RL 状态采用相位级聚合，不直接把所有 movement 原始值塞给控制器。这样可以更贴合“当前应放行哪个相位”的决策语义。

## 当前实验状态

### full_v3 预测训练

当前 full_v3 正式数据集摘要：

- `X_shape = [2448, 12, 1512]`
- `y_shape = [2448, 15, 453]`
- `n_train = 1700`
- `n_val = 340`
- `n_test = 408`

当前 full_v3 训练中，表现最好的预测模型为：

- `transformer_v1`

当前模型 registry：

- `active_model = transformer_v1`
- `active_artifact = transformer_v1_model.pt`

### full_v3 RL 控制

当前 `reports/rl_signal_control/full_v3_pred_control/` 下已经包含：

- `webster_1800_eval.csv`
- `max_pressure_1800_eval.csv`
- `dqn_no_pred_eval.csv`
- `pred_v2_*_eval.csv`
- `policy_comparison_1800.csv`
- `sweep_summary.csv`

截至目前，可以严谨表达的结论是：

- `DQN-no-pred` 已达到或略优于 `MaxPressure`
- 当前“预测增强 DQN”链路已经跑通
- 但在 full_v3 同口径实验下，尚未证明预测增强控制优于无预测控制

## 环境准备

### 1. 创建 Conda 环境

推荐使用 Python 3.10：

```cmd
conda create -n traffic_pred python=3.10 -y
conda activate traffic_pred
```

### 2. 安装依赖

基础依赖：

```cmd
pip install -r requirements_web.txt
```

PyTorch 建议根据本机 CUDA 环境单独安装。示例：

CPU 版本：

```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

GPU 版本请按本机 CUDA 版本选择对应 wheel。

### 3. 配置 SUMO Python 工具

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

## 快速启动

启动系统：

```cmd
python app.py
```

浏览器访问：

```text
http://127.0.0.1:8000
```

服务启动后，后端会加载预测配置与模型服务，并在后台启动 SUMO 仿真。关闭网页不会终止 SUMO；终止 `python app.py` 才会结束当前运行。

## 常用命令

### 1. 生成信号灯路网

```cmd
python -m sim.scripts.build_webster_tls_net --overwrite
```

输出通常位于：

- `data/processed/czq_tls_webster.net.xml`
- `data/processed/czq_tls_webster_summary.json`

### 2. 生成 movement 字典

```cmd
python -m sim.scripts.build_movement_config
```

输出通常位于：

- `configs/movement_config.json`
- `data/processed/movement_map.csv`

### 3. 构建 movement 图结构

```cmd
python -m sim.scripts.build_movement_graph --movement-config configs/movement_config.json --out data/processed/movement_graph.json
```

### 4. 批量运行 SUMO 生成训练数据

完整 full 场景：

```cmd
python -m sim.scripts.run_batch_sumo --overwrite --scenario-preset full --sim-end 3600 --collector movement --output-csv data/raw/batch_movement_aggregates_full_v3.csv --scenario-dir data/raw/scenarios_full_v3
```

小样本 smoke：

```cmd
python -m sim.scripts.run_batch_sumo --overwrite --scenario-preset full --limit 3 --sim-end 1800 --collector movement --output-csv data/raw/batch_movement_aggregates_full_v3_smoke.csv --scenario-dir data/raw/scenarios_full_v3_smoke
```

### 5. movement 数据 QA

```cmd
python -m sim.scripts.inspect_movement_catalog --movement-config configs/movement_config.json --out reports/movement_catalog_quality.json
python -m sim.scripts.diagnose_movement_data --csv data/raw/batch_movement_aggregates_full_v3.csv --manifest data/raw/scenarios_full_v3/manifest.csv --out reports/movement_data_quality_full_v3.json
```

### 6. 训练预测模型

```cmd
python -m prediction.training --csv data/raw/batch_movement_aggregates_full_v3.csv --dataset-dir data/datasets/full_v3_movement_control --artifact-dir models/artifacts_full_v3 --report-dir reports/full_v3_training --update-config
```

训练脚本会完成：

- 数据集构建
- HA / XGBoost / LSTM / Transformer V1 / Transformer V2 训练与评估
- 生成 `metrics.csv`
- 生成 `model_registry.json`
- 将配置更新到新的 artifact 路径

### 7. RL 基线评估

```cmd
python -m rl.evaluate_policy --config configs/rl_signal_config.json --policy webster --sim-end 1800 --out reports/rl_signal_control/full_v3_pred_control/webster_1800_eval.csv
python -m rl.evaluate_policy --config configs/rl_signal_config.json --policy max_pressure --sim-end 1800 --out reports/rl_signal_control/full_v3_pred_control/max_pressure_1800_eval.csv
```

### 8. 训练 DQN-no-pred

```cmd
python -m rl.train_dqn --config configs/rl_signal_config.json --timesteps 10000 --sim-end 1800 --use-prediction false --device cuda
```

### 9. 训练 / 优化预测增强 DQN

单轮：

```cmd
python -m rl.train_dqn --config configs/rl_signal_config.json --timesteps 5000 --sim-end 1800 --use-prediction true --device cuda
```

多轮 sweep：

```cmd
python -m rl.optimize_pred_v2 --rounds 5 --timesteps 10000 --sim-end 1800 --device cuda --report-dir reports/rl_signal_control/full_v3_pred_control --artifact-dir models/artifacts_rl/full_v3_pred_control
```

### 10. 夜间一键全流程

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full_v3_nightly.ps1 -Device cuda -DqnTimesteps 10000 -DqnRounds 5
```

## 主要 API

### 预测配置

```http
GET /api/prediction/config
```

返回内容通常包括：

- movement / edge 观察配置
- 历史窗口和预测步长
- 当前 active model
- fallback 状态
- artifact 路径
- 输入特征维度说明

### 最新预测

```http
GET /api/prediction/latest
```

返回最近一次聚合观测与预测结果。

### 在线预测

```http
POST /api/predict
```

输入历史窗口，返回未来 15 步预测。当前输出同时包含：

- `movements`
- 兼容前端展示的 legacy `nodes`

### 场景 run 列表

```http
GET /api/prediction/scenario-runs
```

返回场景 run 元信息，用于正常 / 限速 / 事故对比。

### 场景对比

```http
POST /api/prediction/scenario-compare
```

输入正常 run、扰动 run、路段与模型，返回预测对比与差值摘要。

## 目录结构

```text
.
├── app.py
├── configs/
├── data/
├── models/
├── prediction/
├── reports/
├── rl/
├── scripts/
├── sim/
├── static/
├── czq.net.xml
├── czq_demand.rou.xml
├── intersection.sumocfg
├── gui-settings.xml
├── requirements_web.txt
└── README.md
```

## 目录说明

### 根目录

- `app.py`：主服务入口，负责启动 FastAPI、SUMO 后台仿真、WebSocket 推送和预测 API
- `czq.net.xml`：基础 SUMO 路网文件
- `czq_demand.rou.xml`：基础交通需求 route 文件
- `intersection.sumocfg`：SUMO 仿真配置
- `gui-settings.xml`：SUMO GUI 显示配置
- `vtypes.add.xml`：车辆类型配置
- `requirements_web.txt`：Web 服务、数据处理、机器学习与绘图依赖

### `configs/`

- `prediction_config.json`：预测配置，当前默认指向 `full_v3`
- `movement_config.json`：movement 字典定义
- `rl_signal_config.json`：RL 信号控制配置
- 其他信号控制相关配置文件

### `prediction/`

- `collector.py`：早期 edge 级采集逻辑
- `movement_collector.py`：当前默认 movement 级采集逻辑
- `dataset.py`：滑窗数据集构建
- `torch_models.py`：LSTM、Transformer V1、Transformer V2 模型结构
- `service.py`：在线预测服务与模型加载
- `phase_aggregation.py`：movement 预测向相位级控制特征的聚合
- `training.py`：训练、评估、保存、更新配置入口

### `sim/`

- `route_tools.py`：交通需求与 route 调整工具
- `signal_timing.py`：固定配时与 Webster 风格逻辑
- `validation.py`：环境与文件检查
- `scripts/build_webster_tls_net.py`：生成信号灯路网
- `scripts/build_movement_config.py`：生成 movement 字典
- `scripts/build_movement_graph.py`：生成 movement 图结构
- `scripts/inspect_movement_catalog.py`：movement 字典 QA
- `scripts/diagnose_movement_data.py`：movement 数据 QA
- `scripts/run_batch_sumo.py`：批量场景仿真入口

### `rl/`

- `env.py`：SUMO 信号控制环境
- `state_builder.py`：RL 状态构造
- `phase_controller.py`：安全切相信号执行逻辑
- `reward.py`：奖励函数
- `baselines.py`：Webster / MaxPressure 基线
- `train_dqn.py`：SB3 DQN 训练脚本
- `evaluate_policy.py`：控制策略评估脚本
- `optimize_pred_v2.py`：预测增强 DQN 多轮优化
- `summarize_results.py`：结果汇总与图表生成

### `static/`

- `index.html`：前端页面结构
- `css/style.css`：前端样式
- `js/main.js`：地图渲染、车辆显示、WebSocket、预测面板与控制展示逻辑
- `assets/`：车辆图标与其他静态资源

### `data/`

- `raw/`：实时与批量仿真原始聚合输出
- `processed/`：处理后的信号网、movement 图和摘要文件
- `datasets/`：训练数据集缓存
- `archive/`：历史训练产物与旧实验归档

### `models/`

- `artifacts_full_v3/`：当前正式 full_v3 模型产物
- `artifacts_rl/`：RL 控制模型产物

### `reports/`

- `full_v3_training/`：预测训练指标、摘要与图表
- `rl_signal_control/full_v3_pred_control/`：RL 控制评估结果、对比表与 sweep 摘要
- 其他 QA、诊断和中间分析报告

## 数据口径说明

当前默认训练数据为 movement 长表。常见字段包括：

- `run_id`
- `scenario_id`
- `timestamp`
- `step`
- `movement_id`
- `tls_id`
- `incoming_edge`
- `outgoing_edge`
- `turn_type`
- `arrival_flow`
- `discharge_flow`
- `mean_speed_mps`
- `speed_kmh`
- `queue_veh`
- `queue_meter`
- `incident_flag`
- `phase_id`
- `phase_elapsed_s`
- `green_remaining_s`
- `signal_state`
- `zone_quality`

训练时会严格按 `run_id` 分段构建滑窗样本，避免不同仿真运行之间发生拼窗。

## Git 上传策略

仓库默认保留：

- 代码
- 配置
- 基础 SUMO 文件
- 前端页面
- 说明文档

仓库默认不上传：

- 大型批量仿真 CSV
- 训练数据集缓存
- PyTorch 权重
- joblib 模型文件
- 自动生成图表
- 历史归档数据

如需复现实验，应在本地重新生成数据并重新训练。

## 注意事项

1. 运行前需确保 SUMO 可用，且 `sumo --version` 正常。
2. `traci` 与 `sumolib` 来自 SUMO `tools` 目录，需要正确配置 `PYTHONPATH`。
3. 修改 `base_demand_factor`、route 分布、signal 网或 movement 配置后，应重新生成批量数据并重训模型。
4. 当前“预测增强 RL 已打通”不等于“预测增强 RL 已优于无预测 RL”，论文和答辩表述应保持严谨。
5. 若 GitHub 网页可访问但 `git push` 失败，通常需要为当前仓库单独配置代理。

## 后续方向

从当前工程状态看，后续最有价值的推进方向包括：

- 继续调优 `Transformer V2`，使其稳定超过 `Transformer V1`
- 将 `movement_graph.json` 更显式地接入 V2 时空结构
- 优化 RL 预测增强状态设计，减少无效前瞻特征噪声
- 在统一 full_v3 口径下继续验证 `DQN-pred-v2` 是否优于 `DQN-no-pred`

