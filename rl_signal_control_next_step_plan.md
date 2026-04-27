# 项目后续完善与强化学习信号控制实现抓手

适用仓库：`https://github.com/xuejiali1234/jianshe.git`  
版本判断：仓库已进入 **movement 级交通流预测 + Transformer V2 调试后** 的下一阶段。本文档面向 Codex / 开发实现者，用于继续完善预测链路，并落地强化学习信号灯控制。

---

## 0. 当前工程状态判断

仓库当前已经形成比较完整的数字孪生预测链路：`FastAPI + SUMO + TraCI + WebSocket + 前端看板`，并包含离线批量仿真、滑窗数据集、模型训练评估和事故场景对比能力。README 中仍写的是 HA、XGBoost、LSTM、Transformer V1 训练链路，但代码与配置已经向 movement 级和 Transformer V2 迁移。

当前最关键的进展如下：

1. `configs/prediction_config.json` 已设置 `observation_level = movement`，训练数据路径切到 `data/raw/batch_movement_aggregates_fast_v2.csv`，模型产物路径切到 `models/artifacts_fast_v2`，目标为 `arrival_flow / mean_speed / queue_veh`。
2. `prediction/movement_collector.py` 已经把短进口 edge 作为停止线/放行检测点，同时把上游 predecessor edges 组合为功能性进口道检测区，用于 arrival 和 queue 统计。这说明之前“短进口道 edge 导致预测状态失真”的核心问题已经被工程化处理。
3. `sim/movement_tools.py` 已经能从 SUMO 信号灯 connection 构建 movement 字典，movement 粒度为 `tls_id + incoming_edge + turn_type + outgoing_edge`，并追踪上游检测区、绿灯相位、link index 等元数据。
4. `prediction/torch_models.py` 已有 `SpatioTemporalTransformerForecaster`，并通过 `build_torch_model(kind="transformer_v2")` 注册为 Transformer V2。该模型包含实体投影、时间编码、movement/entity embedding、temporal encoder 和 spatial encoder。
5. `prediction/training.py` 已经把 `transformer_v2` 纳入训练模型组，并设置了 control feature ablation；这对后续证明“信号控制特征/预测特征是否有效”很有价值。
6. `prediction/phase_aggregation.py` 已经能把 movement 级预测结果按 `tls_id + phase_id` 聚合为相位级 arrival、queue 和 pressure。这正好是后续强化学习信号控制的输入桥梁。

---

## 1. 当前仍需完善的地方

### 1.1 文档与配置状态不一致，需要先修正

**问题**：README 仍描述为 edge 级预测和 Transformer V1；但当前配置和代码已经进入 movement 级和 Transformer V2。  
**影响**：后续成员或 Codex 容易按旧命令和旧数据路径操作，导致训练、服务加载和看板展示不一致。

**处理建议**：

- 更新 `README.md`：
  - 将“Transformer V1”改为“Transformer V1 / Transformer V2”。
  - 将预测对象改为“movement 级信号灯进口转向流”。
  - 新增 `batch_movement_aggregates_fast_v2.csv`、`models/artifacts_fast_v2`、`reports/fast_v2_training/metrics.csv` 的说明。
  - 明确当前默认训练入口：`python -m prediction.training --csv data/raw/batch_movement_aggregates_fast_v2.csv --artifact-dir models/artifacts_fast_v2 --report-dir reports/fast_v2_training`。
- 更新 `configs/prediction_config.json`：
  - 当前 `model` 字段仍是 `ha_baseline`。如果 V2 已经可用，建议不要直接写死为 `transformer_v2`，而是新增：
    - `active_model_from_registry: true`
    - `fallback_model: ha_baseline`
    - `preferred_model: transformer_v2`
  - 服务端优先读取 `model_registry.json`，没有 registry 或加载失败才 fallback。

### 1.2 movement_config 需要质量审查和再生成机制

**问题**：`configs/movement_config.json` 很大，属于由路网和 observed_edges 生成的派生文件。它现在可以保留，但必须增加质量审查，否则 RL 会直接继承错误 movement。

**需要补的审查项**：

- movement 总数、每个 TLS 的 movement 数、每个 TLS 的 left/through/right 分布。
- `zone_quality = short_upstream` 的比例，若超过 20%，要在报告中解释。
- `phase_id = -1` 或 `green_phase_ids = []` 的 movement 数，必须为 0 或有明确豁免。
- arrival 检测区和 queue 检测区是否覆盖足够长度，建议 `zone_length_m >= 80m`。
- 同一个 `movement_id` 是否唯一。

**建议新增命令**：

```bash
python -m sim.scripts.inspect_movement_catalog \
  --movement-config configs/movement_config.json \
  --out reports/movement_catalog_quality.json
```

验收标准：输出中必须包含：`movement_count`、`tls_count`、`turn_type_counts`、`phase_missing_count`、`short_upstream_count`、`zone_quality_summary`。

### 1.3 批量仿真数据需要建立 QA 门槛

**问题**：已有 movement 级采集，但还需要证明训练数据不是“全 0、低方差、无事故覆盖、无相位扰动覆盖”。

**建议新增或强化 `sim/scripts/diagnose_movement_data.py` 的输出**：

- 全局记录数、run 数、scenario 数、movement 数。
- 每个目标字段的均值、分位数、0 值比例：`arrival_flow / mean_speed / queue_veh`。
- 按 turn_type 分组的流量和排队分布。
- 按 tls_id 分组的样本量和拥堵比例。
- 按 `incident_flag`、`event_type`、`signal_variant` 分组的样本量。
- `train / val / test` 是否按 run_id 或 scenario_id 切分，而不是随机滑窗切分。

**硬性门槛**：

- 每个主要 TLS 至少有 30 个有效时间窗。
- 事故样本占比建议 10%–30%，不能只有少量 token。
- 每个 target 的非零比例至少要能支撑训练：`arrival_flow` 非零比例建议 > 15%，`queue_veh` 非零比例建议 > 5%。
- test set 必须包含未见过的需求强度、随机种子和至少一种事故场景。

### 1.4 Transformer V2 还缺显式空间拓扑约束

**当前优点**：V2 已经把 movement 当作 entity，并引入 temporal encoder + spatial encoder，已经比 flatten 版 Transformer V1 更适合 movement 级预测。

**仍需完善**：当前空间 encoder 更像“全连接 movement 注意力”，还没有显式利用路网拓扑。建议后续补一个 movement graph：

- `same_tls`：同一个信号灯下的 movement 相连。
- `same_incoming_edge`：同一进口道的左/直/右相连。
- `same_phase`：同一绿灯相位服务的 movement 相连。
- `upstream_downstream`：上游 movement 与下游相邻 movement 相连。
- `conflict`：冲突流向相连，可作为负相关或 conflict mask。

**MVP 处理方式**：先生成 `data/processed/movement_graph.json`，不用立刻改模型；RL 阶段可以直接用这个图做邻域聚合。后续再把它做成 Transformer V2 的 attention bias。

### 1.5 评估报告需要分维度，不要只看 overall MAE

建议新增以下评估表：

| 维度 | 必需指标 | 目的 |
|---|---|---|
| target | MAE/RMSE/WAPE by arrival_flow, mean_speed, queue_veh | 找出哪个目标拖后腿 |
| horizon | h5/h10/h15 | 判断远期预测是否塌陷 |
| turn_type | left/through/right | 检查左转小样本问题 |
| tls_id | 每个路口 | 找出表现最差的控制点 |
| incident | normal vs incident | 支撑事故推演场景 |
| control feature | with_control vs without_control | 支撑“控制特征有用” |
| model | HA/XGBoost/LSTM/V1/V2 | 支撑 Transformer V2 的价值 |

验收门槛：Transformer V2 至少在 `queue_veh` 或 `arrival_flow` 的 h5/h10 上优于 HA 和 LSTM；若 overall 不占优，也必须说明它在哪些场景或目标上占优。

### 1.6 在线服务需要公开预测质量和 fallback 状态

当前服务有 fallback 机制是好事，但看板和 API 应当明确暴露：

- `active_model`
- `fallback_used`
- `history_size / history_required`
- `model_artifact_path`
- `prediction_latency_ms`
- `phase_summary_available`

否则演示时模型没加载成功也可能“看起来在预测”，但实际上只是 HA baseline。

---

## 2. 下一步强化学习信号控制的总路线

强化学习不要一开始做全网多路口。建议采用 **单路口安全闭环 → 单路口预测增强 → 多路口共享策略** 的递进路线。

### 2.1 RL 阶段目标

第一阶段目标不是追求复杂算法，而是建立可复现实验闭环：

```text
SUMO 仿真 → movement 级采集 → phase 聚合 → RL 选择相位 → TraCI 执行动作 → 指标评估 → 与 Webster 固定配时对比
```

第一阶段只控制 1 个 TLS，例如优先选：

- `12254671324`：movement 数据和人工筛选信息较完整。
- `12260384797`：复杂度较高，适合第二个实验点。
- `J42` 或 `J55`：如果局部拥堵更明显，可以作为事故场景对照点。

### 2.2 不推荐直接端到端控制全部相位程序

不要让 RL 直接生成任意红黄绿字符串。原因：

- 容易产生冲突相位。
- 黄灯/全红过渡不安全。
- 动作空间过大，训练不稳定。
- 不便与已有 Webster 信号程序对比。

**推荐动作空间**：在已有合法绿灯相位中选择“保持当前相位”或“切换到某个合法绿灯相位”。

---

## 3. RL 环境设计

### 3.1 新增目录结构

建议新增：

```text
rl/
  __init__.py
  env.py
  state_builder.py
  reward.py
  phase_controller.py
  baselines.py
  train_dqn.py
  evaluate_policy.py
  replay_buffer.py          # 若不用 stable-baselines3，则需要
  policy_io.py

configs/
  rl_signal_config.json

reports/rl_signal_control/
  metrics.csv
  episode_summary.json
  figures/
```

### 3.2 `configs/rl_signal_config.json` 建议内容

```json
{
  "sumo_cfg": "intersection.sumocfg",
  "net_file": "data/processed/czq_tls_webster.net.xml",
  "route_file": "czq_demand.rou.xml",
  "movement_config": "configs/movement_config.json",
  "prediction_config": "configs/prediction_config.json",
  "target_tls_id": "12254671324",
  "control_interval_s": 10,
  "sample_interval_s": 60,
  "warmup_s": 300,
  "episode_s": 3600,
  "min_green_s": 10,
  "max_green_s": 60,
  "yellow_s": 3,
  "all_red_s": 1,
  "use_prediction_features": true,
  "prediction_horizons": [5, 10, 15],
  "reward_weights": {
    "queue": 1.0,
    "waiting": 0.5,
    "pressure": 0.4,
    "throughput": 0.3,
    "switch": 0.05,
    "emission": 0.0
  },
  "train": {
    "algorithm": "dqn",
    "episodes": 200,
    "seed": 42,
    "learning_rate": 0.0005,
    "gamma": 0.99,
    "batch_size": 64,
    "buffer_size": 50000,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay_steps": 30000
  }
}
```

### 3.3 环境类接口

文件：`rl/env.py`

```python
class SignalControlEnv:
    def __init__(self, config_path: str | Path): ...
    def reset(self, seed: int | None = None) -> np.ndarray: ...
    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]: ...
    def close(self) -> None: ...
```

如果使用 Gymnasium：

```python
class SumoSignalControlEnv(gym.Env):
    observation_space: gym.spaces.Box
    action_space: gym.spaces.Discrete
```

### 3.4 状态空间

状态按 phase 聚合，不直接把所有 movement 原样拼给 DQN。推荐状态：

```text
current_phase_onehot
phase_elapsed_s / max_green_s
for each legal green phase:
    current_queue_sum
    current_arrival_flow_sum
    current_discharge_flow_sum
    current_mean_speed_mean
    predicted_arrival_h5_sum
    predicted_queue_h5_mean
    predicted_arrival_h10_sum
    predicted_queue_h10_mean
    predicted_arrival_h15_sum
    predicted_queue_h15_mean
incident_flag_any
```

如果首版不接 Transformer V2，则预测字段填 0 或去掉，做 “no_prediction” baseline。

### 3.5 动作空间

对目标 TLS 提取所有合法绿灯相位 `green_phase_ids`。设共有 N 个绿灯相位：

```text
action = 0: keep current green phase
action = 1..N: request switch to green_phase_ids[action-1]
```

动作执行规则：

1. 若当前绿灯未达到 `min_green_s`，则忽略 switch，只保持当前相位。
2. 若超过 `max_green_s`，强制切换到非当前且 pressure 最大的相位，避免死锁。
3. 切换前插入 `yellow_s` 和 `all_red_s` 过渡。
4. 只允许切换到 movement_config 中存在服务 movement 的绿灯相位。

### 3.6 Reward 设计

首版 reward 不要太复杂，建议：

```text
reward =
  - w_queue    * normalized_queue
  - w_waiting  * normalized_waiting_time
  - w_pressure * normalized_pressure
  + w_through  * normalized_throughput
  - w_switch   * switch_penalty
```

其中：

```text
normalized_queue = sum(queue_veh for target tls phases) / queue_norm
normalized_pressure = max(phase_queue + phase_arrival) / pressure_norm
normalized_throughput = discharge_vehicle_count / throughput_norm
switch_penalty = 1 if action caused a phase switch else 0
```

首版不要把碳排放放进 reward，先作为评价指标。等 DQN 能稳定优于 Webster，再加入 emission 权重。

---

## 4. RL 与 Transformer V2 的衔接方式

### 4.1 不要让 RL 直接训练预测模型

预测模型保持冻结，作为状态增强器。RL 的训练目标是控制策略，不要同时更新 Transformer V2，否则调试难度会大幅上升。

### 4.2 在线调用流程

```text
MovementRealtimeCollector 每 60 秒生成 movement snapshot
PredictionService 读取最近 history_steps=12 个窗口
Transformer V2 输出未来 15 步 movement 预测
phase_aggregation 聚合成每个 TLS、每个 phase 的 pressure
RL state_builder 读取当前 phase 状态 + predicted phase pressure
DQN 输出 action
phase_controller 执行动作
```

### 4.3 做两个版本对比

必须做 ablation：

| 实验名 | 状态输入 | 目的 |
|---|---|---|
| RL-no-pred | 只用当前 queue / flow / phase | 基础 RL |
| RL-pred-v2 | 当前状态 + Transformer V2 h5/h10/h15 预测 | 证明预测模块对控制有用 |
| Webster | 固定配时 | 工程基线 |
| MaxPressure | 当前 phase pressure 贪心 | 强启发式基线 |

如果 RL-pred-v2 比 RL-no-pred 没有显著优势，不要在论文里硬说预测提升控制；可以改写为“预测模块支撑事故推演和趋势预警，控制模块当前采用实时状态闭环”。

---

## 5. 基线控制器

### 5.1 Webster 固定配时

当前仓库已有 Webster 风格固定配时路网，它是首要基线。

### 5.2 MaxPressure 启发式

新增 `rl/baselines.py`：

```python
def choose_max_pressure_phase(phase_summary: dict, current_phase: int, min_green_ok: bool) -> int:
    if not min_green_ok:
        return current_phase
    return argmax_phase_by(queue_sum + arrival_sum)
```

MaxPressure 很适合做 RL 的强基线，因为它不需要训练，也能体现 phase 聚合数据是否有效。

### 5.3 Actuated-like 控制

可选：若当前相位仍有 queue 且未达到 max_green，则延长；否则切到 pressure 最大相位。

---

## 6. Codex 实现任务拆解

### Task A：补齐 RL 配置与目录

新增文件：

```text
configs/rl_signal_config.json
rl/__init__.py
rl/reward.py
rl/state_builder.py
rl/phase_controller.py
rl/env.py
rl/baselines.py
```

验收：

```bash
python -c "from rl.env import SumoSignalControlEnv; print('rl env import ok')"
```

### Task B：实现 phase_controller

职责：

- 读取目标 TLS 的当前相位、当前 RYG state、合法 green phases。
- 执行 `keep` 或 `switch`。
- 保证 min_green、max_green、yellow、all-red 过渡。
- 通过 TraCI 调用控制 SUMO 信号灯。

验收：

```bash
python -m rl.phase_controller --config configs/rl_signal_config.json --smoke-test
```

输出必须包含：当前 phase、合法 green phases、一次 keep、一次 switch。

### Task C：实现 state_builder

职责：

- 读取 MovementRealtimeCollector 最近 snapshot。
- 读取 phase_aggregation 的预测结果。
- 构造固定长度 np.ndarray 状态向量。
- 输出 feature_names，便于 debug。

验收：

```bash
python -m rl.state_builder --config configs/rl_signal_config.json --smoke-test
```

输出：state shape、feature count、前 20 个 feature name。

### Task D：实现 reward

职责：

- 输入当前 snapshot、上一步 snapshot、action_info。
- 输出 reward 与分项信息。

建议返回：

```python
{
  "reward": -1.23,
  "queue_term": -0.7,
  "waiting_term": -0.2,
  "throughput_term": 0.1,
  "switch_penalty": -0.03
}
```

验收：reward 不出现 NaN，且拥堵越严重 reward 越低。

### Task E：实现单路口环境

职责：

- 启动 SUMO。
- warmup。
- 每个 control interval 执行动作。
- 每个 step 收集 movement 数据。
- 返回 Gym 风格 obs/reward/done/info。

验收：

```bash
python -m rl.env --config configs/rl_signal_config.json --smoke-test --steps 20
```

输出：20 次 step 的 action、reward、queue、throughput，且 SUMO 能正常关闭。

### Task F：实现 MaxPressure 和 Webster 评估

先不要训练 RL，先让基线跑起来。

```bash
python -m rl.evaluate_policy --config configs/rl_signal_config.json --policy webster --episodes 5
python -m rl.evaluate_policy --config configs/rl_signal_config.json --policy max_pressure --episodes 5
```

输出：

```text
reports/rl_signal_control/webster_metrics.csv
reports/rl_signal_control/max_pressure_metrics.csv
```

### Task G：实现 DQN 训练

首版可以用 PyTorch 自写 DQN，也可以引入 `stable-baselines3`。为了节省开发时间，建议用 Gymnasium + Stable-Baselines3：

```bash
pip install gymnasium stable-baselines3
```

训练命令：

```bash
python -m rl.train_dqn --config configs/rl_signal_config.json --timesteps 100000
```

输出：

```text
models/artifacts_rl/dqn_signal_single_tls.zip
reports/rl_signal_control/dqn_training_summary.json
```

### Task H：预测增强 RL 对比

训练两组：

```bash
python -m rl.train_dqn --config configs/rl_signal_config.json --timesteps 100000 --use-prediction false
python -m rl.train_dqn --config configs/rl_signal_config.json --timesteps 100000 --use-prediction true
```

评估：

```bash
python -m rl.evaluate_policy --config configs/rl_signal_config.json --policy dqn_no_pred --episodes 20
python -m rl.evaluate_policy --config configs/rl_signal_config.json --policy dqn_pred_v2 --episodes 20
```

验收指标：

- `avg_queue_veh` 低于 Webster。
- `avg_waiting_time_s` 低于 Webster。
- `throughput` 不低于 Webster。
- 事故场景下 `recovery_time_s` 低于 Webster 或 MaxPressure。

---

## 7. 实验指标设计

RL 阶段至少输出：

| 指标 | 定义 | 用途 |
|---|---|---|
| avg_queue_veh | 控制路口相关 movement 平均排队车辆数 | 主指标 |
| max_queue_veh | 最大排队车辆数 | 防止局部极端拥堵 |
| avg_waiting_time_s | 车辆平均等待时间 | 控制效果 |
| throughput | 单位时间通过停止线车辆数 | 通行效率 |
| avg_speed_kmh | 控制区域平均速度 | 运行状态 |
| switch_count | 相位切换次数 | 防止频繁切换 |
| yellow_time_ratio | 黄灯/全红时间占比 | 安全效率 |
| incident_recovery_time_s | 事故后恢复到正常阈值所需时间 | 韧性指标 |
| emission_proxy | 怠速时间或 CO2 输出，如果可取 | 绿色指标 |

---

## 8. 最小可交付版本定义

如果时间紧，按下面 MVP 收敛：

1. 单路口 `target_tls_id = 12254671324`。
2. 状态只用当前 phase + per-phase queue / arrival / speed。
3. 动作只做 keep / switch-to-valid-green-phase。
4. reward 只用 queue + throughput + switch penalty。
5. 跑 Webster、MaxPressure、DQN 三组对比。
6. 再加一次 Transformer V2 预测增强状态对比。

MVP 完成后，论文或展示中可以形成清晰闭环：

```text
高保真 movement 级数字孪生感知
→ Transformer V2 短时预测
→ 相位级需求/排队聚合
→ 强化学习信号控制
→ 与 Webster 固定配时和事故场景对比
```

---

## 9. 风险与兜底策略

### 风险 1：DQN 训练不稳定

兜底：使用 MaxPressure 作为控制算法，RL 作为探索性模块。MaxPressure 也能支撑“智能控制”演示。

### 风险 2：预测增强没有提升控制

兜底：保留 RL-no-pred 为主控制器，把 Transformer V2 用于事故预警和趋势推演。

### 风险 3：多路口控制太复杂

兜底：只做一个主路口 + 周边路口固定 Webster；展示局部瓶颈区控制改善。

### 风险 4：相位切换出现黄灯/冲突问题

兜底：只在已有 SUMO `tlLogic` 的合法 green phase 之间切换，不自定义任意 RYG 字符串。

---

## 10. 推荐开发顺序

```text
Day 1: 更新 README、配置、movement 数据 QA
Day 2: RL 配置、state_builder、reward、phase_controller
Day 3: 单路口 env smoke test
Day 4: Webster / MaxPressure 评估脚本
Day 5-6: DQN 训练与评估
Day 7: 接入 Transformer V2 预测增强状态
Day 8: 事故场景与正常场景对比
Day 9: 看板/API 展示 phase-level control result
Day 10: 汇总报告与竞赛展示材料
```

---

## 11. 给 Codex 的直接执行提示词

请在当前仓库 `xuejiali1234/jianshe` 上继续实现强化学习信号灯控制。当前仓库已经完成 movement 级交通流预测和 Transformer V2 调试。请不要重构已有预测链路，而是在其上新增 RL 控制模块。

具体要求：

1. 更新 README，使其与 movement 级预测、Transformer V2、fast_v2 数据路径一致。
2. 新增 `configs/rl_signal_config.json`。
3. 新增 `rl/` 包，包含：`env.py`、`state_builder.py`、`reward.py`、`phase_controller.py`、`baselines.py`、`train_dqn.py`、`evaluate_policy.py`。
4. 第一版只控制一个信号灯，默认 `target_tls_id = 12254671324`。
5. 动作空间为 keep 或切换到合法 green phase；必须强制 min_green、max_green、yellow、all_red 安全约束。
6. 状态空间先使用当前 phase、phase_elapsed、per-phase queue、arrival、speed；再通过开关加入 Transformer V2 的 h5/h10/h15 预测相位 pressure。
7. reward 使用 queue、waiting、throughput、switch penalty 的加权组合，不要一开始加入复杂碳排项。
8. 实现 Webster、MaxPressure、DQN-no-pred、DQN-pred-v2 四组评估。
9. 所有评估输出到 `reports/rl_signal_control/`。
10. 所有模型输出到 `models/artifacts_rl/`。
11. 每个新增脚本都必须支持 `--smoke-test` 或最小运行命令。
12. 不要破坏现有 FastAPI 看板和预测 API；RL 是新增模块，后续再接入看板。

首批验收命令：

```bash
python -m rl.env --config configs/rl_signal_config.json --smoke-test --steps 20
python -m rl.evaluate_policy --config configs/rl_signal_config.json --policy webster --episodes 2
python -m rl.evaluate_policy --config configs/rl_signal_config.json --policy max_pressure --episodes 2
python -m rl.train_dqn --config configs/rl_signal_config.json --timesteps 20000
python -m rl.evaluate_policy --config configs/rl_signal_config.json --policy dqn --episodes 5
```

---

## 12. 参考依据

- 当前仓库 README：项目为 FastAPI + SUMO + TraCI + WebSocket 的实时仿真与预测展示系统，并提供离线批量仿真、滑窗数据集、模型训练评估和事故场景对比能力。
- 当前 `prediction_config.json`：已配置 `observation_level = movement`、fast_v2 数据路径、目标字段 `arrival_flow / mean_speed / queue_veh`。
- 当前 `movement_collector.py`：已经把短进口 edge 作为 stopbar/discharge detector，同时用上游功能区统计 arrival 和 queue。
- 当前 `torch_models.py`：已经注册 `transformer_v2 = SpatioTemporalTransformerForecaster`。
- 当前 `phase_aggregation.py`：已经能把 movement 级预测聚合为 TLS-phase 级 arrival、queue 和 pressure。
- SUMO TraCI 官方文档：TraCI 支持运行时交通灯控制；`setPhase` 可切换当前信号相位，`getControlledLinks` 可获得由交通灯控制的 connection/link index。强化学习控制模块应只在合法相位间切换，避免任意生成冲突灯色。
