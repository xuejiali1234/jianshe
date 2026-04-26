# Codex 实现计划：将当前 edge 级交通流预测改造为 movement 级短时预测

> 目标读者：Codex / 开发实现者  
> 适用仓库：`xuejiali1234/jianshe`  
> 目标：解决“交叉口进口道 edge 很短导致预测状态失真”的问题，并把直行、左转、右转等 movement 级数据接入 Transformer 短时预测和后续强化学习信号控制。

---

## 0. 当前仓库判断

当前项目已经有完整雏形：

- `app.py`：FastAPI + SUMO + TraCI + WebSocket 看板入口。
- `configs/prediction_config.json`：当前仍使用 `observed_edges`，并配置了 `sample_interval_s=60`、`history_steps=12`、`horizon_steps=15`、`targets=[flow, speed, queue]`。
- `prediction/collector.py`：当前 `EdgeRealtimeCollector` 逐 edge 聚合 `flow / speed_mps / queue / incident_flag`。
- `prediction/dataset.py`：当前按 `edge_id` 透视成长表滑窗数据集。
- `prediction/torch_models.py`：已有 LSTM 和 `TransformerForecaster`，当前 Transformer 是时间序列 flatten 输入版本。
- `sim/scripts/run_batch_sumo.py`：已有批量仿真、需求倍率、事故降速代理场景。
- `configs/signal_control_config.json`：已有人工筛选的信号灯 junction 列表，应作为 movement 级检测的主入口。

当前问题不是模型结构优先问题，而是观测对象定义问题：现在用短进口 edge 作为预测节点，会把“停止线附近瞬时放行状态”误当成“完整进口道交通需求与排队状态”。

---

## 1. 实现总原则

### 1.1 不建议直接拉长 SUMO 路网 edge

不要通过修改路网几何、强行拉长短 edge 来解决。这样容易破坏 lane connection、traffic light link index、route 连通性和现有 Web 看板。

正确处理方式是：

```text
短 edge 保留为停止线/放行检测点；
上游 edge + 短 edge 组合成“功能性进口道检测区”；
按 movement 统计左转、直行、右转；
训练时所有 movement 作为多节点统一训练；
控制时按 phase 聚合 movement 预测结果。
```

### 1.2 movement 定义

`movement` 不是单纯 lane，也不是单纯 edge，而是：

```text
movement = tls_id + incoming_edge + turn_type + outgoing_edge
```

推荐 `movement_id` 格式：

```text
{tls_id}|{incoming_edge}|{turn_type}|{outgoing_edge}
```

示例：

```text
12254671324|E3|through|E5
12254671324|E3|left|E2
12254671324|E3|right|-E7
```

### 1.3 训练策略

不要把左转、直行、右转加总成一个流量；也不要每个 movement 单独训练一个模型。

采用：

```text
分 movement 采集；
统一 Transformer 训练；
多 movement、多目标输出；
进入 RL 信号控制时再按 phase_id 聚合。
```

---

## 2. 新增/修改文件清单

### 2.1 新增文件

#### `configs/movement_detection_config.json`

用途：定义 movement 检测参数。

建议内容：

```json
{
  "collector_mode": "movement",
  "tls_ids_source": "configs/signal_control_config.json",
  "sample_interval_s": 60,
  "history_steps": 12,
  "horizon_steps": 15,
  "approach_upstream_m": 120.0,
  "arrival_detector_m": 80.0,
  "stopbar_detector_m": 5.0,
  "queue_speed_threshold_mps": 0.1,
  "queue_time_threshold_s": 1.0,
  "targets": [
    "arrival_flow",
    "discharge_flow",
    "mean_speed_mps",
    "queue_veh"
  ],
  "input_features": [
    "arrival_flow",
    "discharge_flow",
    "mean_speed_mps",
    "queue_veh",
    "green_flag",
    "green_remaining_s",
    "incident_flag"
  ],
  "output_csv": "data/raw/batch_movement_aggregates.csv",
  "runtime_csv": "data/raw/realtime_movement_aggregates.csv"
}
```

#### `sim/movement_catalog.py`

用途：从信号灯路网中构建 movement 目录。

核心输出：

```python
@dataclass(frozen=True)
class MovementSpec:
    movement_id: str
    tls_id: str
    incoming_edge: str
    outgoing_edge: str
    turn_type: str              # left / through / right / uturn / unknown
    incoming_lanes: tuple[str, ...]
    outgoing_lanes: tuple[str, ...]
    link_indices: tuple[int, ...]
    green_phase_indices: tuple[int, ...]
    upstream_edges: tuple[str, ...]
    upstream_lanes: tuple[str, ...]
    approach_length_m: float
```

需要实现的函数：

```python
def build_movement_catalog(
    net_file: str | Path,
    tls_ids: list[str],
    approach_upstream_m: float = 120.0,
) -> list[MovementSpec]:
    ...
```

建议实现逻辑：

1. 使用 `sumolib.net.readNet(str(net_file), withPrograms=True)` 读取路网。
2. 遍历指定 `tls_ids`。
3. 对每个 TLS，读取 controlled connections / link index / incoming lane / outgoing lane。
4. 对每条 connection 提取：
   - incoming lane / incoming edge；
   - outgoing lane / outgoing edge；
   - link index；
   - turn direction。
5. 将相同 `(tls_id, incoming_edge, outgoing_edge, turn_type)` 聚合为一个 movement。
6. 向上游递归追踪 predecessor edges/lane，直到累计长度达到 `approach_upstream_m`。
7. 保存 `data/processed/movement_catalog.json`，方便调试和看板读取。

注意：如果 sumolib connection 无法稳定拿到 direction，则用几何角度兜底，或者先输出 `unknown`，但不得阻塞主流程。

#### `prediction/movement_collector.py`

用途：替代/扩展当前 `EdgeRealtimeCollector`，按 movement 聚合仿真观测。

核心类：

```python
class MovementRealtimeCollector:
    def __init__(
        self,
        config: PredictionConfig,
        movement_catalog: list[MovementSpec],
        csv_path: str | Path,
        run_id: str | None = None,
        scenario_id: str = "",
        seed: int | str = "",
        demand_scale: float | str = "",
        base_demand_factor: float | str = "",
        incident_type: str = "",
        incident_start_s: int | str = "",
        incident_end_s: int | str = "",
        affected_edges: list[str] | tuple[str, ...] | None = None,
    ):
        ...

    def record_step(self, traci_module, step: int, sim_time_s: float, incident_edges: set[str] | None = None) -> dict | None:
        ...
```

CSV 字段建议：

```python
MOVEMENT_CSV_FIELDS = [
    "run_id",
    "scenario_id",
    "seed",
    "demand_scale",
    "base_demand_factor",
    "incident_type",
    "incident_start_s",
    "incident_end_s",
    "affected_edges",
    "timestamp",
    "step",
    "node_id",
    "movement_id",
    "tls_id",
    "incoming_edge",
    "outgoing_edge",
    "turn_type",
    "phase_index",
    "arrival_flow",
    "discharge_flow",
    "mean_speed_mps",
    "speed_kmh",
    "queue_veh",
    "vehicle_count",
    "green_flag",
    "green_remaining_s",
    "incident_flag",
]
```

#### `sim/scripts/inspect_movement_catalog.py`

用途：开发时快速检查 movement 目录是否合理。

命令：

```bash
python -m sim.scripts.inspect_movement_catalog
```

输出：

```text
TLS 12254671324: movements=12, incoming_edges=4
  12254671324|E3|through|E5 lanes=2 upstream_len=120.0m phases=[0,2]
  ...
```

并写出：

```text
data/processed/movement_catalog.json
```

### 2.2 修改文件

#### `prediction/config.py`

新增字段，但保留原 `observed_edges`，保证旧流程不被破坏：

```python
collector_mode: str = "edge"  # edge | movement
movement_config_file: str = "configs/movement_detection_config.json"
observed_movements: list[str] = field(default_factory=list)
node_id_column: str = "edge_id"
target_column_map: dict[str, str] = field(default_factory=dict)
```

当 `collector_mode == "movement"` 时：

```python
node_id_column = "node_id"
target_column_map = {
    "flow": "arrival_flow",
    "arrival_flow": "arrival_flow",
    "discharge_flow": "discharge_flow",
    "speed": "mean_speed_mps",
    "mean_speed_mps": "mean_speed_mps",
    "queue": "queue_veh",
    "queue_veh": "queue_veh"
}
```

#### `prediction/dataset.py`

将 edge 级逻辑抽象为 node 级逻辑。

当前逻辑：

```python
edge_id
edge_ids = config.observed_edges
```

改为兼容：

```python
node_id_col = config.node_id_column  # edge_id or node_id
node_ids = config.observed_movements if movement mode else config.observed_edges
```

字段命名改为：

```python
feature_names = [f"{node_id}__{feature}" ...]
target_feature_names = [f"{node_id}__{target}" ...]
```

注意：保留 `edge_ids` 字段兼容旧 artifact，但新增 `node_ids`。服务层优先使用 `node_ids`。

#### `prediction/training.py`

新增命令参数：

```bash
--mode edge|movement
--csv data/raw/batch_movement_aggregates.csv
--dataset-dir data/datasets/movement_transformer
```

在 movement 模式下默认：

```text
csv_path = data/raw/batch_movement_aggregates.csv
dataset_dir = data/datasets/movement_transformer
```

保留当前 HA、XGBoost、LSTM、Transformer V1 训练流程。

#### `prediction/torch_models.py`

第一阶段不必大改 Transformer。当前 flatten Transformer 可继续使用。

只需保证 `input_size` 和 `output_size` 来自 movement 级 dataset：

```text
input_size = N_movements * F_features + tod_features
output_size = N_movements * K_targets
```

后续可升级 `MovementTransformerV2`，但本轮不要扩大范围。

#### `sim/scripts/run_batch_sumo.py`

新增参数：

```bash
--collector edge|movement
--movement-config configs/movement_detection_config.json
```

默认建议：

```python
collector = "movement"
output_csv = data/raw/batch_movement_aggregates.csv
```

但保留 edge 模式，旧命令仍可运行。

需要在 SUMO 启动后：

1. 构建或读取 movement catalog；
2. 初始化 `MovementRealtimeCollector`；
3. 每步调用 `record_step(...)`；
4. 输出 movement 长表。

#### `app.py`

根据配置选择 collector：

```python
if prediction_config.collector_mode == "movement":
    movement_catalog = load_or_build_movement_catalog(...)
    realtime_collector = MovementRealtimeCollector(...)
else:
    realtime_collector = EdgeRealtimeCollector(...)
```

WebSocket payload 中的 `prediction` 继续使用 `prediction_service.latest_prediction`，但 prediction nodes 改为支持：

```json
{
  "node_id": "tls|incoming|turn|outgoing",
  "movement_id": "...",
  "tls_id": "...",
  "turn_type": "left",
  "pred_arrival_flow": [...],
  "pred_queue_veh": [...]
}
```

#### `prediction/service.py` 和 `prediction/schemas.py`

兼容 `edge_id` 与 `node_id`：

- 输入窗口中允许 node 使用 `node_id`；
- 输出节点中优先返回 `node_id`；
- 如果旧模型 artifact 只有 `edge_ids`，仍按旧字段返回。

#### `static/js/main.js`

最低要求：不要因为预测节点从 `edge_id` 变成 `node_id` 而报错。

建议：

```javascript
const nodeId = node.node_id || node.edge_id;
```

movement 级结果先在预测面板展示，不强制画到地图上。地图高亮后续再做。

---

## 3. MovementRealtimeCollector 关键算法

### 3.1 初始化

每个 movement 初始化 accumulator：

```python
{
  movement_id: {
    "seen_vehicle_ids": set(),
    "arrival_vehicle_ids": set(),
    "discharge_vehicle_ids": set(),
    "speeds": [],
    "queue_counts": [],
    "vehicle_counts": [],
    "green_flags": [],
    "green_remaining_s": [],
  }
}
```

维护车辆状态：

```python
self.vehicle_last_edge: dict[str, str]
self.vehicle_last_movement: dict[str, str]
self.vehicle_seen_in_interval: dict[str, set[str]]
```

### 3.2 将车辆匹配到 movement

推荐函数：

```python
def infer_vehicle_movement(vehicle_id: str, traci_module) -> str | None:
    route = traci.vehicle.getRoute(vehicle_id)
    route_index = traci.vehicle.getRouteIndex(vehicle_id)
    current_edge = traci.vehicle.getRoadID(vehicle_id)

    # 跳过 internal edge
    if current_edge.startswith(":"):
        return cached_last_movement_or_none

    # 在 route 中寻找最近的 incoming_edge -> outgoing_edge 对
    # 如果 current_edge 是 incoming_edge，检查下一条 edge 是否 outgoing_edge
    # 如果 current_edge 是上游 edge，检查未来 route 是否经过某 movement 的 incoming_edge 和 outgoing_edge
```

索引结构：

```python
by_incoming_edge[incoming_edge] -> list[MovementSpec]
by_upstream_edge[upstream_edge] -> list[MovementSpec]
by_pair[(incoming_edge, outgoing_edge)] -> MovementSpec
```

匹配优先级：

1. 当前 edge 是 movement 的 incoming_edge，且 route 下一条 edge 是 outgoing_edge；
2. 当前 edge 是 movement 的 upstream_edge，且未来 route 包含 incoming_edge 后接 outgoing_edge；
3. 使用上一时刻缓存 movement；
4. 无法识别则跳过。

### 3.3 arrival_flow

定义：采样周期内进入功能性进口道检测区的车辆数。

MVP 实现：

```python
if vehicle_id 第一次出现在 movement 的 upstream_edges 或 incoming_edge 范围内:
    arrival_vehicle_ids.add(vehicle_id)
```

注意：这不是严格 E1 断面计数，但比短 edge 停止线计数更适合作为“到达需求”近似。

后续可升级为精确断面：车辆距离停止线从 `> arrival_detector_m` 变成 `<= arrival_detector_m` 时计数。

### 3.4 discharge_flow

定义：采样周期内从 incoming_edge 进入 outgoing_edge 的车辆数。

实现：

```python
last_edge = self.vehicle_last_edge.get(vehicle_id)
current_edge = traci.vehicle.getRoadID(vehicle_id)

if (last_edge, current_edge) in movement_pair_index:
    movement.discharge_vehicle_ids.add(vehicle_id)
```

这比短 edge 内部车辆数更能表达信号实际放行结果。

### 3.5 queue_veh

定义：功能性进口道检测区内，速度低于阈值的最大车辆数。

实现：

```python
if vehicle 在 movement approach zone 内 and speed <= queue_speed_threshold_mps:
    queue_count += 1
```

每个 step 记录一次 `queue_count`，周期输出：

```python
queue_veh = max(queue_counts)
```

### 3.6 mean_speed_mps

定义：采样周期内 movement approach zone 内车辆速度均值。

实现：

```python
speeds.append(traci.vehicle.getSpeed(vehicle_id))
mean_speed_mps = sum(speeds) / len(speeds) if speeds else 0.0
```

### 3.7 green_flag 与 green_remaining_s

对每个 movement，根据 `tls_id` 当前信号灯状态和 movement 的 `link_indices` 判断。

```python
state = traci.trafficlight.getRedYellowGreenState(tls_id)
green_flag = any(state[i] in "Gg" for i in movement.link_indices)
next_switch = traci.trafficlight.getNextSwitch(tls_id)
green_remaining_s = max(0.0, next_switch - sim_time_s) if green_flag else 0.0
```

---

## 4. 数据集结构

### 4.1 movement 长表

输出文件：

```text
data/raw/batch_movement_aggregates.csv
```

每 60 秒，每个 movement 一行。

示例字段：

```csv
run_id,scenario_id,seed,demand_scale,timestamp,step,node_id,movement_id,tls_id,incoming_edge,outgoing_edge,turn_type,arrival_flow,discharge_flow,mean_speed_mps,queue_veh,green_flag,green_remaining_s,incident_flag
```

### 4.2 训练张量

仍然使用当前滑窗结构：

```text
X: [B, history_steps, N_movements * F_features + 2]
y: [B, horizon_steps, N_movements * K_targets]
```

其中 `+2` 是 `tod_sin / tod_cos`。

### 4.3 推荐 targets

第一版：

```json
["arrival_flow", "mean_speed_mps", "queue_veh"]
```

第二版再加入：

```json
["discharge_flow"]
```

因为 `discharge_flow` 受当前信号配时强影响，直接作为预测目标可能让模型混合学习需求和控制结果。第一版先把需求、速度、排队预测稳住。

---

## 5. 批量仿真场景增强

当前 `run_batch_sumo.py` 已有：

```text
S1_normal
S2_peak
S3_congested
S4_incident
```

本轮建议保留这些场景，但增加 turn/movement 多样性。

### 5.1 首先保留当前全局 demand_scale

不要一次重写 route 生成逻辑。先保证 movement collector 可稳定产数。

### 5.2 第二阶段增加 route/turn 随机化

新增：

```python
Scenario.turn_bias: str = "balanced"  # balanced | left_heavy | through_heavy | asymmetric
Scenario.direction_bias: str = "balanced"  # ew_heavy | ns_heavy | balanced
```

如果当前 route 文件中的 flow id 或 route edges 可解析，则按 route 经过的 selected TLS movement 分类后乘倍率：

```text
left_heavy: left movement route flow * 1.5
through_heavy: through movement route flow * 1.3
asymmetric: 东西向 * 1.4，南北向 * 0.8
```

如果解析失败，则不阻塞主流程，只写 warning。

---

## 6. 训练与模型策略

### 6.1 不分开训练

所有 movement 统一进入一个模型。

原因：

- 左转、直行、右转共享空间和信号周期；
- 排队回溢会跨 movement 影响；
- 后续 RL 控制需要 phase 级需求，而 phase 级需求来自多个 movement 聚合；
- 统一模型更容易学习交叉口整体时空耦合。

### 6.2 Transformer V1 先不重构

当前 `TransformerForecaster` 可以作为 MVP。

只要 dataset 从 edge 级改为 movement 级，Transformer 输入输出维度会自动变化。

### 6.3 后续可选升级

新增 `MovementTransformerV2`，输入保持四维：

```text
[B, T, N, F]
```

加入：

```text
movement embedding
tls embedding
turn_type embedding
phase embedding
```

但这不是本次必须实现项。

---

## 7. 给 RL 信号控制的接口设计

训练输出保留 movement 粒度，但控制输入按相位聚合。

新增工具函数：

```python
def aggregate_predictions_by_phase(prediction_payload, movement_catalog):
    ...
```

输出：

```json
{
  "tls_id": "12254671324",
  "phases": [
    {
      "phase_index": 0,
      "movement_ids": ["..."],
      "pred_arrival_flow_sum": [...],
      "pred_queue_veh_sum": [...]
    }
  ]
}
```

RL 状态建议：

```text
current_queue_by_phase
current_arrival_flow_by_phase
pred_queue_by_phase_5min
pred_arrival_flow_by_phase_5min
current_phase_index
green_remaining_s
incident_flag_by_phase
```

本轮只实现聚合函数和 API，不必实现 RL 控制器。

---

## 8. API 与服务兼容性

### 8.1 `/api/prediction/config`

新增返回：

```json
{
  "collector_mode": "movement",
  "node_id_column": "node_id",
  "node_count": 128,
  "movement_catalog_path": "data/processed/movement_catalog.json"
}
```

### 8.2 `/api/prediction/latest`

movement 模式下返回：

```json
{
  "model": "transformer_v1",
  "horizon": [1,2,3],
  "nodes": [
    {
      "node_id": "12254671324|E3|through|E5",
      "movement_id": "12254671324|E3|through|E5",
      "tls_id": "12254671324",
      "turn_type": "through",
      "pred_arrival_flow": [...],
      "pred_queue_veh": [...]
    }
  ]
}
```

### 8.3 `/api/prediction/phase-aggregate`

新增可选接口：

```text
GET /api/prediction/phase-aggregate
```

返回当前 latest prediction 的 phase 聚合结果。

---

## 9. 命令验收流程

Codex 实现后，以下命令必须可运行。

### 9.1 生成信号灯路网

当前 README 写过 `--overwrite`，但脚本未必支持。Codex 需要确认并修复 README 或脚本参数。

```bash
python -m sim.scripts.build_webster_tls_net
```

### 9.2 检查 movement catalog

```bash
python -m sim.scripts.inspect_movement_catalog
```

验收：

```text
data/processed/movement_catalog.json 存在；
每个 manual_tls_junction 至少有 2 个 movement；
无重复 movement_id；
大部分 movement 有 upstream_edges；
```

### 9.3 小规模仿真产数

```bash
python -m sim.scripts.run_batch_sumo --collector movement --limit 2 --overwrite
```

验收：

```text
data/raw/batch_movement_aggregates.csv 存在；
CSV 中有 node_id / movement_id / turn_type / arrival_flow / queue_veh；
每个 run_id 至少有 20 个有效采样时刻；
```

### 9.4 构建数据集并训练

```bash
python -m prediction.training \
  --mode movement \
  --csv data/raw/batch_movement_aggregates.csv \
  --dataset-dir data/datasets/movement_transformer \
  --smoke-test
```

验收：

```text
reports/metrics.csv 存在；
models/artifacts/model_registry.json 存在；
Transformer V1 能训练完成或在数据不足时给出明确错误；
HA baseline 必须可用；
```

### 9.5 启动看板

```bash
python app.py
```

验收：

```text
http://127.0.0.1:8000 可打开；
WebSocket 不报 node_id/edge_id 字段错误；
/api/prediction/config 返回 collector_mode；
/api/prediction/latest 有 movement 级 nodes；
```

---

## 10. 最低验收标准

本次实现完成后，应满足：

1. 短进口 edge 不再作为唯一预测观测点。
2. 系统能输出 movement 级数据，至少包含：
   - arrival_flow
   - mean_speed_mps
   - queue_veh
   - green_flag
   - green_remaining_s
3. 左转、直行、右转作为不同 `movement_id` 分开记录。
4. 训练时所有 movement 进入同一个数据集和同一个 Transformer 模型。
5. 旧 edge 模式不被删除，出现问题时可以回退。
6. batch 仿真、dataset、training、app 四条链路都能跑通。

---

## 11. 需要避免的实现错误

### 错误 1：把左转/直行/右转简单加总

这样会丢失后续信号控制需要的相位需求信息。

### 错误 2：每个 movement 单独训练模型

这样会丢失转向之间、相位之间、上下游之间的耦合关系。

### 错误 3：只在停止线短 edge 上计数

这只能测放行结果，不代表上游到达需求。

### 错误 4：一次性重写整套项目

当前项目已经有可用链路。应增量实现：先保留 edge pipeline，再新增 movement pipeline。

### 错误 5：没有场景维度切分

训练、验证、测试应继续按 `run_id` 分割，不能随机切单个窗口，否则会数据泄漏。

---

## 12. Codex 任务提示词

可以直接把下面这段作为 Codex 执行提示：

```text
请在当前仓库中实现 movement 级交通流预测数据管线。不要删除现有 edge 级管线，要保持兼容。

背景：当前项目使用 SUMO + TraCI + FastAPI，已有 EdgeRealtimeCollector、batch_sumo、dataset、training、TransformerForecaster。当前问题是 observed_edges 中很多交叉口进口道 edge 很短，不适合作为预测观测点。需要把观测对象升级为 movement，即 tls_id + incoming_edge + turn_type + outgoing_edge。

请完成：
1. 新增 movement catalog 构建逻辑，从 configs/signal_control_config.json 的 manual_tls_junction_ids 和 data/processed/czq_tls_webster.net.xml 构建 movement 列表，输出 data/processed/movement_catalog.json。
2. 新增 MovementRealtimeCollector，基于 TraCI 车辆 route、current edge、last edge、speed 和 traffic light state，聚合 movement 级 arrival_flow、discharge_flow、mean_speed_mps、queue_veh、green_flag、green_remaining_s、incident_flag。
3. 修改 run_batch_sumo.py，支持 --collector edge|movement，movement 模式输出 data/raw/batch_movement_aggregates.csv。
4. 修改 prediction config 和 dataset，使其兼容 node_id/movement_id，不再硬编码 edge_id。旧 edge 模式仍可运行。
5. 修改 training.py，支持 --mode movement，并能基于 movement CSV 训练 HA、XGBoost、LSTM、Transformer V1。
6. 修改 service/schemas/app/static，至少保证 node_id prediction payload 不报错；旧 edge_id payload 继续兼容。
7. 添加 inspect_movement_catalog 脚本和必要 smoke test。

实现顺序：先 catalog，再 collector，再 batch CSV，再 dataset/training，最后 app/service/UI 兼容。

验收命令：
python -m sim.scripts.inspect_movement_catalog
python -m sim.scripts.run_batch_sumo --collector movement --limit 2 --overwrite
python -m prediction.training --mode movement --csv data/raw/batch_movement_aggregates.csv --dataset-dir data/datasets/movement_transformer --smoke-test
python app.py

注意：不要把左转/直行/右转合并成一个总流量；也不要每个 movement 单独训练模型。应当分 movement 采集，统一模型训练，后续再按 phase 聚合给信号控制。
```

---

## 13. 推荐实施优先级

### P0：必须完成

- `movement_catalog.py`
- `MovementRealtimeCollector`
- `run_batch_sumo --collector movement`
- `dataset.py` 支持 `node_id`
- `training.py --mode movement`

### P1：建议完成

- `/api/prediction/latest` 返回 movement 元信息
- `/api/prediction/phase-aggregate`
- `inspect_movement_catalog.py`
- README 增加 movement pipeline 命令

### P2：后续增强

- 精确 arrival detector 断面穿越判定
- E1/E2/E3 XML detector 生成与输出解析
- turn-ratio route 随机化
- MovementTransformerV2：显式 `[B,T,N,F]` 输入 + movement embedding
- RL 信号控制器接入 phase 聚合预测
