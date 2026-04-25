# 交通流预测与 SUMO 数字孪生看板

本项目面向城市道路短时交通流预测实验，基于 `FastAPI + SUMO + TraCI + WebSocket + 前端地图看板` 构建实时仿真与预测展示系统，并提供离线批量仿真、滑窗数据集构建、模型训练评估和事故场景对比能力。

当前工程重点不是只训练一个模型，而是搭建一条可复现实验链路：

1. 使用 SUMO 路网和需求文件生成仿真交通流。
2. 对信号灯进口道检测路段进行 60 秒粒度聚合采样。
3. 构建短时预测数据集。
4. 训练和评估 HA、XGBoost、LSTM、Transformer V1。
5. 将训练后的模型接入同一个 Web 服务，在地图看板中展示实时预测和事故对比。

## 当前状态

- 后端服务入口：`app.py`
- 默认服务地址：`http://127.0.0.1:8000`
- 仿真平台：Eclipse SUMO + TraCI
- 后端框架：FastAPI
- 前端形式：静态页面 + WebSocket 实时更新
- 预测对象：当前信号灯进口道检测路段
- 采样间隔：60 秒
- 历史窗口：12 步
- 预测步长：15 步
- 当前基础需求系数：`base_demand_factor = 0.25`
- 默认路网：优先使用 `data/processed/czq_tls_webster.net.xml`，不存在时回退到基础路网

> 说明：训练数据、模型权重、批量仿真结果和图表报告通常体积较大，默认不上传 Git，仅保留代码、配置、基础路网和说明文档。

## 主要功能

### 1. 实时 SUMO 看板

运行 `python app.py` 后，系统会启动 FastAPI 服务，并在后台启动 SUMO 仿真。前端看板通过 WebSocket 接收实时仿真状态，包括车辆、速度、排队、排放、事故状态和预测结果。

### 2. 交通预测服务

预测模块支持统一接口：

- `HA / Last-Value baseline`
- `XGBoost`
- `LSTM`
- `Transformer V1`

当训练模型加载失败或历史窗口不足时，系统会自动回退到 HA baseline，保证看板不中断。

### 3. 批量仿真数据生成

批量脚本支持不同需求强度和事故场景：

- `S1_normal`
- `S2_peak`
- `S3_congested`
- `S4_incident`

事故场景采用容量下降代理方式，在指定时间窗口内降低目标路段速度，用于生成事故扰动下的训练样本和对比实验。

### 4. 信号灯控制路网

项目包含基于人工筛选交叉口和 Webster 思路生成的固定配时信号灯路网。信号灯路网用于让仿真更接近真实城市交叉口运行状态。

### 5. 训练闭环

训练流程包括：

- 从长表 CSV 构建滑窗数据集
- 按 `run_id` 分段生成样本，避免跨仿真轮次拼窗
- 训练 HA、XGBoost、LSTM、Transformer V1
- 输出 MAE、RMSE、WAPE 等指标
- 保存最佳模型供服务加载

## 快速开始

### 1. 创建环境

推荐使用 Conda 创建 Python 3.10 环境：

```cmd
conda create -n traffic_pred python=3.10 -y
conda activate traffic_pred
```

安装 Web 与数据处理依赖：

```cmd
pip install -r requirements_web.txt
```

PyTorch 建议按本机 CUDA 版本单独安装。例如 CPU 版本：

```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 2. 配置 SUMO Python 工具

如果已经安装 SUMO，例如安装在 `D:\SUMO`，需要让 Python 能找到 TraCI：

```cmd
set SUMO_HOME=D:\SUMO
set PYTHONPATH=%SUMO_HOME%\tools;%PYTHONPATH%
```

验证：

```cmd
sumo --version
python -c "import traci, sumolib; print('SUMO tools OK')"
```

### 3. 启动系统

```cmd
python app.py
```

浏览器打开：

```text
http://127.0.0.1:8000
```

## 常用命令

### 生成 Webster 信号灯路网

```cmd
python -m sim.scripts.build_webster_tls_net --overwrite
```

输出通常位于：

- `data/processed/czq_tls_webster.net.xml`
- `data/processed/czq_tls_webster_summary.json`

### 运行批量仿真

```cmd
python -m sim.scripts.run_batch_sumo --overwrite
```

输出通常位于：

- `data/raw/batch_edge_aggregates.csv`
- `data/raw/scenarios/manifest.csv`

### 训练预测模型

训练入口位于：

```text
prediction/training.py
```

训练完成后，模型产物默认写入：

```text
models/artifacts/
```

评估结果默认写入：

```text
reports/
```

## API 接口

### 预测配置

```http
GET /api/prediction/config
```

返回观测路段、历史窗口、预测步长、基础需求系数、当前 active model 等信息。

### 最新预测

```http
GET /api/prediction/latest
```

返回最近一次 60 秒聚合观测和预测结果。

### 在线预测

```http
POST /api/predict
```

输入一个历史窗口，返回未来 15 步预测结果。

### 场景列表

```http
GET /api/prediction/scenario-runs
```

返回可用于正常/事故对比的批量仿真 run 信息。

### 事故对比

```http
POST /api/prediction/scenario-compare
```

输入正常 run、事故 run、路段和模型，返回预测曲线和差值摘要。

## 目录结构

```text
.
├── app.py
├── configs/
├── data/
├── models/
├── prediction/
├── reports/
├── sim/
├── static/
├── czq.net.xml
├── czq_demand.rou.xml
├── intersection.sumocfg
├── gui-settings.xml
├── vtypes.add.xml
├── requirements_web.txt
└── README.md
```

## 文件与目录说明

### 根目录

- `app.py`：主服务入口，负责启动 FastAPI、SUMO 后台仿真、WebSocket 推送和预测 API。
- `czq.net.xml`：基础 SUMO 路网文件。
- `czq_demand.rou.xml`：基础交通需求 route 文件。
- `intersection.sumocfg`：SUMO 仿真配置文件。
- `gui-settings.xml`：SUMO GUI 显示配置。
- `vtypes.add.xml`：SUMO 车辆类型配置。
- `requirements_web.txt`：Web 服务、数据处理、传统机器学习和绘图依赖。
- `交通流预测实现流程文档_仿真数据_Transformer短时预测_最终版_v2.docx`：项目规划与实现流程文档。
- `.gitignore`：Git 忽略规则，排除训练数据、模型权重、日志和缓存。
- `README.md`：项目说明文档。

### `configs/`

- `prediction_config.json`：预测配置，包括观测路段、采样间隔、历史窗口、预测步长、目标字段、基础需求系数和默认路网。
- `signal_control_config.json`：信号灯控制配置，包括人工筛选的信号灯交叉口和禁用交叉口。

### `prediction/`

- `config.py`：预测配置读取与默认值管理。
- `schemas.py`：预测 API 的输入输出数据结构。
- `baselines.py`：HA / Last-Value 基线预测模型。
- `collector.py`：实时仿真数据聚合采集逻辑。
- `dataset.py`：从批量仿真长表构建滑窗训练数据集。
- `metrics.py`：MAE、RMSE、WAPE 等评价指标。
- `torch_models.py`：LSTM 和 Transformer V1 模型结构。
- `model_io.py`：模型保存、加载和 registry 管理。
- `service.py`：在线预测服务封装，负责模型选择、fallback 和统一输出。
- `training.py`：离线训练、评估、保存模型和生成报告的训练入口。
- `__init__.py`：Python 包标识文件。

### `sim/`

- `network_tools.py`：路网路径解析工具，优先选择生成后的信号灯路网。
- `route_tools.py`：交通需求 route 文件处理工具，支持需求倍率和基础需求系数。
- `signal_timing.py`：信号灯路网生成与 Webster 风格固定配时逻辑。
- `validation.py`：SUMO 环境、文件可读性和观测路段合法性校验。
- `scripts/build_webster_tls_net.py`：生成信号灯路网的命令行脚本。
- `scripts/run_batch_sumo.py`：批量运行 SUMO 场景并输出聚合数据的命令行脚本。
- `scripts/__init__.py`、`scripts/.gitkeep`：脚本目录包标识与占位文件。
- `__init__.py`：Python 包标识文件。

### `static/`

- `index.html`：前端看板页面结构。
- `css/style.css`：前端样式，包含地图、指标卡片、预测面板和事故对比布局。
- `js/main.js`：前端逻辑，负责地图渲染、WebSocket 连接、车辆显示、路段高亮、模型切换和预测图表绘制。

### `data/`

- `README.md`：数据目录说明。
- `raw/`：实时与批量仿真的原始聚合数据输出目录。
- `processed/`：处理后的路网、信号灯路网和摘要文件输出目录。
- `datasets/`：训练数据集缓存目录。
- `archive/`：旧数据、旧训练产物和历史实验归档目录。

Git 默认只保留 `data/README.md` 和 `.gitkeep` 占位文件，不上传大型 CSV、缓存和归档数据。

### `models/`

- `README.md`：模型目录说明。
- `artifacts/`：训练后的模型权重、XGBoost joblib 文件和模型注册表。

Git 默认不上传 `models/artifacts/`，避免仓库体积过大。

### `reports/`

- `README.md`：报告目录说明。
- `metrics.csv`：模型评估指标。
- `p4_training_summary.json`：训练摘要。
- `figures/`：预测曲线、事故对比图和指标对比图。

Git 默认不上传生成的报告 CSV、JSON 和图片，仅保留说明文件。

## 数据说明

实时和批量仿真聚合数据通常包含以下字段：

- `run_id`：一次仿真运行或一个场景的编号。
- `timestamp`：采样时间。
- `step`：SUMO 仿真步。
- `edge_id`：检测路段编号。
- `flow`：采样窗口内通过或出现的车辆数近似值。
- `speed_mps`：平均速度，单位 m/s。
- `speed_kmh`：平均速度，单位 km/h。
- `queue`：排队或停车车辆数。
- `incident_flag`：是否处于事故扰动影响中。

训练时会按 `run_id` 分段构建滑窗样本，避免不同仿真运行之间的数据被拼接。

## Git 上传策略

本仓库上传：

- 代码
- 配置
- 基础 SUMO 文件
- 前端页面
- 项目说明文档

本仓库不上传：

- 批量仿真 CSV
- 实时运行日志
- 数据集缓存
- PyTorch 权重
- XGBoost joblib 文件
- 自动生成图表
- 历史归档数据

如果需要复现实验，应在本地重新运行批量仿真和训练流程。

## 注意事项

1. 运行前需要确保 SUMO 已安装，并且 `sumo --version` 可用。
2. TraCI 和 sumolib 来自 SUMO 的 `tools` 目录，需要加入 `PYTHONPATH`。
3. 如果 GitHub 能在浏览器打开但 Git 推送失败，通常是 Git 未配置代理。可为当前仓库配置：

```cmd
git config --local http.proxy http://127.0.0.1:7892
git config --local https.proxy http://127.0.0.1:7892
```

4. 当前模型训练结果依赖本地生成的数据。更改 `observed_edges`、`base_demand_factor` 或信号灯路网后，需要重新生成数据并重新训练模型。
