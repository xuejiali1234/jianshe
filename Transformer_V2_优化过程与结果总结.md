# Transformer V2 优化过程与结果总结

本文档用于整理当前项目中 Transformer V2 的调优过程、核心指标、阶段性结论和后续需要进一步分析的问题。文档目标不是论文正文，而是作为诊断材料，方便交给网页端 GPT 或其他模型继续分析。

## 1. 当前实验口径

当前预测实验使用的是 `full_v3` 数据口径。

| 项目 | 当前设置 |
|---|---|
| 数据版本 | `full_v3` |
| 数据文件 | `data/raw/batch_movement_aggregates_full_v3.csv` |
| 预测对象 | movement 级交通流 |
| movement 定义 | `tls_id + incoming_edge + turn_type + outgoing_edge` |
| 历史窗口 | 12 个 60s 窗口 |
| 预测步长 | 15 个 60s 窗口 |
| 预测目标 | `arrival_flow / mean_speed / queue_veh` |
| 主要对比模型 | Transformer V1 / Transformer V2 |
| V1 基本形式 | 扁平化时间 Transformer，输入为 `[B, L, N*F + time_features]` |
| V2 基本形式 | movement 级时空 Transformer，内部 reshape 为 `[B, L, N, F]` |

其中，`N` 为 movement 数量，当前约为 `151`；输出维度为 `N * 3 = 453`。

## 2. V2 优化目标

最初引入 Transformer V2 的目的，是希望模型能够比 V1 更充分利用 movement 之间的空间关系、信号相位控制信息以及上下游拓扑关系。

理论预期如下：

1. V1 将所有 movement 特征扁平化输入，能够学习时间依赖，但对 movement 结构没有显式建模。
2. V2 将输入还原为 `[时间, movement, 特征]` 结构，理论上可以分别建模时间依赖和空间依赖。
3. 信号相位、同一进口道、同一相位、上下游 movement 等关系应当对排队和到达流预测有帮助。

因此，V2 的优化方向主要围绕三类信息展开：

- movement 个体差异；
- 信号控制特征；
- movement graph 空间关系。

## 3. 实验路径与主要改动

### 3.1 初始 full_v3 基线

初始 full_v3 训练结果中，Transformer V2 已经能正常训练和加载，但整体指标没有超过 Transformer V1。

指标来源：`reports/full_v3_training/metrics.csv`

| 模型 | MAE | RMSE | WAPE | MAE@5 | MAE@10 | MAE@15 |
|---|---:|---:|---:|---:|---:|---:|
| HA baseline | 1.2614 | 2.5231 | 0.4684 | 1.1270 | 1.2051 | 1.2614 |
| XGBoost | 1.0745 | 2.0600 | 0.3990 | 1.0732 | 1.0737 | 1.0745 |
| LSTM | 0.9035 | 1.8704 | 0.3355 | 0.8946 | 0.9011 | 0.9035 |
| Transformer V1 | 0.8545 | 1.7882 | 0.3173 | 0.8411 | 0.8484 | 0.8545 |
| Transformer V2 | 0.8591 | 1.8218 | 0.3190 | 0.8391 | 0.8479 | 0.8591 |

初始结论：

- V2 的短步长 `MAE@5` 略低于 V1；
- 但整体 `MAE / RMSE / WAPE` 均没有超过 V1；
- 初始 V2 没有体现出稳定优势。

### 3.2 V2 中等增强

随后对 V2 做了一轮中等增强，主要包括：

1. 增加 `entity_output_bias`，让每个 movement 具备独立输出偏置能力；
2. temporal pooling 从 `last-only` 改为 `0.7 * last + 0.3 * mean`；
3. 接入 movement graph residual；
4. 给 V2 设置专属训练超参；
5. 使用更稳定的训练策略和梯度裁剪。

指标来源：`reports/full_v3_training_v2tuned/metrics.csv`

| 模型 | MAE | RMSE | WAPE | MAE@5 | MAE@10 | MAE@15 |
|---|---:|---:|---:|---:|---:|---:|
| Transformer V1 | 0.8545 | 1.7882 | 0.3173 | 0.8411 | 0.8484 | 0.8545 |
| Transformer V2 tuned | 0.8610 | 1.8092 | 0.3197 | 0.8328 | 0.8485 | 0.8610 |

阶段结论：

- tuned V2 的 `RMSE` 比初始 V2 有所下降；
- `MAE@5` 进一步改善，说明 V2 对短期预测有一定潜力；
- 但 overall MAE 从 `0.8591` 变为 `0.8610`，没有超过旧 V1。

### 3.3 phase_state_v1：相位状态标量化

这一轮尝试修正控制特征表达方式。原先 `phase_id` 作为连续数值输入，这在语义上不合理，因为相位编号本质是类别变量。

因此，`phase_state_v1` 做了如下处理：

- 不再把裸 `phase_id` 作为普通连续输入；
- 使用 `movement_config.json` 中的 `green_phase_ids` 判断当前 movement 是否放行；
- 构造 `is_green`；
- 构造 `is_red_or_yellow = 1 - is_green`；
- 保留 `phase_elapsed_s / green_remaining_s` 作为连续控制特征。

指标来源：`reports/full_v3_training_phase_state_v1/metrics.csv`

| 模型 | MAE | RMSE | WAPE | MAE@5 | MAE@10 | MAE@15 |
|---|---:|---:|---:|---:|---:|---:|
| Transformer V1 | 0.8727 | 1.8247 | 0.3241 | 0.8572 | 0.8659 | 0.8727 |
| Transformer V2 phase_state_v1 | 0.8714 | 1.8206 | 0.3236 | 0.8458 | 0.8609 | 0.8714 |

阶段结论：

- 在 `phase_state_v1` 口径内部，V2 略优于 V1；
- 但二者都明显弱于旧 full_v3 的 V1；
- 说明简单的 `is_green / is_red_or_yellow` 标量特征并没有带来整体收益；
- 控制特征表达方式仍然不够理想。

### 3.4 phase_embed_graph_v1：相位 Embedding + Graph Attention Bias

下一轮尝试把相位作为类别上下文建模，而不是简单标量化。

主要改动：

1. 增加 phase embedding；
2. 使用 `phase_id` 作为 embedding index；
3. 将 `phase_elapsed_s / green_remaining_s` 保留为连续输入；
4. 引入 movement graph attention bias；
5. 利用 `same_incoming_edge / upstream_downstream / same_phase` 等关系增强空间注意力。

指标来源：`reports/full_v3_training_phase_embed_graph_v1/metrics.csv`

| 模型 | MAE | RMSE | WAPE | MAE@5 | MAE@10 | MAE@15 |
|---|---:|---:|---:|---:|---:|---:|
| LSTM | 0.8806 | 1.8390 | 0.3270 | 0.8709 | 0.8752 | 0.8806 |
| Transformer V1 | 0.8434 | 1.7874 | 0.3132 | 0.8266 | 0.8367 | 0.8434 |
| Transformer V2 phase_embed_graph_v1 | 0.8946 | 1.8405 | 0.3322 | 0.8776 | 0.8875 | 0.8946 |

阶段结论：

- 相位 embedding 与 graph 信息让 V1 明显变强，V1 MAE 从旧口径 `0.8545` 降到 `0.8434`；
- 但同样的信息让 V2 变差，V2 MAE 上升到 `0.8946`；
- 这说明相位信息本身并非无效，问题更可能出在 V2 的融合方式；
- 可能是 V2 中 temporal encoder、spatial encoder、phase embedding、graph bias 的组合方式存在冲突。

### 3.5 V2 单组件消融

为了定位 V2 变差的原因，又单独跑了三组 V2 消融：

1. `phase_embed_only`：只启用 phase embedding；
2. `graph_bias_only`：只启用 graph attention bias；
3. `queue_weight_only`：只启用 `queue_veh` 加权损失。

指标来源：`reports/full_v3_training_v2_ablation/v2_ablation_summary.csv`

| V2 变体 | MAE | RMSE | WAPE | MAE@5 | MAE@10 | MAE@15 |
|---|---:|---:|---:|---:|---:|---:|
| phase_embed_only | 0.8828 | 1.8236 | 0.3278 | 0.8603 | 0.8733 | 0.8828 |
| graph_bias_only | 0.8791 | 1.8398 | 0.3264 | 0.8548 | 0.8692 | 0.8791 |
| queue_weight_only | 0.8726 | 1.8225 | 0.3240 | 0.8466 | 0.8617 | 0.8726 |

其中 `queue_weight_only` 是三者中最好的，因此又训练了一个 `v2_stable_selected`。

指标来源：`reports/full_v3_training_v2_ablation/v2_stable_selected_metrics.csv`

| V2 稳态候选 | MAE | RMSE | WAPE | MAE@5 | MAE@10 | MAE@15 |
|---|---:|---:|---:|---:|---:|---:|
| v2_stable_selected | 0.8726 | 1.8225 | 0.3240 | 0.8466 | 0.8617 | 0.8726 |

阶段结论：

- `queue_weight_only` 是三个单组件中最好的；
- 但其 overall MAE 为 `0.8726`，仍明显弱于旧 V1 的 `0.8545`；
- `phase_embed_only` 和 `graph_bias_only` 均未证明有效；
- V2 的问题不是某个单一组件可以简单修好的。

## 4. 子集表现观察

### 4.1 tuned V2 在扰动场景中有一定优势

在 `full_v3_training_v2tuned/control_feature_ablation.csv` 中：

| 模型 | subset | MAE | RMSE | WAPE |
|---|---|---:|---:|---:|
| Transformer V1 | overall | 0.8545 | 1.7882 | 0.3173 |
| Transformer V2 tuned | overall | 0.8610 | 1.8092 | 0.3197 |
| Transformer V1 | control_perturbation | 0.9412 | 1.8986 | 0.3622 |
| Transformer V2 tuned | control_perturbation | 0.8968 | 1.8514 | 0.3451 |

观察：

- tuned V2 虽然 overall 不如 V1；
- 但在 `control_perturbation` 子集上明显优于 V1；
- 说明 V2 可能确实更容易捕捉部分控制扰动场景；
- 但这种优势没有转化成整体稳定收益。

### 4.2 phase_embed_graph_v1 对 V1 有帮助，对 V2 有害

在 `full_v3_training_phase_embed_graph_v1/control_feature_ablation.csv` 中：

| 模型 | subset | MAE | RMSE | WAPE |
|---|---|---:|---:|---:|
| Transformer V1 | overall | 0.8434 | 1.7874 | 0.3132 |
| Transformer V2 | overall | 0.8946 | 1.8405 | 0.3322 |
| Transformer V1 | control_perturbation | 0.9021 | 1.8691 | 0.3471 |
| Transformer V2 | control_perturbation | 0.9570 | 1.9082 | 0.3683 |

观察：

- V1 在该口径下成为当前已知最好的整体模型；
- V2 被显著拉低；
- 这说明相位类别信息可能有效，但 V2 的使用方式不对。

## 5. 当前总体结论

### 5.1 V2 尚未证明优于 V1

目前所有中等增强方案中，V2 都没有稳定超过旧 V1。

最关键对比如下：

| 方案 | V1 MAE | V2 MAE | 是否证明 V2 更优 |
|---|---:|---:|---|
| 初始 full_v3 | 0.8545 | 0.8591 | 否 |
| V2 tuned | 0.8545 | 0.8610 | 否 |
| phase_state_v1 | 0.8727 | 0.8714 | 仅在该弱口径内部略优 |
| phase_embed_graph_v1 | 0.8434 | 0.8946 | 否，且 V2 明显变差 |
| V2 ablation 最佳 | 0.8545 | 0.8726 | 否 |

### 5.2 V2 短期预测和扰动子集有一定信号

V2 并非完全无效：

- 初始 V2 的 `MAE@5` 略优于 V1；
- tuned V2 在 `control_perturbation` 子集上优于 V1；
- 说明 V2 可能对短期动态和控制扰动有一定敏感性。

但这个优势不稳定，且无法覆盖 overall 指标。

### 5.3 当前不建议继续小修小补 V2

已经尝试过：

- entity-specific 输出偏置；
- temporal pooling 改造；
- graph residual；
- phase_state 标量特征；
- phase embedding；
- graph attention bias；
- queue weighted loss；
- 单组件消融。

这些中等规模改动都没有让 V2 稳定超过 V1。

因此，当前建议：

1. 正式展示和论文主线优先采用 V1 或当前稳定模型；
2. V2 作为探索性模型保留；
3. 后续如果继续做 V2，不应继续堆小组件，而应考虑结构性重构。

## 6. 可能失败原因猜测

### 6.1 V2 的 temporal -> spatial 融合方式可能不合适

当前 V2 大体流程是：

1. 对每个 movement 做时间编码；
2. 通过 temporal pooling 得到每个 movement 的状态；
3. 再做 movement 间空间编码；
4. 输出未来 15 步预测。

这可能导致：

- 空间交互只发生在时间压缩之后；
- movement 间的动态传播无法在每个时间步充分建模；
- 对排队传播、上下游拥堵扩散等现象表达不足。

### 6.2 graph bias 可能过早约束空间注意力

movement graph 是人工构造的关系，包括：

- same incoming edge；
- same phase；
- same tls；
- upstream downstream；
- conflict same tls different phase。

这些关系并不一定等价于真实交通影响强度。直接加入 graph bias 可能让模型过度依赖不够准确的先验，反而损害学习。

### 6.3 phase embedding 与 V2 token 表达可能冲突

phase embedding 在 V1 中有效，但在 V2 中无效，可能原因包括：

- phase embedding 加入 entity token 的方式过于简单；
- phase_id 是路口级控制状态，但 V2 将其注入到每个 movement token 中，可能造成冗余或噪声；
- signal_state 与 phase_id 的关系没有被清晰编码；
- 绿灯、黄灯、全红、冲突相位等控制语义没有被结构化表示。

### 6.4 数据规模可能不足以支撑更复杂 V2

当前 full_v3 虽然已经比早期数据丰富，但 movement 数量约 151，预测输出 453 维，V2 结构比 V1 更复杂。

如果训练样本数量、场景多样性或扰动强度仍不足，V2 可能更容易过拟合或欠拟合，而 V1 的扁平结构反而更稳。

### 6.5 多目标共享输出头可能限制了 V2

当前输出目标同时包括：

- arrival_flow；
- mean_speed；
- queue_veh。

这三个目标的量纲、动态规律和对信号控制的敏感性不同。共享一个输出头可能让 V2 难以同时优化三类目标，尤其是 queue 目标。

## 7. 当前建议

### 7.1 工程展示建议

当前系统前端和 API 中，建议默认使用稳定模型，不强行切换到 V2。

如果需要展示 V2，可以表述为：

> 本项目实现了 movement 级 Transformer V2 时空预测模型，并进行了相位特征、图关系和排队加权等多轮增强实验。实验表明，V2 在部分短期预测和控制扰动子集上具有一定潜力，但当前 overall 指标尚未稳定超过 Transformer V1，因此正式演示中采用更稳健的模型作为默认预测器。

### 7.2 论文表述建议

不要写成：

> Transformer V2 明显优于 V1。

建议写成：

> Transformer V2 作为时空结构扩展模型被纳入对比实验。结果显示，V2 在控制扰动子集和短预测步长上出现一定改善，但整体精度尚未稳定超过 V1。该结果说明，在当前样本规模和 movement graph 构造方式下，显式时空建模仍需进一步优化。

### 7.3 后续技术路线建议

如果继续优化 V2，建议不要再做小规模组件堆叠，而是考虑以下方向：

1. spatial-first 或 time-space alternating attention；
2. 每个时间步都进行空间交互，而不是时间池化后再空间交互；
3. 使用 STGCN、Graph WaveNet、DCRNN 等更成熟的时空交通预测结构；
4. 分目标输出头，分别预测 flow、speed、queue；
5. 对 phase/control 特征单独建模，而不是简单拼到 movement token；
6. 增加训练数据规模和扰动场景多样性。

## 8. 希望网页端 GPT 帮忙分析的问题

请重点分析以下问题：

1. 为什么 V2 在加入相位 embedding 与 graph bias 后反而变差？
2. 当前 V2 是否存在时间编码和空间编码融合顺序问题？
3. 是否应该改为 spatial-first，或者采用 time-space alternating attention？
4. graph attention bias 是否可能过度约束模型，使其难以学习真实相关性？
5. 是否应引入更强的 graph encoder，例如 STGCN、Graph WaveNet、DCRNN？
6. 是否需要对 `arrival_flow / mean_speed / queue_veh` 分目标建模，而不是共享一个输出头？
7. 当前 movement 数量约 151，输出维度 453，样本规模是否不足以支撑更复杂的 V2？
8. phase_id、signal_state、green_remaining_s 等信号控制特征应该如何更合理地输入时空模型？
9. 如果论文时间紧，是否应该将 V1 作为主模型，把 V2 作为探索性扩展？
10. 如果只允许再做一轮改进，最值得尝试的是结构重构、损失函数、数据增强，还是图关系重构？

## 9. 一句话总结

当前 Transformer V2 已完成多轮结构增强和消融实验，但尚未稳定超过 Transformer V1；V2 在短期预测和控制扰动子集上有一定潜力，问题更可能出在时空融合结构与控制特征注入方式，而不是相位信息本身无效。
