# RealData 接入模块

本模块提供一个轻量级真实交通数据接入口。外部视频、雷达、线圈、GPS 或信号机日志程序先把原始数据整理成标准 JSON，再通过 `POST /api/realdata/snapshot` 送入当前项目。

## 数据链路

```text
真实检测数据
-> POST /api/realdata/snapshot
-> RealDataAdapter
-> movement 级 snapshot
-> RealDataStore
-> /api/realdata/latest 和 WebSocket payload.real_data
```

当 `configs/real_data_config.json` 中 `use_for_prediction=true` 时，snapshot 会继续送入 `prediction_service.update_observation()`，复用现有 Transformer 预测服务。默认值为 `false`，避免干扰当前 SUMO 演示主线。

## 输入格式

```json
{
  "source": "camera_demo",
  "timestamp": "2026-05-10T10:00:00",
  "step": 1,
  "records": [
    {
      "movement_id": "12254692358__158074689_2__s__158074689_1",
      "arrival_flow": 12,
      "discharge_flow": 10,
      "mean_speed_mps": 7.5,
      "queue_veh": 4,
      "queue_meter": 32.0,
      "occupancy": 0.25,
      "incident_flag": 0,
      "phase_id": 2,
      "phase_elapsed_s": 18.0,
      "green_remaining_s": 12.0,
      "signal_state": "rrGGrr"
    }
  ]
}
```

如果外部系统只能提供 `detector_id`，可在 `configs/real_detector_map.example.json` 中配置 `detector_id -> movement_id` 映射。若同时提供 `movement_id` 和 `detector_id`，优先使用 `movement_id`。

## 缺失 Movement

Transformer 预测模型要求 movement 输入维度稳定，因此默认会补齐 `configs/movement_config.json` 中的全部 movement。缺失项优先沿用上一帧并标记为 `real_last_filled`；没有上一帧时填 0 并标记为 `real_missing_filled`。

## 接口

- `GET /api/realdata/config`：查看模块启用状态、预测接入开关和映射配置。
- `GET /api/realdata/latest`：查看最近一次真实数据 snapshot。
- `POST /api/realdata/snapshot`：提交新的真实交通数据 snapshot。

第一版不接入 RTSP、雷达私有协议、MQTT、Kafka 或数据库，也不直接控制真实信号机。
