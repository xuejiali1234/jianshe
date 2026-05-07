# -*- coding: utf-8 -*-
import os
import sys
import json
import csv
import asyncio
from pathlib import Path
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

# 鑷姩灏嗗綋鍓嶅伐浣滅洰褰曞垏鎹负 app.py 鎵€鍦ㄧ殑鐩綍
os.chdir(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = Path(__file__).resolve().parent

from sim import (
    configure_sumo_python_path,
    prepare_runtime_route_file,
    resolve_runtime_net_file,
    validate_prediction_runtime,
)

configure_sumo_python_path()

import traci
from prediction import (
    MovementRealtimeCollector,
    PredictRequest,
    PredictionModelSwitchRequest,
    ScenarioCompareRequest,
    PredictionService,
    load_prediction_config,
)
from realdata import RealDataAdapter, RealDataStore, RealTrafficSnapshotRequest

app = FastAPI(title="SUMO 城市交通预测看板")

# Mount static directory for HTML/CSS/JS
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

prediction_config = load_prediction_config(PROJECT_ROOT / "configs" / "prediction_config.json")
runtime_net_file = resolve_runtime_net_file(PROJECT_ROOT, prediction_config.sumo_net_file)
# SUMO headless command (no gui to save CPU and GPU)
import sumolib
sumoBinary = sumolib.checkBinary('sumo')
runtime_route_file = prepare_runtime_route_file(
    PROJECT_ROOT / "czq_demand.rou.xml",
    PROJECT_ROOT / "data" / "raw" / "runtime_routes",
    scale_factor=prediction_config.base_demand_factor,
)
sumoCmd = [
    sumoBinary, 
    "-n", str(runtime_net_file),
    "-r", str(runtime_route_file), 
    "--begin", "0",
    "--end", "3600",
    "--start", 
    "--no-warnings", 
    "--ignore-route-errors", 
    "--time-to-teleport", "15",              # 更早强制瞬移卡死的车
    "--ignore-junction-blocker", "5",         # 忽略阻挡路口的车
    "--time-to-impatience", "10",             # 车停10秒后强行加塞（解决无灯路口的死等）
    "--default.action-step-length", "0.5",    # 更细的车道动作步长，减少强行刹车
    "--device.rerouting.probability", "0.8",  # 80%的车辆配备导航设备，会主动绕开拥堵路段
    "--device.rerouting.period", "30",        # 导航设备每30秒更新一次路况并重新规划路线
    "--device.rerouting.adaptation-interval", "10", # 路况更新频率
]

# Connected WebSocket clients
connected_clients = set()

# Load net globally
net_file = str(runtime_net_file)
net_obj = sumolib.net.readNet(net_file)
edges = net_obj.getEdges()
default_edge_id = edges[0].getID() if edges else ""
current_edge_id = 'ALL' # default_edge_id

prediction_service = PredictionService(
    prediction_config,
    PROJECT_ROOT / prediction_config.artifact_dir,
    PROJECT_ROOT / prediction_config.metrics_file,
    PROJECT_ROOT / prediction_config.batch_csv_file,
    PROJECT_ROOT / prediction_config.scenario_manifest_file,
)
edge_collector = MovementRealtimeCollector(
    prediction_config,
    PROJECT_ROOT / "data" / "raw" / "realtime_movement_aggregates.csv",
    base_demand_factor=prediction_config.base_demand_factor,
    project_root=PROJECT_ROOT,
    net_file=runtime_net_file,
)
real_data_adapter = RealDataAdapter(
    PROJECT_ROOT,
    PROJECT_ROOT / "configs" / "real_data_config.json",
)
real_data_store = RealDataStore(maxlen=120)
validate_prediction_runtime(
    PROJECT_ROOT,
    runtime_net_file,
    net_obj,
    prediction_config.observed_edges,
    sumoBinary,
)


import math
import random
import time
import uuid
from pydantic import BaseModel
from collections import deque
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Global incidents store
active_incidents = {}
auto_incident_tracker = {} # edge_id -> continuous_steps
manual_incident_vehicle_bindings = {}
manual_incident_lane_controls = {}
manual_incident_speed_controls = {}
MANUAL_VSL_FACTOR = 0.25

class IncidentRequest(BaseModel):
    road_name: str
    edge_id: str | None = None
    event_type: str | None = None
    desc: str
    action: str = "create" # "create" or "resolve"


def _net_has_edge(edge_id: str) -> bool:
    try:
        net_obj.getEdge(edge_id)
        return True
    except Exception:
        return False


def _collect_manual_incidents():
    return {
        inc_id: incident
        for inc_id, incident in active_incidents.items()
        if incident.get("source") == "manual" and incident.get("active", True)
    }


def _restore_manual_incident_vehicle(incident_id: str):
    binding = manual_incident_vehicle_bindings.pop(incident_id, None)
    if not binding:
        return
    vehicle_id = str(binding.get("vehicle_id", "")).strip()
    if not vehicle_id:
        return
    try:
        if vehicle_id not in traci.vehicle.getIDList():
            return
        if binding.get("spawned"):
            try:
                traci.vehicle.remove(vehicle_id)
            except Exception:
                pass
            return
        traci.vehicle.setSpeed(vehicle_id, -1)
        traci.vehicle.setSpeedMode(vehicle_id, int(binding.get("speed_mode", 31)))
        lane_change_mode = binding.get("lane_change_mode")
        if lane_change_mode is not None:
            traci.vehicle.setLaneChangeMode(vehicle_id, int(lane_change_mode))
        color = binding.get("color")
        if color is not None:
            traci.vehicle.setColor(vehicle_id, tuple(color))
    except Exception:
        pass


def _restore_manual_incident_lane_control(incident_id: str):
    binding = manual_incident_lane_controls.pop(incident_id, None)
    if not binding:
        return
    for lane_id, original_disallowed in binding.get("lanes", {}).items():
        try:
            if lane_id in traci.lane.getIDList():
                traci.lane.setDisallowed(lane_id, list(original_disallowed))
        except Exception:
            pass


def _restore_manual_incident_speed_control(incident_id: str):
    binding = manual_incident_speed_controls.pop(incident_id, None)
    if not binding:
        return
    for lane_id, original_speed in binding.get("lanes", {}).items():
        try:
            if lane_id in traci.lane.getIDList():
                traci.lane.setMaxSpeed(lane_id, float(original_speed))
        except Exception:
            pass


def _restore_all_manual_incident_controls():
    for incident_id in list(manual_incident_vehicle_bindings.keys()):
        _restore_manual_incident_vehicle(incident_id)
    for incident_id in list(manual_incident_lane_controls.keys()):
        _restore_manual_incident_lane_control(incident_id)
    for incident_id in list(manual_incident_speed_controls.keys()):
        _restore_manual_incident_speed_control(incident_id)


def _spawn_manual_incident_vehicle(incident_id: str, incident: dict):
    edge_id = str(incident.get("edge_id", "")).strip()
    if not edge_id or not _net_has_edge(edge_id):
        return None
    edge = net_obj.getEdge(edge_id)
    lanes = [lane for lane in edge.getLanes() if lane.getLength() > 12]
    if not lanes:
        lanes = list(edge.getLanes())
    if not lanes:
        return None
    lane = lanes[min(len(lanes) // 2, len(lanes) - 1)]
    lane_id = lane.getID()
    lane_index = lane.getIndex() if hasattr(lane, "getIndex") else max(0, len(lanes) // 2)
    lane_length = max(float(lane.getLength()), 1.0)
    pos = min(max(8.0, lane_length * 0.45), max(8.0, lane_length - 8.0))

    route_edges = [edge_id]
    try:
        outgoing = list(lane.getOutgoing() or [])
    except Exception:
        outgoing = []
    for connection in outgoing:
        try:
            to_lane = connection.getTo()
            to_edge = to_lane.getEdge() if hasattr(to_lane, "getEdge") else None
            to_edge_id = to_edge.getID() if to_edge is not None else ""
            if to_edge_id and not to_edge_id.startswith(":"):
                route_edges.append(to_edge_id)
                break
        except Exception:
            continue

    route_id = f"manualIncidentRoute.{incident_id[:8]}"
    vehicle_id = f"manualIncident.{incident_id[:8]}"
    try:
        if route_id not in traci.route.getIDList():
            traci.route.add(route_id, route_edges)
    except Exception:
        try:
            traci.route.add(route_id, [edge_id])
        except Exception:
            return None

    try:
        traci.vehicle.add(
            vehicle_id,
            route_id,
            "DEFAULT_VEHTYPE",
            "now",
            str(lane_index),
            str(pos),
            "0",
        )
    except Exception:
        return None

    color = (255, 80, 80, 255)
    try:
        traci.vehicle.moveTo(vehicle_id, lane_id, pos)
    except Exception:
        pass
    try:
        traci.vehicle.setLaneChangeMode(vehicle_id, 0)
    except Exception:
        pass
    try:
        traci.vehicle.setSpeedMode(vehicle_id, 0)
        traci.vehicle.setSpeed(vehicle_id, 0.0)
    except Exception:
        pass
    try:
        traci.vehicle.setColor(vehicle_id, color)
    except Exception:
        pass

    binding = {
        "vehicle_id": vehicle_id,
        "spawned": True,
        "lane_id": lane_id,
        "lane_index": lane_index,
        "pos": pos,
        "color": color,
        "speed_mode": 31,
        "lane_change_mode": 1621,
    }
    manual_incident_vehicle_bindings[incident_id] = binding
    return binding


def _apply_manual_incident_lane_closure(incident_id: str, incident: dict):
    edge_id = str(incident.get("edge_id", "")).strip()
    if not edge_id or not _net_has_edge(edge_id):
        return False
    edge = net_obj.getEdge(edge_id)
    binding = manual_incident_lane_controls.get(incident_id)
    if not binding:
        lanes = {}
        for lane in edge.getLanes():
            lane_id = lane.getID()
            try:
                original_disallowed = tuple(traci.lane.getDisallowed(lane_id))
            except Exception:
                original_disallowed = tuple()
            lanes[lane_id] = original_disallowed
        binding = {"lanes": lanes}
        manual_incident_lane_controls[incident_id] = binding

    for lane_id, original_disallowed in binding.get("lanes", {}).items():
        try:
            current = set(traci.lane.getDisallowed(lane_id))
            current.add("passenger")
            traci.lane.setDisallowed(lane_id, list(current))
        except Exception:
            pass

    pos_base = str(incident.get("pos_base") or f"📍 {incident.get('road_name', '')}")
    incident["desc"] = f"{pos_base}<br/>该路段已封停：已有车辆可驶离，后续 passenger 车辆禁止进入"
    return True


def _apply_manual_incident_speed_limit(incident_id: str, incident: dict):
    edge_id = str(incident.get("edge_id", "")).strip()
    if not edge_id or not _net_has_edge(edge_id):
        return False
    edge = net_obj.getEdge(edge_id)
    binding = manual_incident_speed_controls.get(incident_id)
    if not binding:
        lanes = {}
        for lane in edge.getLanes():
            lane_id = lane.getID()
            try:
                original_speed = float(traci.lane.getMaxSpeed(lane_id))
            except Exception:
                original_speed = float(lane.getSpeed() if hasattr(lane, "getSpeed") else 13.89)
            lanes[lane_id] = original_speed
        binding = {"lanes": lanes}
        manual_incident_speed_controls[incident_id] = binding

    for lane_id, original_speed in binding.get("lanes", {}).items():
        try:
            traci.lane.setMaxSpeed(lane_id, max(0.1, float(original_speed) * MANUAL_VSL_FACTOR))
        except Exception:
            pass

    pos_base = str(incident.get("pos_base") or f"📍 {incident.get('road_name', '')}")
    incident["desc"] = f"{pos_base}<br/>该路段已限速：最高速度降至基准速度的 {int(MANUAL_VSL_FACTOR * 100)}%"
    incident["incident_speed_factor"] = MANUAL_VSL_FACTOR
    return True


def _apply_manual_incident_controls():
    active_manual_incidents = _collect_manual_incidents()
    active_ids = set(active_manual_incidents.keys())
    for incident_id in list(manual_incident_vehicle_bindings.keys()):
        if incident_id not in active_ids:
            _restore_manual_incident_vehicle(incident_id)
    for incident_id in list(manual_incident_lane_controls.keys()):
        if incident_id not in active_ids:
            _restore_manual_incident_lane_control(incident_id)
    for incident_id in list(manual_incident_speed_controls.keys()):
        if incident_id not in active_ids:
            _restore_manual_incident_speed_control(incident_id)

    affected_edges = set()
    for incident_id, incident in active_manual_incidents.items():
        edge_id = str(incident.get("edge_id", "")).strip()
        if not edge_id or not _net_has_edge(edge_id):
            continue
        affected_edges.add(edge_id)
        requested_event_type = str(incident.get("requested_event_type", "")).strip() or "stopped_vehicle"
        if requested_event_type == "closure":
            _restore_manual_incident_vehicle(incident_id)
            _restore_manual_incident_speed_control(incident_id)
            _apply_manual_incident_lane_closure(incident_id, incident)
            continue
        if requested_event_type == "vsl":
            _restore_manual_incident_vehicle(incident_id)
            _restore_manual_incident_lane_control(incident_id)
            _apply_manual_incident_speed_limit(incident_id, incident)
            continue

        _restore_manual_incident_lane_control(incident_id)
        _restore_manual_incident_speed_control(incident_id)
        pos_base = str(incident.get("pos_base") or f"📍 {incident.get('road_name', '')}")
        binding = manual_incident_vehicle_bindings.get(incident_id)
        vehicle_id = str(binding.get("vehicle_id", "")).strip() if binding else ""
        if not vehicle_id or vehicle_id not in traci.vehicle.getIDList():
            binding = _spawn_manual_incident_vehicle(incident_id, incident)
            vehicle_id = str(binding.get("vehicle_id", "")).strip() if binding else ""
        if not vehicle_id:
            incident["desc"] = f"{pos_base}<br/>无法在该路段生成事故车"
            continue
        try:
            if binding:
                lane_id = str(binding.get("lane_id", "")).strip()
                lane_pos = float(binding.get("pos", 0.0))
                if lane_id:
                    try:
                        traci.vehicle.moveTo(vehicle_id, lane_id, lane_pos)
                    except Exception:
                        pass
                try:
                    traci.vehicle.setLaneChangeMode(vehicle_id, 0)
                except Exception:
                    pass
            traci.vehicle.setSpeedMode(vehicle_id, 0)
            traci.vehicle.setSpeed(vehicle_id, 0.0)
            traci.vehicle.setColor(vehicle_id, (255, 80, 80, 255))
        except Exception:
            pass
        try:
            x, y = traci.vehicle.getPosition(vehicle_id)
            lon_wgs, lat_wgs = net_obj.convertXY2LonLat(x, y)
            glon, glat = wgs84_to_gcj02(lon_wgs, lat_wgs)
            incident["lnglat"] = [glon, glat]
        except Exception:
            pass
        incident["desc"] = (
            f"{pos_base}<br/>"
            f"事故车 {vehicle_id} 已占道静止，后续车辆仍可继续进入"
        )
    return affected_edges

def run_holtwinters_forecast(recent_speeds):
    if len(recent_speeds) < 18:
        return recent_speeds[-1] if recent_speeds else 0.0, [recent_speeds[-1] if recent_speeds else 0.0] * 10
    try:
        model = ExponentialSmoothing(
            list(recent_speeds), 
            trend='add', damped_trend=True, 
            initialization_method='estimated'
        )
        # 极度降低平滑系数，使其像一块“生铁”一样稳定：
        # smoothing_level=0.1（几乎只看历史大盘，完全无视近几十秒的单点跳动）
        # smoothing_trend=0.01（趋势几乎锁死，只有出现长达好几分钟的持续下降/上升，才会慢慢转弯）
        fit_model = model.fit(smoothing_level=0.1, smoothing_trend=0.01, optimized=False)
        current_fit = float(fit_model.fittedvalues[-1])
        forecast = fit_model.forecast(10)
        
        # 添加物理限制，避免负数或过分离谱的数据
        clamped_forecast = [min(25.0, max(0.0, float(x))) for x in forecast]
        return max(0.0, current_fit), clamped_forecast
    except Exception as e:
        return recent_speeds[-1] if recent_speeds else 0.0, [recent_speeds[-1] if recent_speeds else 0.0] * 10

def wgs84_to_gcj02(lng, lat):
    """
    WGS84 to GCJ02 Coordinate Transformation
    """
    if out_of_china(lng, lat):
        return lng, lat
    
    a = 6378245.0
    ee = 0.00669342162296594323
    
    dlat = transform_lat(lng - 105.0, lat - 35.0)
    dlng = transform_lng(lng - 105.0, lat - 35.0)
    
    radlat = lat / 180.0 * math.pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * math.pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * math.pi)
    
    mglat = lat + dlat
    mglng = lng + dlng
    return mglng, mglat

def transform_lat(lng, lat):
    ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + 0.1 * lng * lat + 0.2 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * math.pi) + 20.0 * math.sin(2.0 * lng * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * math.pi) + 40.0 * math.sin(lat / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * math.pi) + 320 * math.sin(lat * math.pi / 30.0)) * 2.0 / 3.0
    return ret

def transform_lng(lng, lat):
    ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + 0.1 * lng * lat + 0.1 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * math.pi) + 20.0 * math.sin(2.0 * lng * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lng * math.pi) + 40.0 * math.sin(lng / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lng / 12.0 * math.pi) + 300.0 * math.sin(lng / 30.0 * math.pi)) * 2.0 / 3.0
    return ret

def out_of_china(lng, lat):
    return not (72.004 <= lng <= 137.8347 and 0.8293 <= lat <= 55.8271)


def xy_to_gcj_lnglat(x, y):
    lon_wgs, lat_wgs = net_obj.convertXY2LonLat(x, y)
    return list(wgs84_to_gcj02(lon_wgs, lat_wgs))


def build_stopbar_segment(lane_shape, half_width_m=3.2, backoff_m=1.2):
    if not lane_shape or len(lane_shape) < 2:
        return None
    end_x, end_y = lane_shape[-1]
    prev_x, prev_y = lane_shape[-2]
    dx = end_x - prev_x
    dy = end_y - prev_y
    length = math.hypot(dx, dy)
    if length <= 0:
        return None
    ux = dx / length
    uy = dy / length
    center_x = end_x - ux * backoff_m
    center_y = end_y - uy * backoff_m
    perp_x = -uy
    perp_y = ux
    left = (center_x - perp_x * half_width_m, center_y - perp_y * half_width_m)
    right = (center_x + perp_x * half_width_m, center_y + perp_y * half_width_m)
    return [xy_to_gcj_lnglat(*left), xy_to_gcj_lnglat(*right)]


def build_signal_stopbars():
    stopbars = []
    seen = set()
    for edge in net_obj.getEdges():
        if edge.getFunction() == "internal":
            continue
        for lane in edge.getLanes():
            segment = build_stopbar_segment(lane.getShape())
            if not segment:
                continue
            for connection in lane.getOutgoing() or []:
                tls_id = connection.getTLSID() if hasattr(connection, "getTLSID") else ""
                link_index = connection.getTLLinkIndex() if hasattr(connection, "getTLLinkIndex") else -1
                if not tls_id or link_index is None or link_index < 0:
                    continue
                key = (tls_id, link_index, lane.getID())
                if key in seen:
                    continue
                seen.add(key)
                stopbars.append({
                    "tlsId": tls_id,
                    "linkIndex": int(link_index),
                    "edgeId": edge.getID(),
                    "laneId": lane.getID(),
                    "toEdgeId": connection.getTo().getID() if hasattr(connection, "getTo") else "",
                    "dir": connection.getDirection() if hasattr(connection, "getDirection") else "",
                    "shape": segment,
                })
    return stopbars


@app.get("/")
async def get_index():
    return FileResponse("static/index.html")

@app.get("/network")
async def get_network():
    try:
        lanes = []
        for edge in net_obj.getEdges():
            is_internal = edge.getFunction() == "internal"
            for lane in edge.getLanes():
                shape = lane.getShape()
                lonlat_shape = [list(wgs84_to_gcj02(*net_obj.convertXY2LonLat(x, y))) for x, y in shape]
                
                dirs = []
                if not is_internal:
                    outgoing = lane.getOutgoing()
                    if outgoing:
                        dirs = list(set([c.getDirection() for c in outgoing]))
                
                lanes.append({                    "edgeId": edge.getID(),                    "shape": lonlat_shape,
                    "roadName": edge.getName() or "",
                    "isInternal": is_internal,
                    "dirs": dirs
                })
        return {"status": "ok", "lanes": lanes, "signalStopbars": build_signal_stopbars()}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/incidents")
async def handle_manual_incident(req: IncidentRequest):
    global active_incidents
    try:
        target_edge = None
        requested_edge_id = str(req.edge_id or "").strip()
        if requested_edge_id and _net_has_edge(requested_edge_id):
            target_edge = net_obj.getEdge(requested_edge_id)
        else:
            # Find an edge with the specified Chinese name
            for edge in net_obj.getEdges():
                name = edge.getName()
                if name and req.road_name in name:
                    target_edge = edge
                    break
                
        if not target_edge:
            return {"status": "error", "message": f"未找到名为 '{req.road_name}' 的道路"}
            
        if req.action == "resolve":
            # Remove all manual incidents for this road
            target_edge_id = target_edge.getID()
            to_remove = [
                inc_id for inc_id, v in active_incidents.items()
                if v["source"] == "manual" and str(v.get("edge_id", "")) == target_edge_id
            ]
            for r in to_remove:
                _restore_manual_incident_vehicle(r)
                _restore_manual_incident_lane_control(r)
                _restore_manual_incident_speed_control(r)
            for r in to_remove:
                active_incidents.pop(r, None)
            return {"status": "ok", "message": "已解除告警"}

        target_edge_id = target_edge.getID()
        existing_ids = [
            inc_id for inc_id, incident in active_incidents.items()
            if incident.get("source") == "manual"
            and str(incident.get("edge_id", "")) == target_edge_id
            and incident.get("active", True)
        ]
        for existing_id in existing_ids:
            _restore_manual_incident_vehicle(existing_id)
            _restore_manual_incident_lane_control(existing_id)
            _restore_manual_incident_speed_control(existing_id)
            active_incidents.pop(existing_id, None)

        # Create new incident
        # get middle coordinate
        shape = target_edge.getShape()
        if not shape:
            return {"status": "error", "message": "道路无形状数据"}
            
        mx, my = shape[len(shape)//2]
        lon_wgs, lat_wgs = net_obj.convertXY2LonLat(mx, my)
        lon, lat = wgs84_to_gcj02(lon_wgs, lat_wgs)
        
        requested_event_type = str(req.event_type or "").strip() or "stopped_vehicle"
        if requested_event_type == "closure":
            incident_event_type = "incident_closure"
            incident_policy = "manual_edge_passenger_closure"
            default_desc = "正在实施事故封停"
        elif requested_event_type == "vsl":
            incident_event_type = "vsl_speed_drop"
            incident_policy = "manual_edge_speed_limit"
            default_desc = "正在实施道路限速"
        else:
            requested_event_type = "stopped_vehicle"
            incident_event_type = "stopped_vehicle_incident"
            incident_policy = "manual_spawned_vehicle_blocking"
            default_desc = "正在生成事故车"

        inc_id = str(uuid.uuid4())
        active_incidents[inc_id] = {
            "id": inc_id,
            "road_name": target_edge.getName() or req.road_name or target_edge_id,
            "desc": req.desc or default_desc,
            "lnglat": [lon, lat],
            "edge_id": target_edge_id,
            "requested_event_type": requested_event_type,
            "source": "manual",
            "active": True,
            "event_type": incident_event_type,
            "event_policy": incident_policy,
            "pos_base": f"📍 {target_edge.getName() or req.road_name or target_edge_id}",
        }
        if requested_event_type == "vsl":
            active_incidents[inc_id]["incident_speed_factor"] = MANUAL_VSL_FACTOR
        
        return {"status": "ok", "incident_id": inc_id}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/prediction/config")
async def get_prediction_config():
    return prediction_service.config_payload()

@app.get("/api/realdata/config")
async def get_realdata_config():
    return real_data_adapter.config_payload()

@app.get("/api/realdata/latest")
async def get_realdata_latest():
    return real_data_store.latest_payload()

@app.post("/api/realdata/snapshot")
async def post_realdata_snapshot(req: RealTrafficSnapshotRequest):
    if not real_data_adapter.config.get("enabled", True):
        raise HTTPException(status_code=503, detail="RealData ingestion is disabled")
    try:
        snapshot = real_data_adapter.build_snapshot(req)
        real_data_store.update(snapshot)
        use_for_prediction = bool(real_data_adapter.config.get("use_for_prediction", False))
        if use_for_prediction:
            prediction_service.update_observation(snapshot)
        return {
            "status": "ok",
            "message": "real traffic snapshot accepted",
            "movement_count": len(snapshot.get("movements", [])),
            "node_count": len(snapshot.get("nodes", [])),
            "records_received": snapshot.get("records_received", 0),
            "records_ignored": snapshot.get("records_ignored", 0),
            "use_for_prediction": use_for_prediction,
            "prediction_model": prediction_service.active_model,
            "fallback_used": bool(prediction_service.latest_prediction.get("fallback_used", False)),
            "timestamp": snapshot.get("timestamp"),
            "step": snapshot.get("step"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/prediction/latest")
async def get_latest_prediction():
    return prediction_service.latest_payload()

@app.get("/api/prediction/phase-aggregate")
async def get_prediction_phase_aggregate():
    return prediction_service.phase_aggregate_payload()

@app.get("/api/prediction/scenario-runs")
async def get_prediction_scenario_runs():
    return prediction_service.scenario_runs_payload()

@app.get("/api/rl/control-summary")
async def get_rl_control_summary():
    candidates = [
        PROJECT_ROOT / "reports" / "rl_signal_control" / "full_v3_pred_control_v1" / "policy_comparison_1800.csv",
        PROJECT_ROOT / "reports" / "rl_signal_control" / "full_v3_pred_control" / "policy_comparison_1800.csv",
        PROJECT_ROOT / "reports" / "rl_signal_control" / "policy_comparison_1800.csv",
    ]
    selected_path = next((path for path in candidates if path.exists()), None)
    if selected_path is None:
        return {"status": "ok", "rows": [], "source_file": None}

    rows = []
    with selected_path.open(newline="", encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            rows.append(row)
    return {
        "status": "ok",
        "rows": rows,
        "source_file": str(selected_path.relative_to(PROJECT_ROOT)),
        "preferred_policy": "DQN-pred-v1",
        "stable_prediction_model": prediction_service.active_model,
    }

@app.post("/api/prediction/active-model")
async def set_active_prediction_model(req: PredictionModelSwitchRequest):
    try:
        return prediction_service.switch_model(req.model_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/predict")
async def predict_traffic(req: PredictRequest):
    try:
        return prediction_service.predict_request(req)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/prediction/scenario-compare")
async def compare_prediction_scenarios(req: ScenarioCompareRequest):
    try:
        return prediction_service.scenario_compare_payload(req)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global current_edge_id
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True:
            # Keep connection alive & handle dynamic edge changes
            data = await websocket.receive_json()
            if data.get("action") == "set_edge" and data.get("edgeId"):
                current_edge_id = data["edgeId"]
                print(f"Edge tracking changed to: {current_edge_id}")
    except WebSocketDisconnect:
        connected_clients.remove(websocket)
    except Exception as e:
        if websocket in connected_clients:
            connected_clients.remove(websocket)

async def sumo_simulation_task():
    global current_edge_id
    print("Starting SUMO Simulation in background...")
    try:
        traci.start(sumoCmd)
        step = 0
        vclass_map = {}
        history_window = deque(maxlen=60) # keep history for up to 600 steps (60*10)
        last_monitored_edge = current_edge_id
        future_speeds = []
        current_fit = 0.0

        while traci.simulation.getMinExpectedNumber() > 0 and step < 3600:
            traci.simulationStep()
            sim_time_s = float(traci.simulation.getTime())

            # Extract data every step or every X steps (here we extract every step for smooth radar)
            if current_edge_id == 'ALL':
                all_vehicles = traci.vehicle.getIDList()
                veh_num = len(all_vehicles)
                if not all_vehicles:
                    mean_speed = 0.0
                    halt_num = 0
                    co2 = 0.0
                else:
                    sum_speed = 0.0
                    halt_num = 0
                    co2 = 0.0
                    for v in all_vehicles:
                        speed = traci.vehicle.getSpeed(v)
                        sum_speed += speed
                        if speed < 0.1:
                            halt_num += 1
                        co2 += traci.vehicle.getCO2Emission(v)
                    mean_speed = sum_speed / len(all_vehicles)
            else:
                veh_num = traci.edge.getLastStepVehicleNumber(current_edge_id)
                mean_speed = traci.edge.getLastStepMeanSpeed(current_edge_id)
                halt_num = traci.edge.getLastStepHaltingNumber(current_edge_id)
                co2 = traci.edge.getCO2Emission(current_edge_id)

            if current_edge_id != last_monitored_edge:
                history_window.clear()
                last_monitored_edge = current_edge_id
                future_speeds = []
                current_fit = mean_speed
            
            # Sample every 10 steps (approx 10s simulation time)
            if step % 10 == 0:
                history_window.append(mean_speed)
                
            # --- Auto-detect incidents (queue/congestion spillover) ---
            global active_incidents, auto_incident_tracker
            current_congested_edges = set()
            if step % 10 == 0:  # check roughly every 10 steps to save cpu
                # Get heavily congested edges
                edge_halt_counts = {}
                for edge in net_obj.getEdges():
                    if edge.getFunction() == "internal": continue
                    edge_id = edge.getID()
                    halt = traci.edge.getLastStepHaltingNumber(edge_id)
                    # For a simple demo: consider > 10 halting vehicles as severe jam
                    if halt > 10:
                        current_congested_edges.add(edge_id)
                        edge_halt_counts[edge_id] = halt
                        
                # Update tracker
                for edge_id in current_congested_edges:
                    auto_incident_tracker[edge_id] = auto_incident_tracker.get(edge_id, 0) + 10
                    # if congestion persists for > 50 simulation steps (approx 5 sec logic but sampled every 10)
                    # wait, let's say 5 checks = 50 steps = 25 seconds (since action-step=0.5)
                    if auto_incident_tracker[edge_id] > 30:
                        inc_id = f"auto-{edge_id}"
                        if inc_id not in active_incidents:
                            edge = net_obj.getEdge(edge_id)
                            raw_name = edge.getName()
                            
                            inc_desc_pos = ""
                            dir_name = "下行方向" if edge_id.startswith("-") else "上行方向"
                            if raw_name:
                                r_name = f"{raw_name} ({dir_name})"
                                inc_desc_pos = f"📍 {raw_name}"
                            else:
                                r_name = f"路段 {edge_id.replace('-', '')} ({dir_name})"
                                inc_desc_pos = f"📍 交叉口连线 {edge_id}"

                            # Extract coords
                            shape = edge.getShape()
                            mx, my = shape[len(shape)//2]
                            lon_wgs, lat_wgs = net_obj.convertXY2LonLat(mx, my)
                            glon, glat = wgs84_to_gcj02(lon_wgs, lat_wgs)
                            
                            active_incidents[inc_id] = {
                                "id": inc_id,
                                "road_name": r_name,
                                "desc": f"{inc_desc_pos}<br/>严重拥堵，排队数量: {edge_halt_counts[edge_id]}辆",
                                "lnglat": [glon, glat],
                                "source": "auto",
                                "active": True,
                                "pos_base": inc_desc_pos
                            }
                        else:
                            pos_base = active_incidents[inc_id].get("pos_base", "")
                            active_incidents[inc_id]["desc"] = f"{pos_base}<br/>严重拥堵，排队数量: {edge_halt_counts[edge_id]}辆"
                
                # Resolve cleared incidents
                to_remove = []
                for edge_id in list(auto_incident_tracker.keys()):
                    if edge_id not in current_congested_edges:
                        auto_incident_tracker[edge_id] -= 20
                        if auto_incident_tracker[edge_id] <= 0:
                            auto_incident_tracker.pop(edge_id, None)
                            inc_id = f"auto-{edge_id}"
                            if inc_id in active_incidents:
                                to_remove.append(inc_id)
                                
                for inc_id in to_remove:
                    active_incidents.pop(inc_id, None)
            # ---------------------------------------------------------

            manual_incident_edges = _apply_manual_incident_controls()

            incident_edges = {
                inc_id.replace("auto-", "", 1)
                for inc_id in active_incidents.keys()
                if inc_id.startswith("auto-")
            }
            incident_edges |= manual_incident_edges
            edge_snapshot = edge_collector.record_step(
                traci,
                step,
                sim_time_s,
                incident_edges,
            )
            if edge_snapshot:
                prediction_service.update_observation(edge_snapshot)
                
            # Perform prediction asynchronously off-thread every 10 steps
            if step % 10 == 0 and len(history_window) >= 10:
                current_fit, future_speeds = await asyncio.to_thread(run_holtwinters_forecast, list(history_window))
            elif not future_speeds:
                future_speeds = [mean_speed] * 10
                current_fit = mean_speed

            radar_data = []
            modal_counts = {}

            # Extract vehicles
            for v_id in traci.vehicle.getIDList():
                x, y = traci.vehicle.getPosition(v_id)
                lon_wgs, lat_wgs = net_obj.convertXY2LonLat(x, y)
                lon, lat = wgs84_to_gcj02(lon_wgs, lat_wgs)
                angle = traci.vehicle.getAngle(v_id)
                
                # Dynamic visual class assignment for demo variety
                if v_id not in vclass_map:
                    base_class = traci.vehicle.getVehicleClass(v_id)
                    if base_class in ["passenger", "ignoring", "unknown"]:
                        vclass_map[v_id] = random.choices(
                            ["passenger", "truck", "bus", "motorcycle", "emergency"],
                            weights=[0.55, 0.15, 0.15, 0.10, 0.05]
                        )[0]
                    else:
                        vclass_map[v_id] = base_class
                
                v_class = vclass_map[v_id]
                radar_data.append({"id": v_id, "x": lon, "y": lat, "angle": angle, "vClass": v_class})

                modal_counts[v_class] = modal_counts.get(v_class, 0) + 1
            # Extract pedestrians (persons)
            for p_id in traci.person.getIDList():
                x, y = traci.person.getPosition(p_id)
                lon_wgs, lat_wgs = net_obj.convertXY2LonLat(x, y)
                lon, lat = wgs84_to_gcj02(lon_wgs, lat_wgs)
                angle = traci.person.getAngle(p_id)
                radar_data.append({"id": p_id, "x": lon, "y": lat, "angle": angle, "vClass": "pedestrian"})
                modal_counts["pedestrian"] = modal_counts.get("pedestrian", 0) + 1

            # Extract traffic lights
            tl_data = []
            for tl_id in traci.trafficlight.getIDList():
                state = traci.trafficlight.getRedYellowGreenState(tl_id)
                try:
                    phase = traci.trafficlight.getPhase(tl_id)
                    next_switch = float(traci.trafficlight.getNextSwitch(tl_id))
                    time_to_switch = max(0.0, next_switch - sim_time_s)
                    # 获取交叉口的中心点坐标
                    x, y = traci.junction.getPosition(tl_id)
                    lon_wgs, lat_wgs = net_obj.convertXY2LonLat(x, y)
                    lon, lat = wgs84_to_gcj02(lon_wgs, lat_wgs)
                    tl_data.append({
                        "id": tl_id,
                        "state": state,
                        "phase": phase,
                        "time_to_switch": time_to_switch,
                        "x": lon,
                        "y": lat,
                    })
                except Exception:
                    pass

            payload = {
                "step": step,
                "stats": {
                    "flow": veh_num,
                    "speed": mean_speed,
                    "queue": halt_num,
                    "co2": co2,
                    "current_pred": current_fit,
                    "future_speeds": future_speeds
                },
                "modal": modal_counts,
                "radar": radar_data,
                "tls": tl_data,
                "incidents": list(active_incidents.values()),
                "prediction": prediction_service.latest_prediction,
                "real_data": real_data_store.latest_payload(),
                "data_source": "sumo_with_realdata_overlay",
            }
            
            if connected_clients:
                message = json.dumps(payload)
                dead_clients = set()
                # Use list(connected_clients) to avoid checking a changing set
                for client in list(connected_clients):
                    try:
                        await client.send_text(message)
                    except Exception:
                        dead_clients.add(client)
                for c in dead_clients:
                    if c in connected_clients:
                        connected_clients.remove(c)
            
            step += 1
            # Control the simulation speed (e.g., 0.1s per step is slower for UI readability)
            await asyncio.sleep(0.15)
            
        _restore_all_manual_incident_controls()
        traci.close()
        print("Simulation Finished.")
    except Exception as e:
        print(f"Simulation Error: {e}")
        try:
            _restore_all_manual_incident_controls()
            traci.close()
        except:
            pass

@app.on_event("startup")
async def startup_event():
    # Start SUMO in background thread
    asyncio.create_task(sumo_simulation_task())

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000)



