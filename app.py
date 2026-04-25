# -*- coding: utf-8 -*-
import os
import sys
import json
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
    EdgeRealtimeCollector,
    PredictRequest,
    PredictionModelSwitchRequest,
    ScenarioCompareRequest,
    PredictionService,
    load_prediction_config,
)

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
    PROJECT_ROOT / "models" / "artifacts",
    PROJECT_ROOT / "reports" / "metrics.csv",
    PROJECT_ROOT / "data" / "raw" / "batch_edge_aggregates.csv",
    PROJECT_ROOT / "data" / "raw" / "scenarios" / "manifest.csv",
)
edge_collector = EdgeRealtimeCollector(
    prediction_config,
    PROJECT_ROOT / "data" / "raw" / "realtime_edge_aggregates.csv",
    base_demand_factor=prediction_config.base_demand_factor,
)
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

class IncidentRequest(BaseModel):
    road_name: str
    desc: str
    action: str = "create" # "create" or "resolve"

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
                    "isInternal": is_internal,
                    "dirs": dirs
                })
        return {"status": "ok", "lanes": lanes}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/incidents")
async def handle_manual_incident(req: IncidentRequest):
    global active_incidents
    try:
        # Find an edge with the specified Chinese name
        target_edge = None
        for edge in net_obj.getEdges():
            name = edge.getName()
            if name and req.road_name in name:
                target_edge = edge
                break
                
        if not target_edge:
            return {"status": "error", "message": f"未找到名为 '{req.road_name}' 的道路"}
            
        if req.action == "resolve":
            # Remove all manual incidents for this road
            to_remove = [inc_id for inc_id, v in active_incidents.items() if v["source"] == "manual" and v["road_name"] == req.road_name]
            for r in to_remove:
                active_incidents.pop(r, None)
            return {"status": "ok", "message": "已解除告警"}

        # Create new incident
        # get middle coordinate
        shape = target_edge.getShape()
        if not shape:
            return {"status": "error", "message": "道路无形状数据"}
            
        mx, my = shape[len(shape)//2]
        lon_wgs, lat_wgs = net_obj.convertXY2LonLat(mx, my)
        lon, lat = wgs84_to_gcj02(lon_wgs, lat_wgs)
        
        inc_id = str(uuid.uuid4())
        active_incidents[inc_id] = {
            "id": inc_id,
            "road_name": req.road_name,
            "desc": req.desc,
            "lnglat": [lon, lat],
            "source": "manual",
            "active": True
        }
        
        return {"status": "ok", "incident_id": inc_id}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/prediction/config")
async def get_prediction_config():
    return prediction_service.config_payload()

@app.get("/api/prediction/latest")
async def get_latest_prediction():
    return prediction_service.latest_payload()

@app.get("/api/prediction/scenario-runs")
async def get_prediction_scenario_runs():
    return prediction_service.scenario_runs_payload()

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

            incident_edges = {
                inc_id.replace("auto-", "", 1)
                for inc_id in active_incidents.keys()
                if inc_id.startswith("auto-")
            }
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
                    # 获取交叉口的中心点坐标
                    x, y = traci.junction.getPosition(tl_id)
                    lon_wgs, lat_wgs = net_obj.convertXY2LonLat(x, y)
                    lon, lat = wgs84_to_gcj02(lon_wgs, lat_wgs)
                    tl_data.append({"id": tl_id, "state": state, "x": lon, "y": lat})
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
            
        traci.close()
        print("Simulation Finished.")
    except Exception as e:
        print(f"Simulation Error: {e}")
        try:
            traci.close()
        except:
            pass

@app.on_event("startup")
async def startup_event():
    # Start SUMO in background thread
    asyncio.create_task(sumo_simulation_task())

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000)



