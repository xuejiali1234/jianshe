const vClassColors = {
    passenger: '#409eff',
    truck: '#e6a23c',
    bus: '#67c23a',
    motorcycle: '#e024df',
    emergency: '#f56c6c',
    pedestrian: '#ffeb3b',
    DEFAULT: '#ffffff'
};
const VEHICLE_ICON_URLS = [
    '/static/assets/vehicle_green_marker.png',
    '/static/assets/vehicle_yellow_marker.png',
    '/static/assets/vehicle_red_marker.png'
];

const I18N = {
    titleDefault: '城市交通数字孪生看板',
    titleOnline: '系统在线',
    titleReconnect: '连接中断，正在重连',
    scopeFull: '全路网',
    scopeLabel: '监测范围：',
    edgeLabel: '监测路段：',
    resetView: '恢复全局视图',
    seriesObserved: '实时速度',
    seriesForecast: '短时预测',
    chartSpeedTitle: '速度变化趋势（km/h）',
    chartFlow: '流量',
    chartSpeed: '速度(km/h)',
    chartQueue: '排队',
    alertTitle: '预警：',
    incidentActiveSuffix: ' 条激活',
    alertTag: '预警',
    clearTag: '恢复',
    recovered: '已恢复通行',
    unitVehicle: '辆',
    minuteUnit: '分钟',
    warningMapFallback: '高德地图资源加载失败，已切换为无地图模式',
    warningChartFallback: '外部图表资源不可用，已切换为内置图表模式'
};

const modelNameMap = {
    ha_baseline: '历史平均基线',
    xgboost: 'XGBoost',
    lstm: 'LSTM',
    transformer_v1: 'Transformer V1（稳定预测模型）',
    transformer_v2: 'Transformer V2（探索模型）'
};

function getVehicleZoomScale() {
    if (!map || typeof map.getZoom !== 'function') return 1;
    const zoom = Number(map.getZoom());
    if (!Number.isFinite(zoom)) return 1;
    return Math.max(0.32, Math.min(1.25, Math.pow(1.28, zoom - 16)));
}

function getMapRotationAngle() {
    if (!map || typeof map.getRotation !== 'function') return 0;
    const rotation = Number(map.getRotation());
    return Number.isFinite(rotation) ? rotation : 0;
}

function getVehicleIconSize(vClass) {
    const scale = getVehicleZoomScale();
    if (vClass === 'pedestrian') return { width: 7, height: 7 };
    const base = { width: 7, height: 13 };
    return {
        width: Math.max(3, Math.round(base.width * scale)),
        height: Math.max(6, Math.round(base.height * scale))
    };
}

function normalizeVehicleAngle(angle) {
    const value = Number(angle);
    const vehicleAngle = Number.isFinite(value) ? value : 0;
    return vehicleAngle + getMapRotationAngle();
}

function getStableVehicleIconUrl(vehicle) {
    const key = String(vehicle && vehicle.id ? vehicle.id : '');
    let hash = 0;
    for (let i = 0; i < key.length; i += 1) {
        hash = ((hash << 5) - hash + key.charCodeAt(i)) | 0;
    }
    return VEHICLE_ICON_URLS[Math.abs(hash) % VEHICLE_ICON_URLS.length];
}

function renderVehicleMarkerContent(vehicle) {
    const color = vClassColors[vehicle.vClass] || vClassColors.DEFAULT;
    const size = getVehicleIconSize(vehicle.vClass);
    const angle = normalizeVehicleAngle(vehicle.angle);
    if (vehicle.vClass === 'pedestrian') {
        return `
            <div class="vehicle-icon vehicle-pedestrian" style="width:${size.width}px;height:${size.height}px;--vehicle-color:${color};">
                <svg viewBox="0 0 8 8" width="8" height="8" aria-hidden="true">
                    <circle cx="4" cy="4" r="3" fill="${color}" stroke="#06111f" stroke-width="1" />
                </svg>
            </div>
        `;
    }
    const iconUrl = getStableVehicleIconUrl(vehicle);
    return `
        <div class="vehicle-icon vehicle-car-icon" style="width:${size.width}px;height:${size.height}px;transform:translate(-50%, -50%) rotate(${angle}deg);transform-origin:50% 50%;pointer-events:none;filter:drop-shadow(0 1px 2px rgba(0,0,0,0.55));--vehicle-color:${color};">
            <img src="${iconUrl}" alt="" draggable="false" style="display:block;width:${size.width}px;height:${size.height}px;object-fit:contain;user-select:none;pointer-events:none;" />
        </div>
    `;
}

function formatModelName(modelName) {
    return modelNameMap[modelName] || modelName || '--';
}

function formatPredictionValue(value, digits = 2) {
    const number = Number(value);
    return Number.isFinite(number) ? number.toFixed(digits) : '--';
}

function formatSignedValue(value, digits = 2, suffix = '') {
    const number = Number(value);
    if (!Number.isFinite(number)) return '--';
    const prefix = number > 0 ? '+' : '';
    return `${prefix}${number.toFixed(digits)}${suffix}`;
}

function formatMeters(value) {
    const number = Number(value);
    if (!Number.isFinite(number) || number <= 0) return '--';
    return `${number.toFixed(number >= 100 ? 0 : 1)}m`;
}

function getTurnLabel(turnType) {
    return TURN_LABELS[String(turnType || '').toLowerCase()] || String(turnType || '--');
}

function getZoneQualityLabel(zoneQuality) {
    return zoneQuality === 'short_upstream' ? '上游不足' : '正常';
}

function formatEventType(eventType, incidentType = '') {
    if (eventType === 'vsl_speed_drop') return `S4 可变限速：${incidentType || 'speed_drop'}`;
    if (eventType === 'incident_closure') return `S5 事故封停：${incidentType || 'closure'}`;
    return incidentType || '--';
}

function getMovementCatalog(edgeId) {
    return predictionConfigState.movementCatalogByEdge?.[edgeId] || null;
}

function formatLaneIds(laneIds) {
    if (!Array.isArray(laneIds) || !laneIds.length) return '--';
    return laneIds.join(', ');
}

function getPredictionEdgeLabel(edgeId) {
    const catalog = getMovementCatalog(edgeId);
    if (!catalog) return edgeId;
    const laneCount = Number(catalog.lane_count || 0);
    const movementCount = Number(catalog.movement_count || 0);
    const laneLabel = laneCount > 0 ? `${laneCount}车道` : '车道未知';
    const movementLabel = movementCount > 0 ? `${movementCount}个转向检测` : '无转向检测';
    return `${edgeId}（${laneLabel} / ${movementLabel}）`;
}

function escapeHtml(text) {
    return String(text ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function getSignalCounts(state) {
    const text = String(state || '');
    const counts = { red: 0, yellow: 0, green: 0 };
    for (const char of text) {
        const lower = char.toLowerCase();
        if (lower === 'r') counts.red += 1;
        else if (lower === 'y') counts.yellow += 1;
        else if (lower === 'g') counts.green += 1;
    }
    return counts;
}

function getSignalDominantColor(state) {
    const counts = getSignalCounts(state);
    if (counts.yellow > 0) return 'yellow';
    if (counts.green > 0) return 'green';
    if (counts.red > 0) return 'red';
    return 'unknown';
}

function formatSignalCountdown(seconds) {
    const number = Number(seconds);
    if (!Number.isFinite(number)) return '--';
    return `${Math.max(0, Math.round(number))}s`;
}

function renderSignalMarkerContent(tl) {
    const dominant = getSignalDominantColor(tl.state);
    const phase = Number.isFinite(Number(tl.phase)) ? Number(tl.phase) : '--';
    return `
        <div class="signal-mini-marker" title="${escapeHtml(tl.id)} 相位 ${phase}">
            <span class="signal-dot red ${dominant === 'red' ? 'active' : ''}"></span>
            <span class="signal-dot yellow ${dominant === 'yellow' ? 'active' : ''}"></span>
            <span class="signal-dot green ${dominant === 'green' ? 'active' : ''}"></span>
            <span>P${escapeHtml(phase)}</span>
        </div>
    `;
}

function formatCo2Rate(rawValue) {
    const value = Number(rawValue);
    if (!Number.isFinite(value)) {
        return { value: '--', unit: 'mg/s' };
    }
    if (value >= 1000000) {
        return { value: (value / 1000000).toFixed(2), unit: 'kg/s' };
    }
    if (value >= 1000) {
        return { value: (value / 1000).toFixed(2), unit: 'g/s' };
    }
    return { value: value.toFixed(1), unit: 'mg/s' };
}

function renderSvgLineChart(container, options) {
    if (!container) return;
    const width = Math.max(container.clientWidth || 280, 220);
    const height = Math.max(container.clientHeight || 140, 68);
    const padding = options.padding || { top: 18, right: 12, bottom: 24, left: 34 };
    const plotWidth = Math.max(width - padding.left - padding.right, 40);
    const plotHeight = Math.max(height - padding.top - padding.bottom, 28);
    const labels = Array.isArray(options.labels) ? options.labels : [];
    const series = Array.isArray(options.series) ? options.series : [];
    const title = options.title || '';
    const legend = Array.isArray(options.legend) ? options.legend : [];

    const numericValues = [];
    series.forEach(item => {
        (item.data || []).forEach(value => {
            const number = Number(value);
            if (Number.isFinite(number)) numericValues.push(number);
        });
    });

    const minValue = numericValues.length ? Math.min(...numericValues) : 0;
    const maxValue = numericValues.length ? Math.max(...numericValues) : 1;
    const range = Math.abs(maxValue - minValue) < 1e-9 ? Math.max(Math.abs(maxValue) || 1, 1) : (maxValue - minValue);
    const yMin = minValue - range * 0.08;
    const yMax = maxValue + range * 0.08;
    const safeRange = Math.max(yMax - yMin, 1);

    const xForIndex = index => {
        if (labels.length <= 1) return padding.left + plotWidth / 2;
        return padding.left + (plotWidth * index) / (labels.length - 1);
    };
    const yForValue = value => padding.top + ((yMax - value) / safeRange) * plotHeight;

    let gridSvg = '';
    for (let i = 0; i <= 3; i += 1) {
        const y = padding.top + (plotHeight * i) / 3;
        const value = yMax - (safeRange * i) / 3;
        gridSvg += `<line x1="${padding.left}" y1="${y.toFixed(1)}" x2="${(padding.left + plotWidth).toFixed(1)}" y2="${y.toFixed(1)}" stroke="rgba(60,82,104,0.45)" stroke-width="1" />`;
        gridSvg += `<text x="${padding.left - 6}" y="${(y + 3).toFixed(1)}" text-anchor="end" fill="#89a9c8" font-size="9">${escapeHtml(formatPredictionValue(value, 1))}</text>`;
    }

    let xLabelsSvg = '';
    const labelStep = labels.length > 8 ? Math.ceil(labels.length / 4) : 1;
    labels.forEach((label, index) => {
        if (index % labelStep !== 0 && index !== labels.length - 1) return;
        xLabelsSvg += `<text x="${xForIndex(index).toFixed(1)}" y="${(padding.top + plotHeight + 16).toFixed(1)}" text-anchor="middle" fill="#89a9c8" font-size="9">${escapeHtml(label)}</text>`;
    });

    const seriesSvg = series.map(item => {
        const points = [];
        (item.data || []).forEach((value, index) => {
            const number = Number(value);
            if (Number.isFinite(number)) points.push([xForIndex(index), yForValue(number)]);
        });
        if (!points.length) return '';
        const pointString = points.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' ');
        const circles = points.map(([x, y]) => (
            `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="2.6" fill="${item.color}" stroke="#09111c" stroke-width="1" />`
        )).join('');
        return `<polyline fill="none" stroke="${item.color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" points="${pointString}" />${circles}`;
    }).join('');

    const legendSvg = legend.map((item, index) => {
        const x = padding.left + index * 88;
        return `<circle cx="${x}" cy="10" r="4" fill="${item.color}" /><text x="${x + 10}" y="13" fill="#a0cfff" font-size="10">${escapeHtml(item.name)}</text>`;
    }).join('');

    const titleSvg = title
        ? `<text x="${width / 2}" y="12" text-anchor="middle" fill="#a0cfff" font-size="12">${escapeHtml(title)}</text>`
        : '';

    container.innerHTML = `
        <svg width="100%" height="100%" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" aria-hidden="true">
            ${titleSvg}
            ${legendSvg}
            ${gridSvg}
            ${xLabelsSvg}
            ${seriesSvg}
        </svg>
    `;
}

const runtimeWarnings = [];
const mapAvailable = typeof AMap !== 'undefined';
const chartAvailable = typeof echarts !== 'undefined';

if (!mapAvailable) runtimeWarnings.push(I18N.warningMapFallback);
if (!chartAvailable) runtimeWarnings.push(I18N.warningChartFallback);

const map = mapAvailable ? new AMap.Map('mapContainer', {
    viewMode: '3D',
    pitch: 55,
    rotation: -15,
    showBuildingBlock: true,
    zoom: 16,
    mapStyle: 'amap://styles/dark',
    center: [114.402316, 38.007137]
}) : null;

const speedChart = chartAvailable ? echarts.init(document.getElementById('speedChart')) : null;
const realtimeChartEls = {
    flow: document.getElementById('predictionFlowChart'),
    speed: document.getElementById('predictionSpeedChart'),
    queue: document.getElementById('predictionQueueChart')
};
const compareChartEls = {
    flow: document.getElementById('compareFlowChart'),
    speed: document.getElementById('compareSpeedChart'),
    queue: document.getElementById('compareQueueChart')
};
const realtimeCharts = {
    flow: chartAvailable && realtimeChartEls.flow ? echarts.init(realtimeChartEls.flow) : null,
    speed: chartAvailable && realtimeChartEls.speed ? echarts.init(realtimeChartEls.speed) : null,
    queue: chartAvailable && realtimeChartEls.queue ? echarts.init(realtimeChartEls.queue) : null
};
const compareCharts = {
    flow: chartAvailable && compareChartEls.flow ? echarts.init(compareChartEls.flow) : null,
    speed: chartAvailable && compareChartEls.speed ? echarts.init(compareChartEls.speed) : null,
    queue: chartAvailable && compareChartEls.queue ? echarts.init(compareChartEls.queue) : null
};

const predictionModelSelectEl = document.getElementById('predictionModelSelect');
const predictionEdgeSelectEl = document.getElementById('predictionEdgeSelect');
const baselineRunSelectEl = document.getElementById('baselineRunSelect');
const incidentRunSelectEl = document.getElementById('incidentRunSelect');
const modeRealtimeBtn = document.getElementById('predictionModeRealtime');
const modeCompareBtn = document.getElementById('predictionModeCompare');
const realtimeViewEl = document.getElementById('realtimePredictionView');
const compareViewEl = document.getElementById('scenarioCompareView');
const compareControlsEl = document.getElementById('scenarioCompareControls');
const movementSegmentSwitchEl = document.getElementById('movementSegmentSwitch');
const movementSwitchHintEl = document.getElementById('movementSwitchHint');
const TURN_LABELS = { l: '左转', s: '直行', r: '右转' };

let timeData = [];
let speedData = [];
let pastPredData = [];
let lastRealtimePredictionData = null;
let lastComparePayload = null;
let selectedPredictionEdgeId = null;
let selectedMovementSegmentId = 'aggregate';
let currentMonitorEdgeId = 'ALL';
let predictionPanelMode = 'realtime';
let compareAffectedEdgeIds = [];
let predictionConfigState = {
    activeModel: 'ha_baseline',
    availableModels: [],
    scenarioCompareAvailable: false,
    observedEdges: [],
    movementCatalogByEdge: {}
};
let scenarioRunState = {
    baselineRuns: [],
    incidentRuns: [],
    selectedBaselineRunId: '',
    selectedIncidentRunId: ''
};

let roadLanes = [];
let roadPolylines = [];
let vehicleMarkers = [];
let tlMarkers = [];
let signalStopbarPolylines = [];
let incidentMarkers = {};
let compareIncidentEdgeMarkers = {};
let lastRadarData = [];
let lastTlData = [];
let lastIncidentsData = [];
let renderedIncidentsLog = new Set();

const infoWindow = mapAvailable ? new AMap.InfoWindow({
    offset: new AMap.Pixel(0, -30),
    isCustom: false
}) : null;

function getBaseRoadStyle() {
    return {
        strokeWeight: 4,
        strokeColor: '#409eff',
        strokeOpacity: 0.78,
        isOutline: true,
        outlineColor: '#0a1424',
        borderWeight: 2,
        zIndex: 40
    };
}

function getPredictionRoadStyle() {
    return {
        strokeWeight: 7,
        strokeColor: '#00f0ff',
        strokeOpacity: 0.92,
        isOutline: true,
        outlineColor: '#ffd84d',
        borderWeight: 4,
        zIndex: 95
    };
}

function getMonitorRoadStyle() {
    return {
        strokeWeight: 7,
        strokeColor: '#ff9900',
        strokeOpacity: 0.92,
        isOutline: true,
        outlineColor: '#fff4b8',
        borderWeight: 4,
        zIndex: 100
    };
}

function getSignalStopbarColor(stateChar) {
    const char = String(stateChar || '').trim();
    const lower = char.toLowerCase();
    if (lower === 'g' || lower === 'o') return '#42ff75';
    if (lower === 'y') return '#ffd84d';
    if (lower === 'r') return '#ff4d4f';
    return '#8ca0b4';
}

function getSignalStopbarStyle(stateChar) {
    return {
        strokeWeight: 6,
        strokeColor: getSignalStopbarColor(stateChar),
        strokeOpacity: 0.96,
        isOutline: true,
        outlineColor: 'rgba(5, 10, 18, 0.86)',
        borderWeight: 2,
        zIndex: 106
    };
}

function getIncidentRoadStyle() {
    return {
        strokeWeight: 8,
        strokeColor: '#ff3030',
        strokeOpacity: 0.96,
        strokeStyle: 'dashed',
        strokeDasharray: [12, 8],
        isOutline: true,
        outlineColor: '#ffeb3b',
        borderWeight: 5,
        zIndex: 96
    };
}

function buildLineChartOption(seriesDefs) {
    return {
        animation: false,
        tooltip: { trigger: 'axis' },
        grid: { top: 8, bottom: 18, left: 32, right: 8 },
        xAxis: {
            type: 'category',
            data: [],
            boundaryGap: false,
            axisLine: { lineStyle: { color: '#35506d' } },
            axisTick: { show: false },
            axisLabel: { color: '#89a9c8', fontSize: 9, interval: 4 }
        },
        yAxis: {
            type: 'value',
            splitNumber: 2,
            axisLine: { show: false },
            axisTick: { show: false },
            axisLabel: { color: '#89a9c8', fontSize: 9 },
            splitLine: { lineStyle: { color: 'rgba(60, 82, 104, 0.45)' } }
        },
        series: seriesDefs.map(series => ({
            name: series.name,
            type: 'line',
            data: [],
            smooth: true,
            symbol: 'circle',
            symbolSize: 5,
            showSymbol: true,
            itemStyle: { color: series.color },
            lineStyle: { color: series.color, width: 2 },
            areaStyle: series.area ? {
                color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                    { offset: 0, color: `${series.color}44` },
                    { offset: 1, color: `${series.color}05` }
                ])
            } : undefined
        }))
    };
}

function setLineChart(chart, labels, seriesValues) {
    if (!chart) return;
    chart.setOption({
        xAxis: { data: labels || [] },
        series: seriesValues.map(item => ({ data: item.data || [] }))
    });
}

function setLineChartFallback(container, labels, seriesValues) {
    renderSvgLineChart(container, {
        labels,
        title: '',
        padding: { top: 12, right: 8, bottom: 18, left: 32 },
        legend: [],
        series: seriesValues
    });
}

if (speedChart) {
    speedChart.setOption({
        title: { text: I18N.chartSpeedTitle, textStyle: { color: '#a0cfff', fontSize: 12 }, top: 0, left: 'center' },
        tooltip: { trigger: 'axis' },
        legend: {
            data: [I18N.seriesObserved, I18N.seriesForecast],
            textStyle: { color: '#a0cfff', fontSize: 10 },
            top: 22,
            itemWidth: 12,
            itemHeight: 8
        },
        grid: { top: 55, bottom: 25, left: 35, right: 15 },
        xAxis: {
            type: 'category',
            data: timeData,
            axisLine: { lineStyle: { color: '#606266' } },
            axisLabel: { color: '#a0cfff', fontSize: 10 }
        },
        yAxis: {
            type: 'value',
            splitLine: { lineStyle: { color: '#303133' } },
            axisLabel: { color: '#a0cfff', fontSize: 10 }
        },
        series: [
            {
                name: I18N.seriesObserved,
                type: 'line',
                data: speedData,
                itemStyle: { color: '#67c23a' },
                areaStyle: {
                    color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        { offset: 0, color: 'rgba(103, 194, 58, 0.3)' },
                        { offset: 1, color: 'rgba(103, 194, 58, 0.0)' }
                    ])
                },
                smooth: true
            },
            {
                name: I18N.seriesForecast,
                type: 'line',
                data: [],
                itemStyle: { color: '#e024df' },
                lineStyle: { type: 'dashed' },
                smooth: true
            }
        ]
    });
}

if (realtimeCharts.flow) realtimeCharts.flow.setOption(buildLineChartOption([{ name: I18N.chartFlow, color: '#00d9ff', area: true }]));
if (realtimeCharts.speed) realtimeCharts.speed.setOption(buildLineChartOption([{ name: I18N.chartSpeed, color: '#67c23a', area: true }]));
if (realtimeCharts.queue) realtimeCharts.queue.setOption(buildLineChartOption([{ name: I18N.chartQueue, color: '#ff9900', area: true }]));

const compareSeriesFlow = [
    { name: '正常', color: '#00d9ff', area: false },
    { name: '事故', color: '#ff7d7f', area: false }
];
const compareSeriesSpeed = [
    { name: '正常', color: '#67c23a', area: false },
    { name: '事故', color: '#ff7d7f', area: false }
];
const compareSeriesQueue = [
    { name: '正常', color: '#ffb84d', area: false },
    { name: '事故', color: '#ff4d4f', area: false }
];

if (compareCharts.flow) compareCharts.flow.setOption(buildLineChartOption(compareSeriesFlow));
if (compareCharts.speed) compareCharts.speed.setOption(buildLineChartOption(compareSeriesSpeed));
if (compareCharts.queue) compareCharts.queue.setOption(buildLineChartOption(compareSeriesQueue));

window.addEventListener('resize', () => {
    if (speedChart) speedChart.resize();
    Object.values(realtimeCharts).forEach(chart => chart && chart.resize());
    Object.values(compareCharts).forEach(chart => chart && chart.resize());
});

function appendRuntimeWarnings() {
    const statusEl = document.getElementById('systemStatus');
    if (!statusEl || runtimeWarnings.length === 0) return;
    let warningEl = document.getElementById('runtimeWarnings');
    if (!warningEl) {
        warningEl = document.createElement('div');
        warningEl.id = 'runtimeWarnings';
        warningEl.style.marginTop = '8px';
        warningEl.style.display = 'flex';
        warningEl.style.flexDirection = 'column';
        warningEl.style.alignItems = 'center';
        warningEl.style.gap = '4px';
        warningEl.style.pointerEvents = 'auto';
        statusEl.appendChild(warningEl);
    }
    warningEl.innerHTML = runtimeWarnings.map(message => (
        `<div style="padding:4px 10px;background:rgba(255,125,127,0.14);border:1px solid rgba(255,125,127,0.45);border-radius:999px;color:#ffd5d6;font-size:11px;letter-spacing:0;">${message}</div>`
    )).join('');
}

function buildHeaderMarkup(title) {
    return `
        ${title}
        <div class="edge-label" id="currentEdgeLabel">${I18N.scopeLabel}${I18N.scopeFull}</div>
        <button id="btnGlobalScope" style="margin-top: 10px; padding: 5px 15px; background: rgba(20, 30, 48, 0.85); border: 1px solid #0df; color: #0df; cursor: pointer; font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; border-radius: 4px; pointer-events: auto; letter-spacing: 2px;">${I18N.resetView}</button>
        <div style="display: flex; justify-content: space-between; font-size: 13px; color: #a0cfff; margin-top: 15px; width: 60%; margin-left: 20%; pointer-events: auto; font-family: 'Courier New', Consolas, monospace;">
            <span id="timeStart">2026-04-17 08:00:00</span>
            <span id="timeCurrent" style="display: inline-block; color: #fff; font-weight: bold; text-shadow: 0 0 5px #0df; text-align: center; width: 170px;">2026-04-17 08:00:00</span>
            <span id="timeEnd">2026-04-17 09:00:00</span>
        </div>
        <div class="progress-container" style="margin-top: 5px; width: 60%; margin-left: 20%;">
            <div class="progress-fill" id="progressBar"></div>
        </div>
    `;
}

function attachGlobalScopeButton() {
    const btn = document.getElementById('btnGlobalScope');
    if (!btn) return;
    btn.onclick = () => {
        if (window.appWs && window.appWs.readyState === WebSocket.OPEN) {
            window.appWs.send(JSON.stringify({ action: 'set_edge', edgeId: 'ALL' }));
        }
        currentMonitorEdgeId = 'ALL';
        const labelEl = document.getElementById('currentEdgeLabel');
        if (labelEl) {
            labelEl.innerHTML = `${I18N.scopeLabel}<span style="color:#ff9900">${I18N.scopeFull}</span>`;
        }
        refreshRoadHighlights();
    };
}

function renderHeaderStatus(title) {
    const statusEl = document.getElementById('systemStatus');
    if (!statusEl) return;
    statusEl.innerHTML = buildHeaderMarkup(title);
    attachGlobalScopeButton();
    appendRuntimeWarnings();
}

function refreshRoadHighlights() {
    if (!map || !roadPolylines.length) return;
    roadPolylines.forEach(polyline => {
        polyline.setOptions(getBaseRoadStyle());
    });

    if (currentMonitorEdgeId && currentMonitorEdgeId !== 'ALL') {
        setEdgePolylinesStyle(currentMonitorEdgeId, getMonitorRoadStyle());
    }

    if (selectedPredictionEdgeId && selectedPredictionEdgeId !== 'ALL') {
        setEdgePolylinesStyle(selectedPredictionEdgeId, getPredictionRoadStyle());
    }

    if (predictionPanelMode === 'compare') {
        compareAffectedEdgeIds.forEach(edgeId => {
            setEdgePolylinesStyle(edgeId, getIncidentRoadStyle());
        });
    }

    renderCompareIncidentEdgeMarkers();
}

function setEdgePolylinesStyle(edgeId, style) {
    roadPolylines
        .filter(item => item.getExtData()?.edgeId === edgeId)
        .forEach(polyline => polyline.setOptions(style));
}

function getEdgePolylineCenter(edgeId) {
    const polyline = roadPolylines.find(item => item.getExtData()?.edgeId === edgeId);
    if (!polyline || typeof polyline.getPath !== 'function') return null;
    const path = polyline.getPath();
    if (!Array.isArray(path) || !path.length) return null;
    const point = path[Math.floor(path.length / 2)];
    if (Array.isArray(point)) return point;
    if (point && typeof point.getLng === 'function' && typeof point.getLat === 'function') {
        return [point.getLng(), point.getLat()];
    }
    if (point && Number.isFinite(Number(point.lng)) && Number.isFinite(Number(point.lat))) {
        return [Number(point.lng), Number(point.lat)];
    }
    return null;
}

function clearCompareIncidentEdgeMarkers() {
    if (!map) return;
    Object.values(compareIncidentEdgeMarkers).forEach(marker => map.remove(marker));
    compareIncidentEdgeMarkers = {};
}

function renderCompareIncidentEdgeMarkers() {
    if (!map) return;
    if (predictionPanelMode !== 'compare' || !compareAffectedEdgeIds.length) {
        clearCompareIncidentEdgeMarkers();
        return;
    }
    const activeEdges = new Set(compareAffectedEdgeIds);
    Object.keys(compareIncidentEdgeMarkers).forEach(edgeId => {
        if (!activeEdges.has(edgeId)) {
            map.remove(compareIncidentEdgeMarkers[edgeId]);
            delete compareIncidentEdgeMarkers[edgeId];
        }
    });
    const eventType = lastComparePayload?.event_type || '';
    const speedFactor = Number(lastComparePayload?.incident_speed_factor || 0.25);
    const factorLabel = Number.isFinite(speedFactor) ? Math.round(speedFactor * 100) : 25;
    compareAffectedEdgeIds.forEach(edgeId => {
        const center = getEdgePolylineCenter(edgeId);
        if (!center) return;
        const markerText = eventType === 'incident_closure'
            ? '事故：封停'
            : `限速：${factorLabel}%`;
        const content = `<div class="compare-incident-marker">${markerText}</div>`;
        if (!compareIncidentEdgeMarkers[edgeId]) {
            const marker = new AMap.Marker({
                position: center,
                content,
                offset: new AMap.Pixel(-44, -28),
                zIndex: 107
            });
            map.add(marker);
            compareIncidentEdgeMarkers[edgeId] = marker;
        } else {
            compareIncidentEdgeMarkers[edgeId].setPosition(center);
            compareIncidentEdgeMarkers[edgeId].setContent(content);
        }
    });
}

function updateSignalStopbars() {
    if (!signalStopbarPolylines.length) return;
    const tlsStateById = new Map();
    lastTlData.forEach(tl => {
        if (tl && tl.id) tlsStateById.set(tl.id, String(tl.state || ''));
    });
    signalStopbarPolylines.forEach(polyline => {
        const meta = polyline.getExtData() || {};
        const state = tlsStateById.get(meta.tlsId) || '';
        const stateChar = state.charAt(Number(meta.linkIndex));
        polyline.setOptions(getSignalStopbarStyle(stateChar));
    });
}

function setPredictionPanelMode(mode) {
    predictionPanelMode = mode === 'compare' ? 'compare' : 'realtime';
    if (modeRealtimeBtn) modeRealtimeBtn.classList.toggle('active', predictionPanelMode === 'realtime');
    if (modeCompareBtn) modeCompareBtn.classList.toggle('active', predictionPanelMode === 'compare');
    if (realtimeViewEl) realtimeViewEl.classList.toggle('active', predictionPanelMode === 'realtime');
    if (compareViewEl) compareViewEl.classList.toggle('active', predictionPanelMode === 'compare');
    if (compareControlsEl) compareControlsEl.style.display = predictionPanelMode === 'compare' ? 'grid' : 'none';
    refreshRoadHighlights();

    if (predictionPanelMode === 'compare') {
        fetchScenarioRuns();
        requestScenarioCompare();
    } else if (lastRealtimePredictionData) {
        updateRealtimePredictionPanel(lastRealtimePredictionData);
    }
}

function updatePredictionEdgeOptions(edgeIds) {
    if (!predictionEdgeSelectEl) return;
    const previousEdgeId = selectedPredictionEdgeId;
    const currentEdgeIds = Array.from(predictionEdgeSelectEl.options).map(option => option.value);
    if (currentEdgeIds.join('|') !== edgeIds.join('|')) {
        predictionEdgeSelectEl.innerHTML = '';
        edgeIds.forEach(edgeId => {
            const option = document.createElement('option');
            option.value = edgeId;
            option.textContent = edgeId;
            predictionEdgeSelectEl.appendChild(option);
        });
    }
    Array.from(predictionEdgeSelectEl.options).forEach(option => {
        option.textContent = getPredictionEdgeLabel(option.value);
    });
    if (!selectedPredictionEdgeId || !edgeIds.includes(selectedPredictionEdgeId)) {
        selectedPredictionEdgeId = edgeIds[0] || null;
    }
    if (selectedPredictionEdgeId !== previousEdgeId) {
        selectedMovementSegmentId = 'aggregate';
    }
    predictionEdgeSelectEl.value = selectedPredictionEdgeId || '';
}

function getIncidentRunAffectedEdges(run) {
    if (!run) return [];
    if (Array.isArray(run.affected_edges)) return run.affected_edges.filter(Boolean);
    return String(run.affected_edges || '').split('|').filter(Boolean);
}

function isPredictionEdgeSelectable(edgeId) {
    if (!edgeId) return false;
    if (predictionConfigState.observedEdges.includes(edgeId)) return true;
    return Array.from(predictionEdgeSelectEl?.options || []).some(option => option.value === edgeId);
}

function getFirstSelectableAffectedEdge(run) {
    const affectedEdges = getIncidentRunAffectedEdges(run);
    if (!affectedEdges.length) return '';
    const selectableEdges = new Set([
        ...predictionConfigState.observedEdges,
        ...Array.from(predictionEdgeSelectEl?.options || []).map(option => option.value)
    ]);
    return affectedEdges.find(edgeId => selectableEdges.has(edgeId)) || affectedEdges[0];
}

function selectPredictionEdgeForCompare(edgeId) {
    if (!edgeId) return;
    selectedPredictionEdgeId = edgeId;
    selectedMovementSegmentId = 'aggregate';
    if (predictionEdgeSelectEl) {
        const hasOption = Array.from(predictionEdgeSelectEl.options).some(option => option.value === edgeId);
        if (hasOption) predictionEdgeSelectEl.value = edgeId;
    }
    refreshRoadHighlights();
}

function applyPredictionConfig(payload) {
    if (!payload) return;
    predictionConfigState.activeModel = payload.active_model || 'ha_baseline';
    predictionConfigState.availableModels = Array.isArray(payload.available_models) ? payload.available_models : [];
    predictionConfigState.scenarioCompareAvailable = Boolean(payload.scenario_compare_available);
    predictionConfigState.movementCatalogByEdge = payload.config?.movement_catalog_by_edge || payload.movement_catalog_by_edge || {};
    const configEdges = Array.isArray(payload.config?.observed_edges) ? payload.config.observed_edges : [];
    predictionConfigState.observedEdges = configEdges;
    if (configEdges.length) {
        updatePredictionEdgeOptions(configEdges);
        if (predictionPanelMode === 'compare') {
            const incidentRun = scenarioRunState.incidentRuns.find(run => run.run_id === scenarioRunState.selectedIncidentRunId);
            const defaultAffectedEdge = getFirstSelectableAffectedEdge(incidentRun);
            if (defaultAffectedEdge && (
                !isPredictionEdgeSelectable(selectedPredictionEdgeId)
                || !getIncidentRunAffectedEdges(incidentRun).includes(selectedPredictionEdgeId)
            )) {
                selectPredictionEdgeForCompare(defaultAffectedEdge);
            }
        }
    }

    if (predictionModelSelectEl) {
        const currentOptions = Array.from(predictionModelSelectEl.options).map(option => option.value);
        const nextOptions = predictionConfigState.availableModels;
        if (currentOptions.join('|') !== nextOptions.join('|')) {
            predictionModelSelectEl.innerHTML = '';
            nextOptions.forEach(modelName => {
                const option = document.createElement('option');
                option.value = modelName;
                option.textContent = formatModelName(modelName);
                predictionModelSelectEl.appendChild(option);
            });
        }
        predictionModelSelectEl.value = predictionConfigState.activeModel;
    }

    const activeModelEl = document.getElementById('predictionActiveModel');
    const activeBadgeEl = document.getElementById('predictionActiveBadge');
    const compareModelEl = document.getElementById('compareModelLabel');
    if (activeModelEl) activeModelEl.innerText = formatModelName(predictionConfigState.activeModel);
    if (activeBadgeEl) activeBadgeEl.innerText = formatModelName(predictionConfigState.activeModel);
    if (compareModelEl) compareModelEl.innerText = formatModelName(predictionConfigState.activeModel);
}

async function fetchPredictionConfig() {
    try {
        const response = await fetch('/api/prediction/config');
        const payload = await response.json();
        if (payload.status === 'ok') {
            applyPredictionConfig(payload);
        }
    } catch (error) {
        console.error('Failed to fetch prediction config:', error);
    }
}

async function fetchScenarioRuns() {
    if (!predictionConfigState.scenarioCompareAvailable) return;
    try {
        const response = await fetch('/api/prediction/scenario-runs');
        const payload = await response.json();
        if (payload.status === 'ok') {
            populateScenarioRunSelectors(payload);
        }
    } catch (error) {
        console.error('Failed to fetch scenario runs:', error);
    }
}

function populateScenarioRunSelectors(payload) {
    scenarioRunState.baselineRuns = Array.isArray(payload.baseline_runs) ? payload.baseline_runs : [];
    scenarioRunState.incidentRuns = Array.isArray(payload.incident_runs) ? payload.incident_runs : [];

    if (baselineRunSelectEl) {
        baselineRunSelectEl.innerHTML = '';
        scenarioRunState.baselineRuns.forEach(run => {
            const option = document.createElement('option');
            option.value = run.run_id;
            option.textContent = `${run.run_id} (x${Number(run.demand_scale).toFixed(2)})`;
            baselineRunSelectEl.appendChild(option);
        });
    }
    if (incidentRunSelectEl) {
        incidentRunSelectEl.innerHTML = '';
        scenarioRunState.incidentRuns.forEach(run => {
            const option = document.createElement('option');
            option.value = run.run_id;
            option.textContent = `${run.run_id} (${formatEventType(run.event_type, run.incident_type)})`;
            incidentRunSelectEl.appendChild(option);
        });
    }

    const incidentRun = scenarioRunState.incidentRuns.find(run => run.run_id === scenarioRunState.selectedIncidentRunId)
        || scenarioRunState.incidentRuns[0];
    scenarioRunState.selectedIncidentRunId = incidentRun ? incidentRun.run_id : '';
    const defaultAffectedEdge = getFirstSelectableAffectedEdge(incidentRun);
    if (defaultAffectedEdge && (
        !isPredictionEdgeSelectable(selectedPredictionEdgeId)
        || !getIncidentRunAffectedEdges(incidentRun).includes(selectedPredictionEdgeId)
    )) {
        selectPredictionEdgeForCompare(defaultAffectedEdge);
    }
    const requestedBaselineRunId = incidentRun?.recommended_baseline_run_id
        || scenarioRunState.selectedBaselineRunId
        || scenarioRunState.baselineRuns[0]?.run_id
        || '';
    scenarioRunState.selectedBaselineRunId = scenarioRunState.baselineRuns.some(run => run.run_id === requestedBaselineRunId)
        ? requestedBaselineRunId
        : (scenarioRunState.baselineRuns[0]?.run_id || '');

    if (incidentRunSelectEl) incidentRunSelectEl.value = scenarioRunState.selectedIncidentRunId;
    if (baselineRunSelectEl) baselineRunSelectEl.value = scenarioRunState.selectedBaselineRunId;

    if (predictionPanelMode === 'compare') {
        requestScenarioCompare();
    }
}

async function switchPredictionModel(modelName) {
    if (!modelName) return;
    try {
        if (predictionModelSelectEl) predictionModelSelectEl.disabled = true;
        const response = await fetch('/api/prediction/active-model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_name: modelName })
        });
        const payload = await response.json();
        if (!response.ok) {
            throw new Error(payload.detail || 'Failed to switch prediction model');
        }
        applyPredictionConfig(payload);
        if (payload.prediction) {
            lastRealtimePredictionData = payload.prediction;
            if (predictionPanelMode === 'realtime') {
                updateRealtimePredictionPanel(payload.prediction);
            }
        }
        if (predictionPanelMode === 'compare') {
            requestScenarioCompare();
        }
    } catch (error) {
        console.error('Failed to switch prediction model:', error);
        if (predictionModelSelectEl) predictionModelSelectEl.value = predictionConfigState.activeModel;
    } finally {
        if (predictionModelSelectEl) predictionModelSelectEl.disabled = false;
    }
}

async function requestScenarioCompare() {
    if (predictionPanelMode !== 'compare') return;
    if (!scenarioRunState.selectedBaselineRunId || !scenarioRunState.selectedIncidentRunId || !selectedPredictionEdgeId) return;
    try {
        const response = await fetch('/api/prediction/scenario-compare', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                baseline_run_id: scenarioRunState.selectedBaselineRunId,
                incident_run_id: scenarioRunState.selectedIncidentRunId,
                edge_id: selectedPredictionEdgeId,
                model_name: predictionModelSelectEl ? predictionModelSelectEl.value : predictionConfigState.activeModel,
                horizon: 15
            })
        });
        const payload = await response.json();
        if (!response.ok) {
            throw new Error(payload.detail || 'Failed to compare scenarios');
        }
        updateScenarioComparePanel(payload);
    } catch (error) {
        console.error('Failed to compare scenarios:', error);
    }
}

function getMovementsForSelectedEdge(prediction, edgeId) {
    const catalog = getMovementCatalog(edgeId);
    const catalogMovements = Array.isArray(catalog?.movements) ? catalog.movements : [];
    const byId = new Map();
    (prediction?.movements || []).forEach(movement => {
        if (movement.incoming_edge === edgeId || catalogMovements.some(item => item.movement_id === movement.movement_id)) {
            byId.set(movement.movement_id, movement);
        }
    });
    if (!catalogMovements.length) {
        return Array.from(byId.values());
    }
    return catalogMovements.map(meta => ({ ...meta, ...(byId.get(meta.movement_id) || {}) }));
}

function getMovementSeries(movement) {
    const flowSeries = movement?.pred_arrival_flow || movement?.pred_flow || [];
    const speedSeries = movement?.pred_mean_speed || movement?.pred_speed || [];
    const queueSeries = movement?.pred_queue_veh || movement?.pred_queue || [];
    return {
        flow: flowSeries.map(value => Number(value)),
        speed: speedSeries.map(value => Number(value) * 3.6),
        queue: queueSeries.map(value => Number(value))
    };
}

function getNodeSeries(node) {
    return {
        flow: (node?.pred_flow || []).map(value => Number(value)),
        speed: (node?.pred_speed || []).map(value => Number(value) * 3.6),
        queue: (node?.pred_queue || []).map(value => Number(value))
    };
}

function getMovementSegmentLabel(movement) {
    return `${getTurnLabel(movement.turn_type)}→${movement.outgoing_edge || '--'}`;
}

function getMovementSegmentTitle(movement) {
    const laneText = formatLaneIds(movement.lane_ids);
    const zoneText = formatMeters(movement.zone_length_m);
    const qualityText = getZoneQualityLabel(movement.zone_quality || 'ok');
    return `车道组：${laneText}；检测区：${zoneText}；状态：${qualityText}`;
}

function getMovementSwitchHint(edgeId, movements) {
    if (selectedMovementSegmentId === 'aggregate') {
        const catalog = getMovementCatalog(edgeId);
        if (!catalog) return '进口道聚合';
        const movementCount = Number(catalog.movement_count || movements.length || 0);
        return `聚合 / ${movementCount}个转向 / 检测区${formatMeters(catalog.zone_length_m)}`;
    }
    const movement = movements.find(item => item.movement_id === selectedMovementSegmentId);
    if (!movement) return '进口道聚合';
    return `${getMovementSegmentLabel(movement)} / ${formatLaneIds(movement.lane_ids)} / ${formatMeters(movement.zone_length_m)}`;
}

function renderMovementSegmentSwitch(edgeId, prediction) {
    if (!movementSegmentSwitchEl) return;
    const movements = getMovementsForSelectedEdge(prediction, edgeId);
    const hasSelectedMovement = movements.some(item => item.movement_id === selectedMovementSegmentId);
    if (selectedMovementSegmentId !== 'aggregate' && !hasSelectedMovement) {
        selectedMovementSegmentId = 'aggregate';
    }
    movementSegmentSwitchEl.innerHTML = '';
    const options = [
        {
            id: 'aggregate',
            label: '聚合',
            title: '当前进口道聚合预测'
        },
        ...movements.map(movement => ({
            id: movement.movement_id,
            label: getMovementSegmentLabel(movement),
            title: getMovementSegmentTitle(movement)
        }))
    ];
    options.forEach(option => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = `movement-segment-btn${option.id === selectedMovementSegmentId ? ' active' : ''}`;
        button.textContent = option.label;
        button.title = option.title;
        button.onclick = () => {
            selectedMovementSegmentId = option.id;
            if (lastRealtimePredictionData) {
                updateRealtimePredictionPanel(lastRealtimePredictionData);
            }
        };
        movementSegmentSwitchEl.appendChild(button);
    });
    if (movementSwitchHintEl) {
        movementSwitchHintEl.textContent = getMovementSwitchHint(edgeId, movements);
        movementSwitchHintEl.title = movementSwitchHintEl.textContent;
    }
}

function getSelectedPredictionSeries(prediction, node, edgeId) {
    if (selectedMovementSegmentId === 'aggregate') {
        return getNodeSeries(node);
    }
    const movement = getMovementsForSelectedEdge(prediction, edgeId)
        .find(item => item.movement_id === selectedMovementSegmentId);
    return movement ? getMovementSeries(movement) : getNodeSeries(node);
}

function updateRealtimePredictionPanel(prediction) {
    if (!prediction || !prediction.nodes || !prediction.nodes.length) return;
    lastRealtimePredictionData = prediction;
    updatePredictionEdgeOptions(prediction.nodes.map(node => node.edge_id));

    const node = prediction.nodes.find(item => item.edge_id === selectedPredictionEdgeId) || prediction.nodes[0];
    if (node && selectedPredictionEdgeId !== node.edge_id) {
        selectedPredictionEdgeId = node.edge_id;
    }
    renderMovementSegmentSwitch(selectedPredictionEdgeId, prediction);
    const horizon = prediction.horizon || [];
    const labels = horizon.map(step => `+${step}m`);
    const selectedSeries = getSelectedPredictionSeries(prediction, node, selectedPredictionEdgeId);
    const flow = selectedSeries.flow;
    const speed = selectedSeries.speed;
    const queue = selectedSeries.queue;
    const historySize = Number(prediction.history_size || 0);
    const historyRequired = Number(prediction.history_required || 12);
    const activeModelName = prediction.active_model || predictionConfigState.activeModel;

    const activeModelEl = document.getElementById('predictionActiveModel');
    const activeBadgeEl = document.getElementById('predictionActiveBadge');
    const outputModelEl = document.getElementById('predictionModel');
    const historyEl = document.getElementById('predictionHistoryWindow');
    const horizonEl = document.getElementById('predictionHorizon');
    const flowLeadEl = document.getElementById('predictionFlowLead');
    const speedLeadEl = document.getElementById('predictionSpeedLead');
    const queueLeadEl = document.getElementById('predictionQueueLead');

    if (activeModelEl) activeModelEl.innerText = formatModelName(activeModelName);
    if (activeBadgeEl) activeBadgeEl.innerText = formatModelName(activeModelName);
    if (outputModelEl) outputModelEl.innerText = formatModelName(prediction.model);
    if (historyEl) historyEl.innerText = `${historySize} / ${historyRequired}`;
    if (horizonEl) horizonEl.innerText = `${horizon.length || 0} ${I18N.minuteUnit}`;
    if (flowLeadEl) flowLeadEl.innerText = formatPredictionValue(flow[0]);
    if (speedLeadEl) speedLeadEl.innerText = `${formatPredictionValue(speed[0])} km/h`;
    if (queueLeadEl) queueLeadEl.innerText = formatPredictionValue(queue[0]);

    const flowSeries = [{ color: '#00d9ff', data: flow }];
    const speedSeries = [{ color: '#67c23a', data: speed }];
    const queueSeries = [{ color: '#ff9900', data: queue }];

    if (realtimeCharts.flow) setLineChart(realtimeCharts.flow, labels, flowSeries);
    else setLineChartFallback(realtimeChartEls.flow, labels, flowSeries);
    if (realtimeCharts.speed) setLineChart(realtimeCharts.speed, labels, speedSeries);
    else setLineChartFallback(realtimeChartEls.speed, labels, speedSeries);
    if (realtimeCharts.queue) setLineChart(realtimeCharts.queue, labels, queueSeries);
    else setLineChartFallback(realtimeChartEls.queue, labels, queueSeries);

    refreshRoadHighlights();
}

function updateScenarioComparePanel(payload) {
    if (!payload || !payload.baseline_pred || !payload.incident_pred) return;
    lastComparePayload = payload;
    compareAffectedEdgeIds = Array.isArray(payload.affected_edges) ? payload.affected_edges : [];
    updatePredictionEdgeOptions(payload.baseline_pred.nodes.map(node => node.edge_id));

    const baselineNode = payload.baseline_pred.nodes.find(node => node.edge_id === selectedPredictionEdgeId) || payload.baseline_pred.nodes[0];
    const incidentNode = payload.incident_pred.nodes.find(node => node.edge_id === selectedPredictionEdgeId) || payload.incident_pred.nodes[0];
    const delta = payload.delta || {};
    const labels = (payload.baseline_pred.horizon || []).map(step => `+${step}m`);

    const flowNormal = (baselineNode.pred_flow || []).map(Number);
    const flowIncident = (incidentNode.pred_flow || []).map(Number);
    const speedNormal = (baselineNode.pred_speed || []).map(value => Number(value) * 3.6);
    const speedIncident = (incidentNode.pred_speed || []).map(value => Number(value) * 3.6);
    const queueNormal = (baselineNode.pred_queue || []).map(Number);
    const queueIncident = (incidentNode.pred_queue || []).map(Number);
    const selectedIsDirectAffected = compareAffectedEdgeIds.includes(selectedPredictionEdgeId);
    const incidentStart = Number(payload.incident_start_s ?? payload.anchor_step ?? 0);
    const incidentEnd = Number(payload.incident_end_s ?? 0);
    const incidentSpeedFactor = Number(payload.incident_speed_factor || 0.25);
    const eventType = payload.event_type || '';
    const speedFactorLabel = Number.isFinite(incidentSpeedFactor) ? Math.round(incidentSpeedFactor * 100) : 25;
    const noticeEl = document.getElementById('compareIncidentNotice');
    const eventLabel = formatEventType(eventType, payload.incident_type);
    const eventActionText = eventType === 'incident_closure'
        ? `禁止 passenger 车辆进入影响路段`
        : `最大速度降至基础速度的 ${speedFactorLabel}%`;

    document.getElementById('compareModelLabel').innerText = formatModelName(payload.model_name || predictionConfigState.activeModel);
    document.getElementById('compareIncidentType').innerText = eventLabel !== '--'
        ? `${eventLabel} / ${eventType === 'incident_closure' ? '封停' : `限速${speedFactorLabel}%`}`
        : '--';
    document.getElementById('compareAnchorStep').innerText = incidentEnd > incidentStart
        ? `${incidentStart}s - ${incidentEnd}s`
        : `${payload.anchor_step || 0}s`;
    document.getElementById('compareAffectedEdges').innerText = compareAffectedEdgeIds.length
        ? `${compareAffectedEdgeIds.join(', ')}（红色虚线）`
        : '--';
    if (noticeEl) {
        noticeEl.classList.toggle('indirect', !selectedIsDirectAffected);
        const affectedText = compareAffectedEdgeIds.length ? compareAffectedEdgeIds.join(', ') : '--';
        noticeEl.innerText = selectedIsDirectAffected
            ? `当前查看的是事件直接影响路段 ${selectedPredictionEdgeId}。${eventLabel}：${incidentStart}s-${incidentEnd}s 内，${affectedText} ${eventActionText}。`
            : `当前查看的 ${selectedPredictionEdgeId} 不是事件直接影响路段，曲线主要表示外溢影响。直接影响路段为 ${affectedText}，事件窗口 ${incidentStart}s-${incidentEnd}s，处理方式：${eventActionText}。`;
    }
    document.getElementById('compareFlowDelta').innerText = formatSignedValue((delta.delta_flow || [])[0], 2);
    document.getElementById('compareSpeedDelta').innerText = formatSignedValue(((delta.delta_speed || [])[0] || 0) * 3.6, 2, ' km/h');
    document.getElementById('compareQueueDelta').innerText = formatSignedValue((delta.delta_queue || [])[0], 2);
    document.getElementById('compareFlowLead').innerText = `常 ${formatPredictionValue(flowNormal[0])} / 事 ${formatPredictionValue(flowIncident[0])}`;
    document.getElementById('compareSpeedLead').innerText = `常 ${formatPredictionValue(speedNormal[0])} / 事 ${formatPredictionValue(speedIncident[0])}`;
    document.getElementById('compareQueueLead').innerText = `常 ${formatPredictionValue(queueNormal[0])} / 事 ${formatPredictionValue(queueIncident[0])}`;

    const flowSeries = [
        { color: compareSeriesFlow[0].color, data: flowNormal },
        { color: compareSeriesFlow[1].color, data: flowIncident }
    ];
    const speedSeries = [
        { color: compareSeriesSpeed[0].color, data: speedNormal },
        { color: compareSeriesSpeed[1].color, data: speedIncident }
    ];
    const queueSeries = [
        { color: compareSeriesQueue[0].color, data: queueNormal },
        { color: compareSeriesQueue[1].color, data: queueIncident }
    ];

    if (compareCharts.flow) setLineChart(compareCharts.flow, labels, flowSeries);
    else setLineChartFallback(compareChartEls.flow, labels, flowSeries);
    if (compareCharts.speed) setLineChart(compareCharts.speed, labels, speedSeries);
    else setLineChartFallback(compareChartEls.speed, labels, speedSeries);
    if (compareCharts.queue) setLineChart(compareCharts.queue, labels, queueSeries);
    else setLineChartFallback(compareChartEls.queue, labels, queueSeries);

    refreshRoadHighlights();
}

function updateMapVectors() {
    if (!map) return;
    while (vehicleMarkers.length < lastRadarData.length) {
        const marker = new AMap.Marker({
            position: [0, 0],
            content: renderVehicleMarkerContent({ vClass: 'passenger', angle: 0 }),
            offset: new AMap.Pixel(0, 0),
            zIndex: 110
        });
        map.add(marker);
        vehicleMarkers.push(marker);
    }
    while (vehicleMarkers.length > lastRadarData.length) {
        const marker = vehicleMarkers.pop();
        map.remove(marker);
    }
    for (let i = 0; i < lastRadarData.length; i += 1) {
        const vehicle = lastRadarData[i];
        vehicleMarkers[i].setPosition([vehicle.x, vehicle.y]);
        vehicleMarkers[i].setContent(renderVehicleMarkerContent(vehicle));
    }

    while (tlMarkers.length < lastTlData.length) {
        const marker = new AMap.Marker({
            position: [0, 0],
            content: renderSignalMarkerContent({ id: '', state: '', phase: '--' }),
            offset: new AMap.Pixel(-34, -10),
            zIndex: 108
        });
        marker.on('click', event => {
            const data = event.target.getExtData();
            if (!data || !infoWindow) return;
            const counts = getSignalCounts(data.state);
            const content = `
                <div style="color:#333; min-width:160px;">
                    <b>信号灯：${escapeHtml(data.id)}</b><br/>
                    相位：P${escapeHtml(data.phase)}<br/>
                    下一次切换：${formatSignalCountdown(data.time_to_switch)}<br/>
                    状态：G${counts.green} / Y${counts.yellow} / R${counts.red}
                </div>
            `;
            infoWindow.setContent(content);
            infoWindow.open(map, marker.getPosition());
        });
        map.add(marker);
        tlMarkers.push(marker);
    }
    while (tlMarkers.length > lastTlData.length) {
        const marker = tlMarkers.pop();
        map.remove(marker);
    }
    for (let i = 0; i < lastTlData.length; i += 1) {
        const tl = lastTlData[i];
        tlMarkers[i].setPosition([tl.x, tl.y]);
        tlMarkers[i].setContent(renderSignalMarkerContent(tl));
        tlMarkers[i].setExtData(tl);
    }
    updateSignalStopbars();

    const currentIncidentIds = new Set(lastIncidentsData.map(inc => inc.id));
    Object.keys(incidentMarkers).forEach(id => {
        if (!currentIncidentIds.has(id)) {
            map.remove(incidentMarkers[id]);
            delete incidentMarkers[id];
        }
    });

    lastIncidentsData.forEach(inc => {
        if (!incidentMarkers[inc.id]) {
            const marker = new AMap.Marker({
                position: inc.lnglat,
                icon: new AMap.Icon({
                    size: new AMap.Size(24, 24),
                    image: 'https://webapi.amap.com/theme/v1.3/markers/n/mark_r.png',
                    imageSize: new AMap.Size(24, 24)
                }),
                offset: new AMap.Pixel(-12, -24),
                zIndex: 200,
                extData: inc
            });
            marker.on('click', event => {
                const data = event.target.getExtData();
                infoWindow.setContent(`<div style="color:#333;"><b>${I18N.alertTitle}${data.road_name}</b><br/><span>${data.desc}</span></div>`);
                infoWindow.open(map, marker.getPosition());
            });
            map.add(marker);
            incidentMarkers[inc.id] = marker;
        } else {
            incidentMarkers[inc.id].setExtData(inc);
            incidentMarkers[inc.id].setPosition(inc.lnglat);
        }
    });
}

function updateIncidentLog(incidents) {
    const listEl = document.getElementById('incidentList');
    const countEl = document.getElementById('incidentCount');
    if (!listEl) return;
    if (countEl) countEl.innerText = `${incidents.length}${I18N.incidentActiveSuffix}`;

    incidents.forEach(inc => {
        if (!renderedIncidentsLog.has(inc.id)) {
            renderedIncidentsLog.add(inc.id);
            const item = document.createElement('div');
            item.id = `log-${inc.id}`;
            item.style.padding = '7px 9px';
            item.style.borderLeft = '2px solid #ff4d4f';
            item.style.background = 'rgba(255, 77, 79, 0.08)';
            item.style.borderRadius = '4px';
            item.style.order = '1';

            const timeStr = new Date().toLocaleTimeString('zh-CN', { hour12: false });
            item.innerHTML = `<div class="log-title"><span style="color:#ff4d4f">${I18N.alertTag}</span> <b>${inc.road_name}</b> <span style="float:right; font-size:11px; color:#9eb8d2; font-weight:normal;">[${timeStr}]</span></div><div class="log-desc" style="color:#d8e8ff; font-size:11px; margin-top:4px; line-height:1.35;">${inc.desc}</div>`;
            listEl.prepend(item);

            while (listEl.children.length > 10) {
                listEl.removeChild(listEl.lastElementChild);
            }
        } else {
            const item = document.getElementById(`log-${inc.id}`);
            if (item && item.getAttribute('data-resolved') !== 'true') {
                const descDiv = item.querySelector('.log-desc');
                if (descDiv && descDiv.innerHTML !== inc.desc) {
                    descDiv.innerHTML = inc.desc;
                }
            }
        }
    });

    const currentIds = new Set(incidents.map(inc => inc.id));
    Array.from(renderedIncidentsLog).forEach(id => {
        if (!currentIds.has(id)) {
            const item = document.getElementById(`log-${id}`);
            if (item && item.getAttribute('data-resolved') !== 'true') {
                item.setAttribute('data-resolved', 'true');
                item.style.borderLeft = '2px solid #67c23a';
                item.style.background = 'rgba(103, 194, 58, 0.08)';
                item.style.order = '2';

                const titleDiv = item.querySelector('.log-title');
                if (titleDiv) {
                    titleDiv.innerHTML = titleDiv.innerHTML.replace(I18N.alertTag, I18N.clearTag).replace('#ff4d4f', '#67c23a');
                }

                const resolvedText = document.createElement('div');
                resolvedText.style.color = '#67c23a';
                resolvedText.style.fontSize = '11px';
                resolvedText.style.marginTop = '4px';
                resolvedText.innerText = I18N.recovered;
                item.appendChild(resolvedText);

                renderedIncidentsLog.delete(id);
            }
        }
    });
}

function updateDashboard(data) {
    if (!data.stats) return;

    const progress = Math.min((data.step / 3600) * 100, 100);
    const progressBar = document.getElementById('progressBar');
    if (progressBar) progressBar.style.width = `${progress}%`;

    const startTime = new Date('2026-04-17T08:00:00');
    const currentTime = new Date(startTime.getTime() + data.step * 1000);
    const currentStr = `${currentTime.getFullYear()}-${String(currentTime.getMonth() + 1).padStart(2, '0')}-${String(currentTime.getDate()).padStart(2, '0')} ${String(currentTime.getHours()).padStart(2, '0')}:${String(currentTime.getMinutes()).padStart(2, '0')}:${String(currentTime.getSeconds()).padStart(2, '0')}`;
    const timeCurrentEl = document.getElementById('timeCurrent');
    if (timeCurrentEl) timeCurrentEl.innerText = currentStr;

    document.getElementById('valFlow').innerHTML = `<span style="display:inline-block; width: 60px; font-family: 'Courier New', Consolas, monospace; text-align: right;">${data.stats.flow}</span><span class="metric-unit">${I18N.unitVehicle}</span>`;
    document.getElementById('valSpeed').innerHTML = `<span style="display:inline-block; width: 60px; font-family: 'Courier New', Consolas, monospace; text-align: right;">${(data.stats.speed * 3.6).toFixed(2)}</span><span class="metric-unit">km/h</span>`;
    document.getElementById('valQueue').innerHTML = `<span style="display:inline-block; width: 60px; font-family: 'Courier New', Consolas, monospace; text-align: right;">${data.stats.queue}</span><span class="metric-unit">${I18N.unitVehicle}</span>`;
    const co2Display = formatCo2Rate(data.stats.co2);
    document.getElementById('valCO2').innerHTML = `<span style="display:inline-block; width: 60px; font-family: 'Courier New', Consolas, monospace; text-align: right;">${co2Display.value}</span><span class="metric-unit">${co2Display.unit}</span>`;

    if (timeData.length > 30) {
        timeData.shift();
        speedData.shift();
        pastPredData.shift();
    }
    timeData.push(`T-${data.step}`);
    speedData.push((data.stats.speed * 3.6).toFixed(2));

    const currentPrediction = (data.stats.current_pred !== undefined) ? data.stats.current_pred : data.stats.speed;
    pastPredData.push((currentPrediction * 3.6).toFixed(2));

    const displayTimeData = [...timeData];
    const displaySpeedData = [...speedData];
    const displayPredSpeedData = [...pastPredData];

    if (Array.isArray(data.stats.future_speeds) && data.stats.future_speeds.length > 0) {
        for (let i = 0; i < data.stats.future_speeds.length; i += 1) {
            displayTimeData.push(`T+${i + 1}m`);
            displaySpeedData.push(null);
            displayPredSpeedData.push((data.stats.future_speeds[i] * 3.6).toFixed(2));
        }
    }

    if (speedChart) {
        speedChart.setOption({
            xAxis: { data: displayTimeData },
            series: [{ data: displaySpeedData }, { data: displayPredSpeedData }]
        });
    } else {
        renderSvgLineChart(document.getElementById('speedChart'), {
            labels: displayTimeData,
            title: I18N.chartSpeedTitle,
            legend: [
                { name: I18N.seriesObserved, color: '#67c23a' },
                { name: I18N.seriesForecast, color: '#e024df' }
            ],
            padding: { top: 28, right: 14, bottom: 24, left: 35 },
            series: [
                { color: '#67c23a', data: displaySpeedData },
                { color: '#e024df', data: displayPredSpeedData }
            ]
        });
    }
}

function connectWS() {
    const protocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
    window.appWs = new WebSocket(`${protocol}${window.location.host}/ws`);

    window.appWs.onopen = () => {
        renderHeaderStatus(I18N.titleOnline);
        fetchPredictionConfig();
        fetchScenarioRuns();
    };

    window.appWs.onmessage = event => {
        const payload = JSON.parse(event.data);
        lastRadarData = payload.radar || [];
        lastTlData = payload.tls || [];
        lastIncidentsData = payload.incidents || [];
        updateDashboard(payload);
        updateMapVectors();
        updateIncidentLog(lastIncidentsData);
        if (payload.prediction) {
            lastRealtimePredictionData = payload.prediction;
            if (predictionPanelMode === 'realtime') {
                updateRealtimePredictionPanel(payload.prediction);
            }
        }
    };

    window.appWs.onclose = () => {
        renderHeaderStatus(I18N.titleReconnect);
        setTimeout(connectWS, 1500);
    };
}

function loadNetworkToMap() {
    if (!map) return;
    fetchPredictionConfig();
    fetchScenarioRuns();
    fetch('/network')
        .then(response => response.json())
        .then(data => {
            if (data.status !== 'ok') {
                console.error('Network fetch failed:', data.message);
                return;
            }

            roadPolylines.forEach(polyline => map.remove(polyline));
            roadPolylines = [];
            signalStopbarPolylines.forEach(polyline => map.remove(polyline));
            signalStopbarPolylines = [];

            roadLanes = data.lanes || [];
            let sumLon = 0;
            let sumLat = 0;
            let pointCount = 0;

            roadLanes.forEach(lane => {
                if (lane.shape && lane.shape.length >= 2) {
                    const path = lane.shape.map(point => new AMap.LngLat(point[0], point[1]));
                    const polyline = new AMap.Polyline({
                        path,
                        bubble: true,
                        cursor: 'pointer',
                        extData: { edgeId: lane.edgeId },
                        ...getBaseRoadStyle()
                    });

                    polyline.on('click', event => {
                        currentMonitorEdgeId = event.target.getExtData().edgeId;
                        const labelEl = document.getElementById('currentEdgeLabel');
                        if (labelEl) {
                            labelEl.innerHTML = `${I18N.edgeLabel}<span style="color:#ff9900">${currentMonitorEdgeId}</span>`;
                        }
                        if (window.appWs && window.appWs.readyState === WebSocket.OPEN) {
                            window.appWs.send(JSON.stringify({ action: 'set_edge', edgeId: currentMonitorEdgeId }));
                        }
                        refreshRoadHighlights();
                    });

                    map.add(polyline);
                    roadPolylines.push(polyline);
                }

                (lane.shape || []).forEach(point => {
                    sumLon += point[0];
                    sumLat += point[1];
                    pointCount += 1;
                });
            });

            (data.signalStopbars || []).forEach(stopbar => {
                if (!stopbar.shape || stopbar.shape.length < 2) return;
                const path = stopbar.shape.map(point => new AMap.LngLat(point[0], point[1]));
                const polyline = new AMap.Polyline({
                    path,
                    bubble: false,
                    cursor: 'pointer',
                    extData: stopbar,
                    ...getSignalStopbarStyle('r')
                });
                polyline.on('click', event => {
                    const meta = event.target.getExtData() || {};
                    if (!infoWindow) return;
                    const state = (lastTlData.find(tl => tl.id === meta.tlsId)?.state || '').charAt(Number(meta.linkIndex));
                    const statusLabel = state ? state.toUpperCase() : '--';
                    infoWindow.setContent(`
                        <div style="color:#333; min-width:180px;">
                            <b>停止线信号</b><br/>
                            信号灯：${escapeHtml(meta.tlsId)}<br/>
                            进口道：${escapeHtml(meta.edgeId)}<br/>
                            车道：${escapeHtml(meta.laneId)}<br/>
                            方向：${escapeHtml(meta.dir || '--')} / link ${escapeHtml(meta.linkIndex)}<br/>
                            当前状态：${escapeHtml(statusLabel)}
                        </div>
                    `);
                    infoWindow.open(map, event.lnglat);
                });
                map.add(polyline);
                signalStopbarPolylines.push(polyline);
            });
            updateSignalStopbars();

            if (pointCount > 0) {
                map.setCenter([sumLon / pointCount, sumLat / pointCount]);
                map.setZoom(16);
            }

            refreshRoadHighlights();
        })
        .catch(error => {
            console.error('Error fetching /network:', error);
        });
}

if (predictionModelSelectEl) {
    predictionModelSelectEl.onchange = () => switchPredictionModel(predictionModelSelectEl.value);
}

if (predictionEdgeSelectEl) {
    predictionEdgeSelectEl.onchange = () => {
        selectedPredictionEdgeId = predictionEdgeSelectEl.value;
        selectedMovementSegmentId = 'aggregate';
        if (predictionPanelMode === 'compare') {
            requestScenarioCompare();
        } else {
            updateRealtimePredictionPanel(lastRealtimePredictionData);
        }
    };
}

if (baselineRunSelectEl) {
    baselineRunSelectEl.onchange = () => {
        scenarioRunState.selectedBaselineRunId = baselineRunSelectEl.value;
        requestScenarioCompare();
    };
}

if (incidentRunSelectEl) {
    incidentRunSelectEl.onchange = () => {
        scenarioRunState.selectedIncidentRunId = incidentRunSelectEl.value;
        const selectedIncident = scenarioRunState.incidentRuns.find(run => run.run_id === incidentRunSelectEl.value);
        selectPredictionEdgeForCompare(getFirstSelectableAffectedEdge(selectedIncident));
        if (selectedIncident?.recommended_baseline_run_id) {
            scenarioRunState.selectedBaselineRunId = selectedIncident.recommended_baseline_run_id;
            if (baselineRunSelectEl) baselineRunSelectEl.value = scenarioRunState.selectedBaselineRunId;
        }
        requestScenarioCompare();
    };
}

if (modeRealtimeBtn) modeRealtimeBtn.onclick = () => setPredictionPanelMode('realtime');
if (modeCompareBtn) modeCompareBtn.onclick = () => setPredictionPanelMode('compare');

renderHeaderStatus(I18N.titleDefault);
connectWS();
if (map) {
    map.on('complete', () => loadNetworkToMap());
    map.on('zoomend', () => updateMapVectors());
    map.on('rotateend', () => updateMapVectors());
    map.on('pitchend', () => updateMapVectors());
} else {
    fetchPredictionConfig();
    fetchScenarioRuns();
}
