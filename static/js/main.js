const vClassColors = {
    passenger: '#409eff',
    truck: '#e6a23c',
    bus: '#67c23a',
    motorcycle: '#e024df',
    emergency: '#f56c6c',
    pedestrian: '#ffeb3b',
    DEFAULT: '#ffffff'
};

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
    transformer_v1: 'Transformer V1'
};

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

function escapeHtml(text) {
    return String(text ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
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

let timeData = [];
let speedData = [];
let pastPredData = [];
let lastRealtimePredictionData = null;
let lastComparePayload = null;
let selectedPredictionEdgeId = null;
let currentMonitorEdgeId = 'ALL';
let predictionPanelMode = 'realtime';
let compareAffectedEdgeIds = [];
let predictionConfigState = {
    activeModel: 'ha_baseline',
    availableModels: [],
    scenarioCompareAvailable: false
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
let incidentMarkers = {};
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

function getIncidentRoadStyle() {
    return {
        strokeWeight: 6,
        strokeColor: '#ff6b81',
        strokeOpacity: 0.9,
        isOutline: true,
        outlineColor: '#ffd5d6',
        borderWeight: 3,
        zIndex: 88
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

    if (predictionPanelMode === 'compare') {
        compareAffectedEdgeIds.forEach(edgeId => {
            const polyline = roadPolylines.find(item => item.getExtData()?.edgeId === edgeId);
            if (polyline) polyline.setOptions(getIncidentRoadStyle());
        });
    }

    if (currentMonitorEdgeId && currentMonitorEdgeId !== 'ALL') {
        const monitorPolyline = roadPolylines.find(item => item.getExtData()?.edgeId === currentMonitorEdgeId);
        if (monitorPolyline) monitorPolyline.setOptions(getMonitorRoadStyle());
    }

    if (selectedPredictionEdgeId && selectedPredictionEdgeId !== 'ALL') {
        const targetPolyline = roadPolylines.find(item => item.getExtData()?.edgeId === selectedPredictionEdgeId);
        if (targetPolyline) targetPolyline.setOptions(getPredictionRoadStyle());
    }
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
    if (!selectedPredictionEdgeId || !edgeIds.includes(selectedPredictionEdgeId)) {
        selectedPredictionEdgeId = edgeIds[0] || null;
    }
    predictionEdgeSelectEl.value = selectedPredictionEdgeId || '';
}

function applyPredictionConfig(payload) {
    if (!payload) return;
    predictionConfigState.activeModel = payload.active_model || 'ha_baseline';
    predictionConfigState.availableModels = Array.isArray(payload.available_models) ? payload.available_models : [];
    predictionConfigState.scenarioCompareAvailable = Boolean(payload.scenario_compare_available);
    const configEdges = Array.isArray(payload.config?.observed_edges) ? payload.config.observed_edges : [];
    if (configEdges.length) {
        updatePredictionEdgeOptions(configEdges);
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
            option.textContent = `${run.run_id} (${run.incident_type})`;
            incidentRunSelectEl.appendChild(option);
        });
    }

    const incidentRun = scenarioRunState.incidentRuns.find(run => run.run_id === scenarioRunState.selectedIncidentRunId)
        || scenarioRunState.incidentRuns[0];
    scenarioRunState.selectedIncidentRunId = incidentRun ? incidentRun.run_id : '';
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

function updateRealtimePredictionPanel(prediction) {
    if (!prediction || !prediction.nodes || !prediction.nodes.length) return;
    lastRealtimePredictionData = prediction;
    updatePredictionEdgeOptions(prediction.nodes.map(node => node.edge_id));

    const node = prediction.nodes.find(item => item.edge_id === selectedPredictionEdgeId) || prediction.nodes[0];
    const horizon = prediction.horizon || [];
    const labels = horizon.map(step => `+${step}m`);
    const flow = (node.pred_flow || []).map(value => Number(value));
    const speed = (node.pred_speed || []).map(value => Number(value) * 3.6);
    const queue = (node.pred_queue || []).map(value => Number(value));
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

    document.getElementById('compareModelLabel').innerText = formatModelName(payload.model_name || predictionConfigState.activeModel);
    document.getElementById('compareIncidentType').innerText = payload.incident_type || '--';
    document.getElementById('compareAnchorStep').innerText = `${payload.anchor_step || 0}s`;
    document.getElementById('compareAffectedEdges').innerText = compareAffectedEdgeIds.length ? compareAffectedEdgeIds.join(', ') : '--';
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
        const marker = new AMap.CircleMarker({
            center: [0, 0],
            radius: 3,
            fillColor: '#ffeb3b',
            strokeOpacity: 0,
            fillOpacity: 1,
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
        const color = vClassColors[vehicle.vClass] || vClassColors.DEFAULT;
        const radius = (vehicle.vClass === 'bus' || vehicle.vClass === 'truck' || vehicle.vClass === 'emergency')
            ? 4.5
            : (vehicle.vClass === 'motorcycle' ? 2 : 3.5);
        vehicleMarkers[i].setCenter([vehicle.x, vehicle.y]);
        vehicleMarkers[i].setOptions({ fillColor: color, radius });
    }

    while (tlMarkers.length < lastTlData.length) {
        const marker = new AMap.CircleMarker({
            center: [0, 0],
            radius: 4,
            fillColor: '#aaaaaa',
            strokeOpacity: 0,
            fillOpacity: 1,
            zIndex: 105
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
        const state = (tl.state || '').toLowerCase();
        let color = '#aaaaaa';
        if (state === 'r') color = '#f56c6c';
        else if (state === 'y') color = '#e6a23c';
        else if (state === 'g') color = '#67c23a';
        tlMarkers[i].setCenter([tl.x, tl.y]);
        tlMarkers[i].setOptions({ fillColor: color });
    }

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
        if (predictionPanelMode === 'compare') {
            updateScenarioComparePanel(lastComparePayload);
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
} else {
    fetchPredictionConfig();
    fetchScenarioRuns();
}
