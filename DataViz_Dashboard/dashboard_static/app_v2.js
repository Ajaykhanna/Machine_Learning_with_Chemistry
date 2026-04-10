const PANEL_CONFIG = {
  energy: {
    key: "energy",
    metric: "energy",
    type: "line",
    canvasId: "energy-chart",
    metaId: "energy-meta",
    selection: "states",
    cardId: "panel-energy-line"
  },
  energyHist: {
    key: "energyHist",
    metric: "energy",
    type: "histogram",
    canvasId: "energy-hist-chart",
    metaId: "energy-hist-meta",
    selection: "states",
    cardId: "panel-energy-hist"
  },
  force: {
    key: "force",
    metric: "force_norm",
    type: "line",
    canvasId: "force-chart",
    metaId: "force-meta",
    selection: "states",
    cardId: "panel-force"
  },
  dipole: {
    key: "dipole",
    metric: "dipole_magnitude",
    type: "line",
    canvasId: "dipole-chart",
    metaId: "dipole-meta",
    selection: "states",
    cardId: "panel-dipole"
  },
  nacr: {
    key: "nacr",
    metric: "nacr_norm",
    type: "line",
    canvasId: "nacr-chart",
    metaId: "nacr-meta",
    selection: "pairs",
    cardId: "panel-nacr"
  },
  denacr: {
    key: "denacr",
    metric: "denacr_norm",
    type: "line",
    canvasId: "denacr-chart",
    metaId: "denacr-meta",
    selection: "pairs",
    cardId: "panel-denacr"
  }
};

const METRIC_ORDER = ["energy", "force_norm", "dipole_magnitude", "nacr_norm", "denacr_norm"];
const METRIC_AXIS_LABELS = {
  energy: "Energy",
  force_norm: "Force Norm",
  dipole_magnitude: "Dipole Magnitude",
  nacr_norm: "NACR Norm",
  denacr_norm: "dENACR Norm"
};
const STATE_METRICS = new Set(["energy", "force_norm", "dipole_magnitude"]);
const PAIR_METRICS = new Set(["nacr_norm", "denacr_norm"]);
const AXIS_DRAG_BAND = 38;
const AXIS_DRAG_CORNER_SIZE = 34;
const AXIS_SCALE_SENSITIVITY = 160;

const state = {
  meta: null,
  selectedStates: new Set(),
  selectedPairs: new Set(),
  start: 0,
  end: 1,
  histogramBins: 64,
  renderTimer: null,
  panelZooms: {},
  isSyncZoom: false,
  lastLineZoomPanelKey: null
};

document.addEventListener("DOMContentLoaded", init);

async function init() {
  setStatus("Loading dataset metadata...");
  state.meta = await fetchJSON("/api/metadata");
  state.selectedStates = new Set(state.meta.default_states.map((value) => `S${value}`));
  state.selectedPairs = new Set(state.meta.default_pairs);
  state.start = 0;
  state.end = state.meta.snapshot_count;
  state.histogramBins = 64;

  applyEnabledLayout();
  renderSummaryCards();
  mountControls();
  mountExportButtons();
  await renderAll();

  window.addEventListener("resize", debounce(() => renderAll(), 160));
}

function getEnabledMetrics() {
  return new Set(state.meta.enabled_metrics || ["energy"]);
}

function getEnabledPanelEntries() {
  const enabled = getEnabledMetrics();
  return Object.entries(PANEL_CONFIG).filter(([, panel]) => enabled.has(panel.metric));
}

function getVisibleLinePanelKeys() {
  return getEnabledPanelEntries()
    .filter(([, panel]) => panel.type === "line")
    .map(([panelKey]) => panelKey);
}

function getPanelZoomEntry(panelKey) {
  return state.panelZooms[panelKey] || {};
}

function setPanelZoomEntry(panelKey, entry) {
  const next = {};
  if (entry.start != null && entry.end != null) {
    next.start = entry.start;
    next.end = entry.end;
  }
  if (entry.xMin != null && entry.xMax != null) {
    next.xMin = entry.xMin;
    next.xMax = entry.xMax;
  }
  if (entry.yMin != null && entry.yMax != null) {
    next.yMin = entry.yMin;
    next.yMax = entry.yMax;
  }

  if (Object.keys(next).length === 0) {
    delete state.panelZooms[panelKey];
    return;
  }
  state.panelZooms[panelKey] = next;
}

function applyLineZoomState(panelKey, zoom) {
  const next = { ...getPanelZoomEntry(panelKey) };
  if (!zoom) {
    delete next.start;
    delete next.end;
    setPanelZoomEntry(panelKey, next);
    return;
  }
  next.start = zoom.start;
  next.end = zoom.end;
  setPanelZoomEntry(panelKey, next);
}

function getPreferredSyncedLineZoom() {
  const visibleLinePanels = getVisibleLinePanelKeys();
  if (
    state.lastLineZoomPanelKey &&
    visibleLinePanels.includes(state.lastLineZoomPanelKey)
  ) {
    const zoom = state.panelZooms[state.lastLineZoomPanelKey];
    if (zoom && zoom.start != null && zoom.end != null) {
      return getClampedLineZoom(zoom.start, zoom.end);
    }
  }

  for (const panelKey of visibleLinePanels) {
    const zoom = state.panelZooms[panelKey];
    if (zoom && zoom.start != null && zoom.end != null) {
      return getClampedLineZoom(zoom.start, zoom.end);
    }
  }
  return null;
}

function synchronizeVisibleLineZooms() {
  if (!state.isSyncZoom) {
    return;
  }
  const zoom = getPreferredSyncedLineZoom();
  getVisibleLinePanelKeys().forEach((panelKey) => applyLineZoomState(panelKey, zoom));
}

function setLineZoomFromInteraction(sourcePanelKey, zoom) {
  state.lastLineZoomPanelKey = sourcePanelKey;
  if (state.isSyncZoom) {
    getVisibleLinePanelKeys().forEach((panelKey) => applyLineZoomState(panelKey, zoom));
    return;
  }
  applyLineZoomState(sourcePanelKey, zoom);
}

async function renderAfterLineZoom(panelKey) {
  if (state.isSyncZoom) {
    await renderAll();
    return;
  }
  await renderPanelByKey(panelKey);
}

function updateSyncZoomButton() {
  const button = document.getElementById("sync-zoom");
  if (!button) {
    return;
  }
  button.classList.toggle("is-active", state.isSyncZoom);
  button.setAttribute("aria-pressed", String(state.isSyncZoom));
}

function getPanelKeyForMetric(metric, plotType) {
  const entry = Object.entries(PANEL_CONFIG).find(([, panel]) => panel.metric === metric && panel.type === plotType);
  return entry ? entry[0] : null;
}

function getLineYDomain(panelKey, rawDomain) {
  const zoom = getPanelZoomEntry(panelKey);
  if (zoom.yMin == null || zoom.yMax == null) {
    return rawDomain;
  }
  const normalized = normalizeUnboundedDomain([zoom.yMin, zoom.yMax], rawDomain);
  return normalized || rawDomain;
}

function getHistogramYDomain(panelKey, rawDomain) {
  const zoom = getPanelZoomEntry(panelKey);
  if (zoom.yMin == null || zoom.yMax == null) {
    return rawDomain;
  }
  const normalized = normalizeHistogramYDomain([zoom.yMin, zoom.yMax], rawDomain);
  return normalized || rawDomain;
}

function clearPanelXZoom(panelKey) {
  const next = { ...getPanelZoomEntry(panelKey) };
  delete next.start;
  delete next.end;
  delete next.xMin;
  delete next.xMax;
  setPanelZoomEntry(panelKey, next);
}

function clearPanelYZoom(panelKey) {
  const next = { ...getPanelZoomEntry(panelKey) };
  delete next.yMin;
  delete next.yMax;
  setPanelZoomEntry(panelKey, next);
}

function clearPanelView(panelKey, syncLineX = false) {
  const panel = PANEL_CONFIG[panelKey];
  if (!panel) {
    return;
  }

  if (panel.type === "line" && syncLineX) {
    getVisibleLinePanelKeys().forEach((visiblePanelKey) => clearPanelXZoom(visiblePanelKey));
  } else {
    clearPanelXZoom(panelKey);
  }
  clearPanelYZoom(panelKey);
}

function getPanelExportLimits(panelKey) {
  if (!panelKey) {
    return {};
  }
  const panel = PANEL_CONFIG[panelKey];
  const zoom = getPanelZoomEntry(panelKey);
  const limits = {};
  if (zoom.yMin != null && zoom.yMax != null) {
    limits.y_min = zoom.yMin;
    limits.y_max = zoom.yMax;
  }
  if (panel && panel.type === "histogram" && zoom.xMin != null && zoom.xMax != null) {
    limits.x_min = zoom.xMin;
    limits.x_max = zoom.xMax;
  }
  return limits;
}

function applyEnabledLayout() {
  const enabled = getEnabledMetrics();
  Object.values(PANEL_CONFIG).forEach((panel) => {
    const card = document.getElementById(panel.cardId);
    if (card) {
      card.hidden = !enabled.has(panel.metric);
    }
  });

  const hasStatePanels = Array.from(enabled).some((metric) => STATE_METRICS.has(metric));
  const hasPairPanels = Array.from(enabled).some((metric) => PAIR_METRICS.has(metric));

  const stateCard = document.getElementById("state-selection-card");
  const pairCard = document.getElementById("coupling-selection-card");
  const histogramControl = document.getElementById("histogram-control");

  if (stateCard) {
    stateCard.hidden = !hasStatePanels;
  }
  if (pairCard) {
    pairCard.hidden = !hasPairPanels;
  }
  if (histogramControl) {
    histogramControl.hidden = !enabled.has("energy");
  }
}

function mountControls() {
  const startSlider = document.getElementById("range-start");
  const endSlider = document.getElementById("range-end");
  const maxSnapshot = state.meta.snapshot_count;

  startSlider.max = String(maxSnapshot - 1);
  endSlider.max = String(maxSnapshot);
  startSlider.value = String(state.start);
  endSlider.value = String(state.end);

  startSlider.addEventListener("input", () => {
    const value = Number(startSlider.value);
    state.start = Math.min(value, state.end - 1);
    startSlider.value = String(state.start);
    updateRangeLabel();
    scheduleRender();
  });

  endSlider.addEventListener("input", () => {
    const value = Number(endSlider.value);
    state.end = Math.max(value, state.start + 1);
    endSlider.value = String(state.end);
    updateRangeLabel();
    scheduleRender();
  });

  const binsSlider = document.getElementById("histogram-bins");
  const binsValue = document.getElementById("histogram-bins-value");
  binsSlider.value = String(state.histogramBins);
  binsValue.textContent = String(state.histogramBins);
  binsSlider.addEventListener("input", () => {
    state.histogramBins = Number(binsSlider.value);
    binsValue.textContent = binsSlider.value;
    scheduleRender();
  });

  document.getElementById("reset-range").addEventListener("click", () => {
    state.start = 0;
    state.end = state.meta.snapshot_count;
    state.panelZooms = {};
    state.lastLineZoomPanelKey = null;
    startSlider.value = "0";
    endSlider.value = String(state.end);
    updateRangeLabel();
    renderAll();
  });

  document.getElementById("sync-zoom").addEventListener("click", () => {
    state.isSyncZoom = !state.isSyncZoom;
    if (state.isSyncZoom) {
      synchronizeVisibleLineZooms();
    }
    updateSyncZoomButton();
    renderAll();
  });

  document.getElementById("reset-zoom").addEventListener("click", () => {
    state.panelZooms = {};
    state.lastLineZoomPanelKey = null;
    renderAll();
  });

  document.getElementById("states-default").addEventListener("click", () => {
    state.selectedStates = new Set(state.meta.default_states.map((value) => `S${value}`));
    renderStateChips();
    renderAll();
  });

  document.getElementById("states-all").addEventListener("click", () => {
    state.selectedStates = new Set(state.meta.state_labels);
    renderStateChips();
    renderAll();
  });

  document.getElementById("pairs-default").addEventListener("click", () => {
    state.selectedPairs = new Set(state.meta.default_pairs);
    renderPairChips();
    renderAll();
  });

  document.getElementById("pairs-all").addEventListener("click", () => {
    state.selectedPairs = new Set(state.meta.pair_labels);
    renderPairChips();
    renderAll();
  });

  document.getElementById("pair-filter").addEventListener("input", () => renderPairChips());

  updateRangeLabel();
  updateSyncZoomButton();
  renderStateChips();
  renderPairChips();
}

function mountExportButtons() {
  const allowedFormats = new Set(state.meta.export_formats || ["png", "pdf"]);
  document.querySelectorAll(".export-button").forEach((button) => {
    if (!allowedFormats.has(button.dataset.format)) {
      button.hidden = true;
      return;
    }
    button.addEventListener("click", async () => {
      const metric = button.dataset.metric;
      const plotType = button.dataset.plotType;
      const format = button.dataset.format;
      const ids = metric === "nacr_norm" || metric === "denacr_norm"
        ? Array.from(state.selectedPairs)
        : Array.from(state.selectedStates);

      if (!ids.length) {
        setStatus("Select at least one series before exporting.", true);
        return;
      }

      const exportWindow = plotType === "histogram"
        ? { start: state.start, end: state.end }
        : getLineWindowByMetric(metric);
      const exportLimits = getPanelExportLimits(getPanelKeyForMetric(metric, plotType));

      setStatus(`Exporting ${metric} as ${format.toUpperCase()}...`);
      const response = await fetchJSON("/api/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          metric,
          plot_type: plotType,
          format,
          ids,
          start: exportWindow.start,
          end: exportWindow.end,
          bins: state.histogramBins,
          ...exportLimits
        })
      });

      setStatus(`Saved ${response.filename}. <a href="/exports/${response.filename}" target="_blank" rel="noopener">Open export</a>`);
    });
  });
}

function renderSummaryCards() {
  const container = document.getElementById("summary-cards");
  const cards = [
    { label: "Snapshots", value: formatSnapshotCount(state.meta.snapshot_count) },
    { label: "States", value: summarizeStateLabels(state.meta.state_labels) },
    { label: "NAC Pairs", value: formatInt(state.meta.pair_labels.length) }
  ];

  container.innerHTML = cards
    .map((card) => `
      <div class="stat-card">
        <div class="stat-label">${card.label}</div>
        <div class="stat-value">${card.value}</div>
      </div>
    `)
    .join("");
}

function renderStateChips() {
  const container = document.getElementById("state-chips");
  container.innerHTML = "";
  state.meta.state_labels.forEach((label) => {
    container.appendChild(createChip(label, state.selectedStates.has(label), state.meta.colors.states[label], () => {
      toggleSelection(state.selectedStates, label);
      renderStateChips();
      scheduleRender();
    }));
  });
}

function renderPairChips() {
  const filterValue = document.getElementById("pair-filter").value.trim().toLowerCase();
  const container = document.getElementById("pair-chips");
  container.innerHTML = "";
  state.meta.pair_labels
    .filter((label) => !filterValue || label.toLowerCase().includes(filterValue))
    .forEach((label) => {
      container.appendChild(createChip(label, state.selectedPairs.has(label), state.meta.colors.pairs[label], () => {
        toggleSelection(state.selectedPairs, label);
        renderPairChips();
        scheduleRender();
      }));
    });
}

function createChip(label, isSelected, color, onClick) {
  const button = document.createElement("button");
  button.className = `chip${isSelected ? " is-selected" : ""}`;
  button.style.setProperty("--chip-color", color);
  button.textContent = label;
  button.addEventListener("click", onClick);
  return button;
}

function updateRangeLabel() {
  const label = document.getElementById("range-label");
  const windowSize = state.end - state.start;
  label.textContent = `Snapshots ${formatInt(state.start)} to ${formatInt(state.end - 1)} | window size ${formatInt(windowSize)}`;
}

function scheduleRender() {
  if (state.renderTimer) {
    clearTimeout(state.renderTimer);
  }
  state.renderTimer = setTimeout(() => renderAll(), 120);
}

async function renderAll() {
  clampPanelZooms();
  setStatus("Refreshing plots...");
  const jobs = getEnabledPanelEntries().map(([panelKey, panel]) => renderPanel(panelKey, panel));
  jobs.push(renderStatisticsPanel());
  await Promise.all(jobs);
  setStatus(
    `Showing snapshots <strong>${formatInt(state.start)}</strong> to <strong>${formatInt(state.end - 1)}</strong> with ` +
    `<strong>${state.selectedStates.size}</strong> states and <strong>${state.selectedPairs.size}</strong> excited-state pairs selected. ` +
    `Use the mouse wheel to zoom, drag across a region to crop, drag an axis to scale one panel, double-click the plot body to reset a panel, or use Reset Zoom for all plots.` +
    (state.isSyncZoom ? ` <strong>Sync Zoom</strong> is active for all visible snapshot-index plots.` : "")
  );
}

async function renderPanelByKey(panelKey) {
  const panel = PANEL_CONFIG[panelKey];
  if (!panel || !getEnabledMetrics().has(panel.metric)) {
    return;
  }
  await renderPanel(panelKey, panel);
}

async function renderPanel(panelKey, panel) {
  const canvas = document.getElementById(panel.canvasId);
  const width = Math.max(640, Math.round(canvas.getBoundingClientRect().width * (window.devicePixelRatio || 1)));
  const ids = panel.selection === "states" ? Array.from(state.selectedStates) : Array.from(state.selectedPairs);
  const meta = document.getElementById(panel.metaId);

  if (!ids.length) {
    drawEmptyPanel(canvas, "No series selected");
    meta.textContent = "Choose at least one legend item to render this panel.";
    return;
  }

  if (panel.type === "histogram") {
    const payload = await fetchJSON(
      `/api/histogram?ids=${encodeURIComponent(ids.join(","))}&start=${state.start}&end=${state.end}&bins=${state.histogramBins}`
    );
    drawHistogramPanel(panel, payload);
    meta.textContent = buildHistogramMetaText(panelKey, payload);
    return;
  }

  const lineWindow = getLineWindow(panelKey);
  const payload = await fetchJSON(
    `/api/series?metric=${panel.metric}&ids=${encodeURIComponent(ids.join(","))}` +
    `&start=${lineWindow.start}&end=${lineWindow.end}&width=${width}`
  );
  drawLinePanel(panel, payload);
  meta.textContent = buildLineMetaText(panelKey, payload);
}

async function renderStatisticsPanel() {
  const payload = await fetchJSON(
    `/api/statistics?start=${state.start}&end=${state.end}` +
    `&states=${encodeURIComponent(Array.from(state.selectedStates).join(","))}` +
    `&pairs=${encodeURIComponent(Array.from(state.selectedPairs).join(","))}`
  );
  drawStatisticsChart(payload);
  renderStatisticsGrid(payload);
  document.getElementById("stats-meta").textContent =
    "Bars are normalized within each metric using aggregate mean, std, and variance.";
}

function drawEmptyPanel(canvas, message) {
  const { ctx, width, height } = prepareCanvas(canvas);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = "#5e6259";
  ctx.font = '600 18px "Segoe UI Variable Text", "Trebuchet MS", sans-serif';
  ctx.textAlign = "center";
  ctx.fillText(message, width / 2, height / 2);
}

function drawLinePanel(panel, payload) {
  const canvas = document.getElementById(panel.canvasId);
  const tooltip = document.getElementById(`${panel.canvasId.replace("-chart", "")}-tooltip`);
  const crosshair = document.getElementById(`${panel.canvasId.replace("-chart", "")}-crosshair`);
  const selectionOverlay = getOrCreateSelection(canvas);
  const { ctx, width, height } = prepareCanvas(canvas);
  const margins = { top: 26, right: 20, bottom: 48, left: 82 };
  const plotWidth = width - margins.left - margins.right;
  const plotHeight = height - margins.top - margins.bottom;
  const rawYDomain = computeLineYDomain(payload.series);
  const yDomain = getLineYDomain(panel.key, rawYDomain);
  const xScale = d3.scaleLinear().domain([payload.start, payload.end - 1]).range([margins.left, margins.left + plotWidth]);
  const yScale = d3.scaleLinear().domain(yDomain).range([margins.top + plotHeight, margins.top]);

  ctx.clearRect(0, 0, width, height);
  drawPlotBackground(ctx, width, height, margins, plotWidth, plotHeight);
  drawAxes(
    ctx,
    xScale,
    yScale,
    margins,
    plotWidth,
    plotHeight,
    withUnit(payload.axis_label || axisLabelForMetric(panel.metric), payload.units),
    "Snapshot index"
  );

  ctx.save();
  ctx.beginPath();
  ctx.rect(margins.left, margins.top, plotWidth, plotHeight);
  ctx.clip();

  payload.series.forEach((series) => {
    ctx.beginPath();
    ctx.lineWidth = 1.35;
    ctx.strokeStyle = series.color;
    series.x.forEach((xValue, index) => {
      const cx = xScale(xValue);
      const cy = yScale(series.y[index]);
      if (index === 0) {
        ctx.moveTo(cx, cy);
      } else {
        ctx.lineTo(cx, cy);
      }
    });
    ctx.stroke();
  });
  ctx.restore();

  canvas.__chartModel = {
    kind: "line",
    panelKey: panel.key,
    payload,
    unitLabel: formatBracketUnit(payload.units),
    xScale,
    yScale,
    rawYDomain,
    margins,
    plotWidth,
    plotHeight,
    width,
    height,
    tooltip,
    crosshair,
    selectionOverlay
  };
  canvas.style.cursor = "crosshair";
  bindChartInteractions(canvas);
}

function drawHistogramPanel(panel, payload) {
  const canvas = document.getElementById(panel.canvasId);
  const tooltip = document.getElementById("energy-hist-tooltip");
  const crosshair = document.getElementById("energy-hist-crosshair");
  const selectionOverlay = getOrCreateSelection(canvas);
  const { ctx, width, height } = prepareCanvas(canvas);
  const margins = { top: 26, right: 20, bottom: 48, left: 72 };
  const plotWidth = width - margins.left - margins.right;
  const plotHeight = height - margins.top - margins.bottom;
  const xMin = Math.min(...payload.series.map((series) => series.edges[0]));
  const xMax = Math.max(...payload.series.map((series) => series.edges[series.edges.length - 1]));
  const xDomain = getHistogramZoomDomain(panel.key, xMin, xMax);
  const yMax = computeHistogramVisibleYMax(payload.series, xDomain);
  const rawYDomain = [0, yMax * 1.08 || 1];
  const yDomain = getHistogramYDomain(panel.key, rawYDomain);
  const xScale = d3.scaleLinear().domain(xDomain).range([margins.left, margins.left + plotWidth]);
  const yScale = d3.scaleLinear().domain(yDomain).range([margins.top + plotHeight, margins.top]);

  ctx.clearRect(0, 0, width, height);
  drawPlotBackground(ctx, width, height, margins, plotWidth, plotHeight);
  drawAxes(
    ctx,
    xScale,
    yScale,
    margins,
    plotWidth,
    plotHeight,
    "Count",
    withUnit(payload.axis_label || axisLabelForMetric("energy"), state.meta.units.energy)
  );

  ctx.save();
  ctx.beginPath();
  ctx.rect(margins.left, margins.top, plotWidth, plotHeight);
  ctx.clip();

  payload.series.forEach((series) => {
    ctx.beginPath();
    ctx.lineWidth = 1.4;
    ctx.strokeStyle = series.color;
    for (let index = 0; index < series.counts.length; index += 1) {
      const left = series.edges[index];
      const right = series.edges[index + 1];
      if (right < xDomain[0] || left > xDomain[1]) {
        continue;
      }
      const yValue = series.counts[index];
      const x0 = xScale(Math.max(left, xDomain[0]));
      const x1 = xScale(Math.min(right, xDomain[1]));
      const y = yScale(yValue);
      if (index === 0 || left <= xDomain[0]) {
        ctx.moveTo(x0, yScale(0));
        ctx.lineTo(x0, y);
      } else {
        ctx.lineTo(x0, y);
      }
      ctx.lineTo(x1, y);
    }
    ctx.stroke();
  });
  ctx.restore();

  canvas.__chartModel = {
    kind: "histogram",
    panelKey: panel.key,
    payload,
    xScale,
    yScale,
    margins,
    plotWidth,
    plotHeight,
    tooltip,
    crosshair,
    selectionOverlay,
    rawXDomain: [xMin, xMax],
    rawYDomain,
    width,
    height
  };
  canvas.style.cursor = "crosshair";
  bindChartInteractions(canvas);
}

function drawStatisticsChart(payload) {
  const canvas = document.getElementById("stats-chart");
  const metrics = METRIC_ORDER
    .map((metric) => ({ key: metric, ...payload.metrics[metric] }))
    .filter((metric) => metric && metric.selected_series_count > 0);

  if (!metrics.length) {
    drawEmptyPanel(canvas, "No statistics available");
    return;
  }

  const { ctx, width, height } = prepareCanvas(canvas);
  const margins = { top: 18, right: 18, bottom: 14, left: 118 };
  const plotWidth = width - margins.left - margins.right;
  const plotHeight = height - margins.top - margins.bottom;
  const rowHeight = plotHeight / metrics.length;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  ctx.textBaseline = "middle";

  metrics.forEach((metric, metricIndex) => {
    const yBase = margins.top + metricIndex * rowHeight;
    const barHeight = 12;
    const gap = 8;
    const values = [
      { magnitude: Math.abs(metric.aggregate.mean), color: "#b55333" },
      { magnitude: metric.aggregate.std, color: "#1768ac" },
      { magnitude: metric.aggregate.variance, color: "#1b998b" }
    ];
    const maxMagnitude = Math.max(...values.map((value) => value.magnitude), 1e-12);

    ctx.fillStyle = "#242a30";
    ctx.font = '600 12px "Segoe UI Variable Text", "Trebuchet MS", sans-serif';
    ctx.fillText(compactMetricTitle(metric.title), 10, yBase + rowHeight / 2);

    values.forEach((value, index) => {
      const y = yBase + 10 + index * (barHeight + gap);
      const barWidth = (value.magnitude / maxMagnitude) * (plotWidth - 8);
      ctx.fillStyle = "rgba(54, 47, 34, 0.08)";
      ctx.fillRect(margins.left, y, plotWidth, barHeight);
      ctx.fillStyle = value.color;
      ctx.fillRect(margins.left, y, barWidth, barHeight);
    });
  });
}

function renderStatisticsGrid(payload) {
  const container = document.getElementById("stats-grid");
  container.innerHTML = "";

  METRIC_ORDER.forEach((metricKey) => {
    const metric = payload.metrics[metricKey];
    const card = document.createElement("div");
    card.className = "stats-item";

    if (!metric || metric.selected_series_count === 0) {
      card.innerHTML = `
        <div class="stats-item-head">
          <div class="stats-item-title">${metric ? metric.title : metricKey}</div>
          <div class="stats-item-unit">No active series</div>
        </div>
      `;
      container.appendChild(card);
      return;
    }

    const agg = metric.aggregate;
    card.innerHTML = `
      <div class="stats-item-head">
        <div class="stats-item-title">${metric.title}</div>
        <div class="stats-item-unit">${formatBracketUnit(metric.units)}</div>
      </div>
      <div class="stats-item-grid">
        ${renderStatPair("Mean", formatStatistic(agg.mean))}
        ${renderStatPair("Std", formatStatistic(agg.std))}
        ${renderStatPair("Variance", formatStatistic(agg.variance))}
        ${renderStatPair("Min", formatStatistic(agg.min))}
        ${renderStatPair("Max", formatStatistic(agg.max))}
        ${renderStatPair("Series", formatInt(metric.selected_series_count))}
      </div>
    `;
    container.appendChild(card);
  });
}

function renderStatPair(label, value) {
  return `
    <div class="stats-pair">
      <div class="stats-pair-label">${label}</div>
      <div class="stats-pair-value">${value}</div>
    </div>
  `;
}

function prepareCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const bounds = canvas.getBoundingClientRect();
  const width = Math.max(320, Math.round(bounds.width || canvas.parentElement.clientWidth || 720));
  const height = Math.max(280, Math.round(bounds.height || 360));
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  return { ctx, width, height };
}

function drawPlotBackground(ctx, width, height, margins, plotWidth, plotHeight) {
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(margins.left, margins.top, plotWidth, plotHeight);
}

function drawAxes(ctx, xScale, yScale, margins, plotWidth, plotHeight, yLabel, xLabel) {
  const xTicks = d3.ticks(xScale.domain()[0], xScale.domain()[1], 6);
  const yTicks = d3.ticks(yScale.domain()[0], yScale.domain()[1], 5);
  const axisBottom = margins.top + plotHeight;

  ctx.strokeStyle = "rgba(36, 42, 48, 0.58)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(margins.left, margins.top);
  ctx.lineTo(margins.left, axisBottom);
  ctx.lineTo(margins.left + plotWidth, axisBottom);
  ctx.stroke();

  ctx.fillStyle = "#5e6259";
  ctx.font = '12px "Segoe UI Variable Text", "Trebuchet MS", sans-serif';

  yTicks.forEach((tick) => {
    const y = yScale(tick);
    ctx.beginPath();
    ctx.moveTo(margins.left - 6, y);
    ctx.lineTo(margins.left, y);
    ctx.stroke();
    ctx.textAlign = "right";
    ctx.fillText(formatNumber(tick), margins.left - 10, y + 4);
  });

  xTicks.forEach((tick) => {
    const x = xScale(tick);
    ctx.beginPath();
    ctx.moveTo(x, axisBottom);
    ctx.lineTo(x, axisBottom + 6);
    ctx.stroke();
    ctx.textAlign = "center";
    ctx.fillText(formatNumber(tick), x, axisBottom + 22);
  });

  ctx.fillStyle = "#242a30";
  ctx.font = '600 12px "Segoe UI Variable Text", "Trebuchet MS", sans-serif';
  ctx.textAlign = "right";
  ctx.fillText(xLabel, margins.left + plotWidth, axisBottom + 40);

  ctx.save();
  ctx.translate(18, margins.top + plotHeight / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillText(yLabel, 0, 0);
  ctx.restore();
}

function computeLineYDomain(seriesList) {
  let min = Infinity;
  let max = -Infinity;
  seriesList.forEach((series) => {
    series.y.forEach((value) => {
      if (value < min) min = value;
      if (value > max) max = value;
    });
  });

  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return [0, 1];
  }
  if (min === max) {
    const pad = Math.abs(min || 1) * 0.1;
    return [min - pad, max + pad];
  }
  const pad = (max - min) * 0.08;
  return [min - pad, max + pad];
}

function computeHistogramVisibleYMax(seriesList, xDomain) {
  let maxCount = 0;
  seriesList.forEach((series) => {
    for (let index = 0; index < series.counts.length; index += 1) {
      const left = series.edges[index];
      const right = series.edges[index + 1];
      if (right < xDomain[0] || left > xDomain[1]) {
        continue;
      }
      maxCount = Math.max(maxCount, series.counts[index]);
    }
  });
  return maxCount || 1;
}

function bindChartInteractions(canvas) {
  if (canvas.dataset.hoverBound === "true") {
    return;
  }

  const interaction = {
    dragging: false,
    mode: null,
    zone: "none",
    startX: 0,
    currentX: 0,
    startY: 0,
    currentY: 0,
    anchorValue: null,
    initialDomain: null,
    selection: getOrCreateSelection(canvas)
  };
  canvas.__interaction = interaction;

  canvas.addEventListener("mousemove", (event) => {
    const model = canvas.__chartModel;
    if (!model) return;

    const [mx, my] = d3.pointer(event, canvas);
    if (interaction.dragging) {
      updateCanvasCursor(canvas, model, interaction.mode);
      if (interaction.mode === "crop") {
        updateSelectionBox(model, interaction.startX, clamp(mx, model.margins.left, model.margins.left + model.plotWidth));
      }
      hideTooltip(model);
      return;
    }

    const zone = getPointerZone(model, mx, my);
    updateCanvasCursor(canvas, model, zone);

    if (zone !== "plot") {
      hideTooltip(model);
      return;
    }

    if (model.kind === "line") {
      handleLineHover(model, mx, event);
    } else {
      handleHistogramHover(model, mx, event);
    }
  });

  canvas.addEventListener("mousedown", (event) => {
    if (event.button !== 0) {
      return;
    }
    const model = canvas.__chartModel;
    if (!model) {
      return;
    }

    const [mx, my] = d3.pointer(event, canvas);
    const zone = getPointerZone(model, mx, my);
    if (zone === "none") {
      return;
    }

    interaction.zone = zone;
    interaction.dragging = true;
    interaction.mode = zone === "plot" ? "crop" : (zone === "x-axis" ? "scale-x" : "scale-y");
    interaction.startX = mx;
    interaction.currentX = mx;
    interaction.startY = my;
    interaction.currentY = my;
    interaction.initialDomain = interaction.mode === "scale-y"
      ? model.yScale.domain().slice()
      : (model.kind === "line" ? [model.payload.start, model.payload.end] : model.xScale.domain().slice());
    interaction.anchorValue = interaction.mode === "scale-y"
      ? model.yScale.invert(clamp(my, model.margins.top, model.margins.top + model.plotHeight))
      : model.xScale.invert(clamp(mx, model.margins.left, model.margins.left + model.plotWidth));

    if (interaction.mode === "crop") {
      interaction.startX = clamp(mx, model.margins.left, model.margins.left + model.plotWidth);
      interaction.currentX = interaction.startX;
      updateSelectionBox(model, interaction.startX, interaction.currentX);
    } else {
      hideSelectionBox(interaction);
    }
    hideTooltip(model);
    updateCanvasCursor(canvas, model, interaction.mode);
    event.preventDefault();
  });

  window.addEventListener("mousemove", (event) => {
    if (!interaction.dragging) {
      return;
    }
    const model = canvas.__chartModel;
    if (!model) {
      return;
    }

    const rect = canvas.getBoundingClientRect();
    interaction.currentX = event.clientX - rect.left;
    interaction.currentY = event.clientY - rect.top;
    if (interaction.mode === "crop") {
      interaction.currentX = clamp(interaction.currentX, model.margins.left, model.margins.left + model.plotWidth);
      updateSelectionBox(model, interaction.startX, interaction.currentX);
    }
  });

  window.addEventListener("mouseup", async () => {
    if (!interaction.dragging) {
      return;
    }
    const model = canvas.__chartModel;
    const width = Math.abs(interaction.currentX - interaction.startX);
    const height = Math.abs(interaction.currentY - interaction.startY);
    const mode = interaction.mode;
    interaction.dragging = false;
    interaction.mode = null;
    interaction.zone = "none";
    hideSelectionBox(interaction);
    updateCanvasCursor(canvas, model, "none");
    if (!model) {
      return;
    }
    if (mode === "crop") {
      if (width < 12) {
        return;
      }
      await applyBrushZoom(model, interaction.startX, interaction.currentX);
      return;
    }
    if (mode === "scale-x") {
      if (width < 6) {
        return;
      }
      await applyAxisScale(model, "x", interaction);
      return;
    }
    if (mode === "scale-y") {
      if (height < 6) {
        return;
      }
      await applyAxisScale(model, "y", interaction);
    }
  });

  canvas.addEventListener("mouseleave", () => {
    const model = canvas.__chartModel;
    if (model && !interaction.dragging) {
      hideTooltip(model);
    }
    if (!interaction.dragging) {
      canvas.style.cursor = "default";
    }
  });

  canvas.addEventListener("dblclick", async (event) => {
    const model = canvas.__chartModel;
    if (!model) {
      return;
    }
    const [mx, my] = d3.pointer(event, canvas);
    const zone = getPointerZone(model, mx, my);

    if (zone === "x-axis") {
      await resetPanelXAxis(model);
      return;
    }
    if (zone === "y-axis") {
      clearPanelYZoom(model.panelKey);
      await renderPanelByKey(model.panelKey);
      return;
    }
    if (zone !== "plot") {
      return;
    }

    if (model.kind === "line") {
      clearPanelView(model.panelKey, state.isSyncZoom);
      await renderAfterLineZoom(model.panelKey);
      return;
    }
    clearPanelView(model.panelKey, false);
    await renderPanelByKey(model.panelKey);
  });

  canvas.addEventListener("wheel", async (event) => {
    const model = canvas.__chartModel;
    if (!model) {
      return;
    }
    const [mx, my] = d3.pointer(event, canvas);
    if (getPointerZone(model, mx, my) !== "plot") {
      return;
    }
    event.preventDefault();
    if (model.kind === "line") {
      await applyLineWheelZoom(model, mx, event.deltaY);
    } else {
      await applyHistogramWheelZoom(model, mx, event.deltaY);
    }
  }, { passive: false });

  canvas.dataset.hoverBound = "true";
}

function getPointerZone(model, mx, my) {
  const axisBottom = model.margins.top + model.plotHeight;
  const axisRight = model.margins.left + model.plotWidth;
  const xAxisTop = axisBottom;
  const xAxisBottom = Math.min(model.height, axisBottom + AXIS_DRAG_BAND);
  const yAxisLeft = Math.max(0, model.margins.left - AXIS_DRAG_BAND);
  const yAxisRight = model.margins.left;

  if (isInsidePlot(model, mx, my)) {
    return "plot";
  }
  if (
    mx >= yAxisLeft &&
    mx <= yAxisRight &&
    my >= xAxisTop &&
    my <= Math.min(model.height, xAxisBottom + AXIS_DRAG_CORNER_SIZE)
  ) {
    return "none";
  }
  if (mx >= model.margins.left + AXIS_DRAG_CORNER_SIZE && mx <= axisRight && my >= xAxisTop && my <= xAxisBottom) {
    return "x-axis";
  }
  if (mx >= yAxisLeft && mx <= yAxisRight && my >= model.margins.top && my <= axisBottom - AXIS_DRAG_CORNER_SIZE) {
    return "y-axis";
  }
  return "none";
}

function updateCanvasCursor(canvas, model, zone) {
  if (!model) {
    canvas.style.cursor = "default";
    return;
  }
  if (zone === "scale-x" || zone === "x-axis") {
    canvas.style.cursor = "ew-resize";
    return;
  }
  if (zone === "scale-y" || zone === "y-axis") {
    canvas.style.cursor = "ns-resize";
    return;
  }
  canvas.style.cursor = zone === "plot" || zone === "crop" ? "crosshair" : "default";
}

function isInsidePlot(model, mx, my) {
  return (
    mx >= model.margins.left &&
    mx <= model.margins.left + model.plotWidth &&
    my >= model.margins.top &&
    my <= model.margins.top + model.plotHeight
  );
}

function getOrCreateSelection(canvas) {
  let selection = canvas.parentElement.querySelector(".chart-selection");
  if (!selection) {
    selection = document.createElement("div");
    selection.className = "chart-selection";
    selection.hidden = true;
    canvas.parentElement.appendChild(selection);
  }
  return selection;
}

function updateSelectionBox(model, x0, x1) {
  const left = Math.min(x0, x1);
  const width = Math.max(1, Math.abs(x1 - x0));
  model.selectionOverlay.hidden = false;
  model.selectionOverlay.style.left = `${left}px`;
  model.selectionOverlay.style.top = `${model.margins.top}px`;
  model.selectionOverlay.style.width = `${width}px`;
  model.selectionOverlay.style.height = `${model.plotHeight}px`;
}

function hideSelectionBox(interaction) {
  interaction.selection.hidden = true;
  interaction.selection.style.width = "0px";
}

async function resetPanelXAxis(model) {
  if (model.kind === "line") {
    if (state.isSyncZoom) {
      setLineZoomFromInteraction(model.panelKey, null);
    } else {
      applyLineZoomState(model.panelKey, null);
    }
    await renderAfterLineZoom(model.panelKey);
    return;
  }
  clearPanelXZoom(model.panelKey);
  await renderPanelByKey(model.panelKey);
}

async function applyAxisScale(model, axis, interaction) {
  if (axis === "x") {
    if (model.kind === "line") {
      await applyLineAxisScale(model, interaction);
      return;
    }
    await applyHistogramAxisScale(model, interaction);
    return;
  }
  await applyYAxisScale(model, interaction);
}

async function applyBrushZoom(model, x0, x1) {
  const left = Math.min(x0, x1);
  const right = Math.max(x0, x1);

  if (model.kind === "line") {
    const zoom = getClampedLineZoom(
      Math.floor(model.xScale.invert(left)),
      Math.ceil(model.xScale.invert(right)) + 1
    );
    setLineZoomFromInteraction(model.panelKey, zoom);
  } else {
    const nextMin = Math.max(model.rawXDomain[0], Math.min(model.xScale.invert(left), model.xScale.invert(right)));
    const nextMax = Math.min(model.rawXDomain[1], Math.max(model.xScale.invert(left), model.xScale.invert(right)));
    if (nextMax - nextMin <= 1e-9) {
      clearPanelXZoom(model.panelKey);
    } else {
      setPanelZoomEntry(model.panelKey, {
        ...getPanelZoomEntry(model.panelKey),
        xMin: nextMin,
        xMax: nextMax,
      });
    }
  }

  if (model.kind === "line") {
    await renderAfterLineZoom(model.panelKey);
    return;
  }
  await renderPanelByKey(model.panelKey);
}

async function applyLineWheelZoom(model, mx, deltaY) {
  const currentStart = model.payload.start;
  const currentEnd = model.payload.end;
  const currentSpan = Math.max(2, currentEnd - currentStart);
  const totalSpan = Math.max(2, state.end - state.start);
  const factor = deltaY < 0 ? 0.8 : 1.25;
  const nextSpan = clamp(Math.round(currentSpan * factor), 8, totalSpan);

  if (nextSpan >= totalSpan - 1) {
    setLineZoomFromInteraction(model.panelKey, null);
    await renderAfterLineZoom(model.panelKey);
    return;
  }

  const center = model.xScale.invert(mx);
  const ratio = (center - currentStart) / currentSpan;
  let nextStart = Math.round(center - ratio * nextSpan);
  let nextEnd = nextStart + nextSpan;

  if (nextStart < state.start) {
    nextStart = state.start;
    nextEnd = nextStart + nextSpan;
  }
  if (nextEnd > state.end) {
    nextEnd = state.end;
    nextStart = nextEnd - nextSpan;
  }

  setLineZoomFromInteraction(model.panelKey, { start: nextStart, end: nextEnd });
  await renderAfterLineZoom(model.panelKey);
}

async function applyHistogramWheelZoom(model, mx, deltaY) {
  const currentDomain = model.xScale.domain();
  const rawDomain = model.rawXDomain;
  const currentSpan = currentDomain[1] - currentDomain[0];
  const rawSpan = rawDomain[1] - rawDomain[0];
  const factor = deltaY < 0 ? 0.82 : 1.22;
  const nextSpan = clamp(currentSpan * factor, rawSpan * 0.01, rawSpan);

  if (nextSpan >= rawSpan * 0.995) {
    clearPanelXZoom(model.panelKey);
    await renderPanelByKey(model.panelKey);
    return;
  }

  const center = model.xScale.invert(mx);
  const ratio = (center - currentDomain[0]) / currentSpan;
  let nextMin = center - ratio * nextSpan;
  let nextMax = nextMin + nextSpan;

  if (nextMin < rawDomain[0]) {
    nextMin = rawDomain[0];
    nextMax = nextMin + nextSpan;
  }
  if (nextMax > rawDomain[1]) {
    nextMax = rawDomain[1];
    nextMin = nextMax - nextSpan;
  }

  setPanelZoomEntry(model.panelKey, {
    ...getPanelZoomEntry(model.panelKey),
    xMin: nextMin,
    xMax: nextMax,
  });
  await renderPanelByKey(model.panelKey);
}

async function applyLineAxisScale(model, interaction) {
  const deltaX = interaction.currentX - interaction.startX;
  const factor = getAxisScaleFactor(deltaX, "x");
  const domain = scaleDomainAroundAnchor(interaction.initialDomain, interaction.anchorValue, factor);
  const zoom = getClampedLineZoom(domain[0], domain[1]);
  setLineZoomFromInteraction(model.panelKey, zoom);
  await renderAfterLineZoom(model.panelKey);
}

async function applyHistogramAxisScale(model, interaction) {
  const deltaX = interaction.currentX - interaction.startX;
  const factor = getAxisScaleFactor(deltaX, "x");
  const domain = scaleDomainAroundAnchor(interaction.initialDomain, interaction.anchorValue, factor);
  const nextDomain = normalizeBoundedDomain(domain, model.rawXDomain, model.rawXDomain);
  const next = { ...getPanelZoomEntry(model.panelKey) };
  if (!nextDomain) {
    delete next.xMin;
    delete next.xMax;
  } else {
    next.xMin = nextDomain[0];
    next.xMax = nextDomain[1];
  }
  setPanelZoomEntry(model.panelKey, next);
  await renderPanelByKey(model.panelKey);
}

async function applyYAxisScale(model, interaction) {
  const deltaY = interaction.currentY - interaction.startY;
  const factor = getAxisScaleFactor(deltaY, "y");
  const domain = scaleDomainAroundAnchor(interaction.initialDomain, interaction.anchorValue, factor);
  const normalized = model.kind === "line"
    ? normalizeUnboundedDomain(domain, model.rawYDomain)
    : normalizeHistogramYDomain(domain, model.rawYDomain);
  const next = { ...getPanelZoomEntry(model.panelKey) };
  if (!normalized) {
    delete next.yMin;
    delete next.yMax;
  } else {
    next.yMin = normalized[0];
    next.yMax = normalized[1];
  }
  setPanelZoomEntry(model.panelKey, next);
  await renderPanelByKey(model.panelKey);
}

function getAxisScaleFactor(deltaPixels, axis) {
  const signedDelta = axis === "x" ? -deltaPixels : deltaPixels;
  return clamp(Math.exp(signedDelta / AXIS_SCALE_SENSITIVITY), 0.05, 20);
}

function scaleDomainAroundAnchor(currentDomain, anchor, factor) {
  const [currentMin, currentMax] = currentDomain;
  const leftSpan = anchor - currentMin;
  const rightSpan = currentMax - anchor;
  return [
    anchor - leftSpan * factor,
    anchor + rightSpan * factor,
  ];
}

function getMinimumDomainSpan(rawDomain) {
  const span = Math.abs(rawDomain[1] - rawDomain[0]);
  return Math.max(span * 1e-4, 1e-9);
}

function normalizeUnboundedDomain(domain, rawDomain) {
  let [nextMin, nextMax] = domain;
  if (!Number.isFinite(nextMin) || !Number.isFinite(nextMax)) {
    return null;
  }
  if (nextMax < nextMin) {
    [nextMin, nextMax] = [nextMax, nextMin];
  }
  const minSpan = getMinimumDomainSpan(rawDomain);
  if (nextMax - nextMin < minSpan) {
    const center = (nextMin + nextMax) / 2;
    nextMin = center - minSpan / 2;
    nextMax = center + minSpan / 2;
  }
  return nextMax > nextMin ? [nextMin, nextMax] : null;
}

function normalizeHistogramYDomain(domain, rawDomain) {
  const normalized = normalizeUnboundedDomain(domain, rawDomain);
  if (!normalized) {
    return null;
  }
  let [nextMin, nextMax] = normalized;
  if (nextMax <= 0) {
    return null;
  }
  if (nextMin < 0) {
    nextMin = 0;
  }
  const minSpan = getMinimumDomainSpan(rawDomain);
  if (nextMax - nextMin < minSpan) {
    nextMax = nextMin + minSpan;
  }
  return nextMax > nextMin ? [nextMin, nextMax] : null;
}

function normalizeBoundedDomain(domain, bounds, rawDomain) {
  let [nextMin, nextMax] = domain;
  const [boundMin, boundMax] = bounds;
  if (!Number.isFinite(nextMin) || !Number.isFinite(nextMax)) {
    return null;
  }
  if (nextMax < nextMin) {
    [nextMin, nextMax] = [nextMax, nextMin];
  }

  const boundSpan = boundMax - boundMin;
  const minSpan = getMinimumDomainSpan(rawDomain);
  if (nextMax - nextMin < minSpan) {
    const center = clamp((nextMin + nextMax) / 2, boundMin + minSpan / 2, boundMax - minSpan / 2);
    nextMin = center - minSpan / 2;
    nextMax = center + minSpan / 2;
  }

  if (nextMin < boundMin) {
    nextMax += boundMin - nextMin;
    nextMin = boundMin;
  }
  if (nextMax > boundMax) {
    nextMin -= nextMax - boundMax;
    nextMax = boundMax;
  }

  nextMin = Math.max(boundMin, nextMin);
  nextMax = Math.min(boundMax, nextMax);
  if (nextMax - nextMin >= boundSpan * 0.9995) {
    return null;
  }
  if (nextMax - nextMin < minSpan) {
    return null;
  }
  return [nextMin, nextMax];
}

function getClampedLineZoom(start, end) {
  const nextStart = clamp(Math.round(start), state.start, state.end - 1);
  const nextEnd = clamp(Math.round(end), nextStart + 1, state.end);
  if (nextEnd - nextStart >= state.end - state.start) {
    return null;
  }
  return { start: nextStart, end: nextEnd };
}

function getLineWindow(panelKey) {
  const zoom = getPanelZoomEntry(panelKey);
  if (!zoom || zoom.start == null || zoom.end == null) {
    return { start: state.start, end: state.end };
  }
  return getClampedLineZoom(zoom.start, zoom.end) || { start: state.start, end: state.end };
}

function getLineWindowByMetric(metric) {
  const entry = Object.entries(PANEL_CONFIG).find(([, panel]) => panel.metric === metric && panel.type === "line");
  return entry ? getLineWindow(entry[0]) : { start: state.start, end: state.end };
}

function getHistogramZoomDomain(panelKey, rawMin, rawMax) {
  const zoom = getPanelZoomEntry(panelKey);
  if (!zoom || zoom.xMin == null || zoom.xMax == null) {
    return [rawMin, rawMax];
  }
  const nextMin = Math.max(rawMin, Math.min(zoom.xMin, rawMax));
  const nextMax = Math.min(rawMax, Math.max(zoom.xMax, rawMin));
  if (nextMax - nextMin <= 1e-9) {
    clearPanelXZoom(panelKey);
    return [rawMin, rawMax];
  }
  return [nextMin, nextMax];
}

function clampPanelZooms() {
  Object.keys(state.panelZooms).forEach((panelKey) => {
    const panel = PANEL_CONFIG[panelKey];
    if (!panel || !getEnabledMetrics().has(panel.metric)) {
      delete state.panelZooms[panelKey];
      return;
    }
    const zoom = { ...getPanelZoomEntry(panelKey) };
    if (panel.type === "line") {
      if (zoom.start != null && zoom.end != null) {
        const nextLineZoom = getClampedLineZoom(zoom.start, zoom.end);
        if (!nextLineZoom) {
          delete zoom.start;
          delete zoom.end;
        } else {
          zoom.start = nextLineZoom.start;
          zoom.end = nextLineZoom.end;
        }
      }
    } else if (
      zoom.xMin != null && zoom.xMax != null &&
      (!Number.isFinite(zoom.xMin) || !Number.isFinite(zoom.xMax) || zoom.xMax <= zoom.xMin)
    ) {
      delete zoom.xMin;
      delete zoom.xMax;
    }
    if (
      zoom.yMin != null && zoom.yMax != null &&
      (!Number.isFinite(zoom.yMin) || !Number.isFinite(zoom.yMax) || zoom.yMax <= zoom.yMin)
    ) {
      delete zoom.yMin;
      delete zoom.yMax;
    }
    setPanelZoomEntry(panelKey, zoom);
  });
  synchronizeVisibleLineZooms();
}

function clearPanelZoom(panelKey) {
  delete state.panelZooms[panelKey];
}

function buildLineMetaText(panelKey, payload) {
  const zoom = getPanelZoomEntry(panelKey);
  const tags = [];
  if (zoom.start != null && zoom.end != null) {
    tags.push("X zoomed");
  }
  if (zoom.yMin != null && zoom.yMax != null) {
    tags.push("Y scaled");
  }
  return `${payload.series.length} trajectories | ${formatInt(payload.end - payload.start)} snapshots in view | ${tags.join(" + ") || "full active window"}`;
}

function buildHistogramMetaText(panelKey, payload) {
  const zoom = getPanelZoomEntry(panelKey);
  const tags = [];
  if (zoom.xMin != null && zoom.xMax != null) {
    tags.push("X scaled");
  }
  if (zoom.yMin != null && zoom.yMax != null) {
    tags.push("Y scaled");
  }
  return `${payload.series.length} distributions | ${payload.bins} bins | ${tags.join(" + ") || "full energy range"}`;
}

function handleLineHover(model, mx, event) {
  const approxX = Math.round(model.xScale.invert(mx));
  const rows = model.payload.series
    .map((series) => {
      const idx = bisectClosest(series.x, approxX);
      return {
        label: series.label,
        color: series.color,
        x: series.x[idx],
        y: series.y[idx]
      };
    })
    .sort((a, b) => a.label.localeCompare(b.label));

  const anchorX = model.xScale(rows[0].x);
  model.crosshair.hidden = false;
  model.crosshair.style.left = `${anchorX}px`;

  const content = [`<strong>Snapshot ${formatInt(rows[0].x)}</strong>`]
    .concat(rows.map((row) => `<div><span style="color:${row.color}">&#9679;</span> ${row.label}: ${Number(row.y).toFixed(2)} ${model.unitLabel}</div>`))
    .join("");
  positionTooltip(model, event, content);
}

function handleHistogramHover(model, mx, event) {
  const xValue = model.xScale.invert(mx);
  const rows = model.payload.series
    .map((series) => {
      const index = findBinIndex(series.edges, xValue);
      if (index < 0) return null;
      return {
        label: series.label,
        color: series.color,
        count: series.counts[index],
        start: series.edges[index],
        end: series.edges[index + 1]
      };
    })
    .filter(Boolean);

  if (!rows.length) {
    hideTooltip(model);
    return;
  }

  model.crosshair.hidden = false;
  model.crosshair.style.left = `${mx}px`;

  const content = [
    `<strong>${rows[0].start.toFixed(2)} to ${rows[0].end.toFixed(2)} ${formatBracketUnit(state.meta.units.energy)}</strong>`,
    ...rows.map((row) => `<div><span style="color:${row.color}">&#9679;</span> ${row.label}: ${formatInt(row.count)}</div>`)
  ].join("");
  positionTooltip(model, event, content);
}

function positionTooltip(model, event, html) {
  const stage = model.tooltip.parentElement.getBoundingClientRect();
  const localX = event.clientX - stage.left;
  const localY = event.clientY - stage.top;
  model.tooltip.hidden = false;
  model.tooltip.innerHTML = html;

  const left = Math.min(localX + 16, stage.width - model.tooltip.offsetWidth - 12);
  const top = Math.max(12, localY - model.tooltip.offsetHeight - 18);
  model.tooltip.style.left = `${left}px`;
  model.tooltip.style.top = `${top}px`;
}

function hideTooltip(model) {
  model.tooltip.hidden = true;
  model.crosshair.hidden = true;
}

function toggleSelection(selectionSet, label) {
  if (selectionSet.has(label)) {
    selectionSet.delete(label);
  } else {
    selectionSet.add(label);
  }
}

function bisectClosest(sortedValues, target) {
  if (sortedValues.length === 1) return 0;
  let low = 0;
  let high = sortedValues.length - 1;
  while (low < high) {
    const mid = Math.floor((low + high) / 2);
    if (sortedValues[mid] < target) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  if (low === 0) return 0;
  const previous = sortedValues[low - 1];
  const current = sortedValues[low];
  return Math.abs(current - target) < Math.abs(previous - target) ? low : low - 1;
}

function findBinIndex(edges, value) {
  for (let index = 0; index < edges.length - 1; index += 1) {
    if (value >= edges[index] && value <= edges[index + 1]) {
      return index;
    }
  }
  return -1;
}

function compactMetricTitle(title) {
  return title.replace(" Norms", "").replace(" Magnitudes", "");
}

function summarizeStateLabels(labels) {
  const ids = labels
    .map((label) => Number(String(label).replace(/^S/i, "")))
    .filter((value) => Number.isFinite(value))
    .sort((a, b) => a - b);

  if (!ids.length) {
    return "--";
  }

  const ranges = [];
  let start = ids[0];
  let end = ids[0];

  for (let index = 1; index < ids.length; index += 1) {
    const value = ids[index];
    if (value === end + 1) {
      end = value;
      continue;
    }
    ranges.push(start === end ? `S${start}` : `S${start}-S${end}`);
    start = value;
    end = value;
  }
  ranges.push(start === end ? `S${start}` : `S${start}-S${end}`);
  return ranges.join(", ");
}

function formatUnitLabel(rawUnit) {
  return String(rawUnit)
    .replace(/\$\\AA\$/g, "\u00C5")
    .replace(/\\AA/g, "\u00C5")
    .replace(/\$/g, "");
}

function formatBracketUnit(rawUnit) {
  const formatted = formatUnitLabel(rawUnit);
  return formatted ? `[${formatted}]` : "";
}

function withUnit(label, rawUnit) {
  const suffix = formatBracketUnit(rawUnit);
  return suffix ? `${label} ${suffix}` : label;
}

function axisLabelForMetric(metric) {
  if (state.meta && state.meta.axis_labels && state.meta.axis_labels[metric]) {
    return state.meta.axis_labels[metric];
  }
  return METRIC_AXIS_LABELS[metric] || metric;
}

function formatStatistic(value) {
  if (!Number.isFinite(value)) {
    return "--";
  }
  const absValue = Math.abs(value);
  if (absValue === 0) {
    return "0.00";
  }
  if (absValue >= 10000 || absValue < 0.01) {
    return Number(value).toExponential(3);
  }
  return Number(value).toFixed(3);
}

function formatSnapshotCount(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return "--";
  }
  if (Math.abs(numeric) < 5000) {
    return formatInt(numeric);
  }

  const absValue = Math.abs(numeric);
  if (absValue < 1_000_000) {
    const compact = numeric / 1000;
    return Number.isInteger(compact) ? `${compact}K` : `${compact.toFixed(1).replace(/\.0$/, "")}K`;
  }

  const compact = numeric / 1_000_000;
  return Number.isInteger(compact) ? `${compact}M` : `${compact.toFixed(1).replace(/\.0$/, "")}M`;
}

async function fetchJSON(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `Request failed for ${url}`);
  }
  return payload;
}

function setStatus(message, isError = false) {
  const status = document.getElementById("global-status");
  status.innerHTML = isError ? `<strong>${message}</strong>` : message;
}

function formatInt(value) {
  return new Intl.NumberFormat("en-US").format(Math.round(value));
}

function formatNumber(value) {
  if (Math.abs(value) >= 1000) {
    return formatInt(value);
  }
  return Number(value).toFixed(2).replace(/\.00$/, "");
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function debounce(fn, wait) {
  let timer = null;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), wait);
  };
}
