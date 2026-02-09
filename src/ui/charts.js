function setupCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;

  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));

  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  return {
    ctx,
    width: rect.width,
    height: rect.height,
  };
}

function extent(values, paddingRatio = 0.1) {
  const min = Math.min(...values);
  const max = Math.max(...values);

  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return [0, 1];
  }

  if (Math.abs(max - min) < 1e-12) {
    return [min - 1, max + 1];
  }

  const span = max - min;
  return [min - span * paddingRatio, max + span * paddingRatio];
}

function mean(values) {
  if (!Array.isArray(values) || values.length === 0) {
    return 0;
  }
  let sum = 0;
  for (const v of values) {
    sum += Number(v) || 0;
  }
  return sum / values.length;
}

const CHART_THEME = {
  axis: "#9aa8ba",
  axisText: "#405064",
  grid: "rgba(63, 79, 101, 0.035)",
  legendText: "#223045",
  bgVolatile: "rgba(58, 88, 124, 0.05)",
  bgStress: "rgba(124, 82, 52, 0.06)",
  shiftLine: "rgba(73, 84, 102, 0.24)",
  zero: "#5f6f84",
};

function applyLineQuality(ctx) {
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
}

function roundedRectPath(ctx, x, y, w, h, r) {
  const radius = Math.max(0, Math.min(r, Math.min(w, h) / 2));
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + w - radius, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + radius);
  ctx.lineTo(x + w, y + h - radius);
  ctx.quadraticCurveTo(x + w, y + h, x + w - radius, y + h);
  ctx.lineTo(x + radius, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
}

function drawAxes(ctx, dims, xLabel, yLabel) {
  const { left, top, right, bottom } = dims;

  ctx.strokeStyle = CHART_THEME.axis;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(left, top);
  ctx.lineTo(left, bottom);
  ctx.lineTo(right, bottom);
  ctx.stroke();

  ctx.fillStyle = CHART_THEME.axisText;
  ctx.font = "500 11px 'IBM Plex Sans', sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(xLabel, (left + right) / 2, bottom + 30);

  ctx.save();
  ctx.translate(left - 40, (top + bottom) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(yLabel, 0, 0);
  ctx.restore();
}

function drawGrid(ctx, dims, ticks = 4) {
  const { left, top, right, bottom } = dims;
  ctx.strokeStyle = CHART_THEME.grid;
  ctx.lineWidth = 1;
  applyLineQuality(ctx);

  for (let i = 1; i <= ticks; i += 1) {
    const y = top + ((bottom - top) * i) / (ticks + 1);
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(right, y);
    ctx.stroke();
  }
}

function drawLegend(ctx, legendItems, y = 10, maxWidth = Number.POSITIVE_INFINITY) {
  let x = 14;
  let rowY = y;
  ctx.font = "500 11px 'IBM Plex Sans', sans-serif";
  applyLineQuality(ctx);

  for (const item of legendItems) {
    const labelWidth = ctx.measureText(item.label).width;
    const chipWidth = 16 + 8 + labelWidth + 6;
    if (x + chipWidth > maxWidth && x > 14) {
      x = 14;
      rowY += 18;
    }

    if (item.kind === "pattern") {
      ctx.save();
      ctx.globalAlpha = item.alpha === undefined ? 1 : item.alpha;
      drawPatternFill(ctx, x, rowY - 1, 14, 6, item.pattern || "solid", item.color);
      ctx.strokeStyle = item.color;
      ctx.lineWidth = 1;
      ctx.strokeRect(x, rowY - 1, 14, 6);
      ctx.restore();
    } else {
      ctx.save();
      ctx.strokeStyle = item.color;
      ctx.lineWidth = 2;
      ctx.globalAlpha = item.alpha === undefined ? 1 : item.alpha;
      ctx.setLineDash(item.dash || []);
      ctx.beginPath();
      ctx.moveTo(x, rowY + 2);
      ctx.lineTo(x + 14, rowY + 2);
      ctx.stroke();
      ctx.restore();
    }

    ctx.fillStyle = CHART_THEME.legendText;
    ctx.textAlign = "left";
    ctx.fillText(item.label, x + 18, rowY + 4);

    x += chipWidth + 10;
  }
}

function drawPatternFill(ctx, x, y, w, h, pattern, color) {
  ctx.save();
  ctx.beginPath();
  ctx.rect(x, y, w, h);
  ctx.clip();

  if (pattern === "solid") {
    ctx.fillStyle = color;
    ctx.fillRect(x, y, w, h);
    ctx.restore();
    return;
  }

  ctx.fillStyle = "#ffffff";
  ctx.fillRect(x, y, w, h);

  ctx.strokeStyle = color;
  ctx.lineWidth = 1;

  if (pattern === "horizontal") {
    for (let yy = y - 2; yy <= y + h + 2; yy += 4) {
      ctx.beginPath();
      ctx.moveTo(x - 1, yy);
      ctx.lineTo(x + w + 1, yy);
      ctx.stroke();
    }
    ctx.restore();
    return;
  }

  if (pattern === "vertical") {
    for (let xx = x - 2; xx <= x + w + 2; xx += 4) {
      ctx.beginPath();
      ctx.moveTo(xx, y - 1);
      ctx.lineTo(xx, y + h + 1);
      ctx.stroke();
    }
    ctx.restore();
    return;
  }

  for (let k = -h; k <= w + h; k += 4) {
    ctx.beginPath();
    ctx.moveTo(x + k, y + h);
    ctx.lineTo(x + k + h, y);
    ctx.stroke();
  }
  ctx.restore();
}

function drawRegimeBackdrop(ctx, dims, states, xToPx, maxT) {
  if (!Array.isArray(states) || states.length === 0) {
    return;
  }

  const stepPx = maxT > 0 ? (dims.right - dims.left) / maxT : 8;

  const shadeState = (state, fill) => {
    const idx = [];
    for (let i = 0; i < states.length; i += 1) {
      if (states[i] === state) {
        idx.push(i);
      }
    }

    if (idx.length === 0) {
      return;
    }

    const segments = [];
    let segStart = idx[0];
    let segEnd = idx[0];

    for (let i = 1; i < idx.length; i += 1) {
      if (idx[i] <= segEnd + 1) {
        segEnd = idx[i];
      } else {
        segments.push([segStart, segEnd]);
        segStart = idx[i];
        segEnd = idx[i];
      }
    }
    segments.push([segStart, segEnd]);

    ctx.fillStyle = fill;
    for (const [a, b] of segments) {
      const left = Math.max(dims.left, xToPx(a) - stepPx * 0.5);
      const right = Math.min(dims.right, xToPx(b) + stepPx * 0.5);
      ctx.fillRect(left, dims.top, Math.max(1, right - left), dims.bottom - dims.top);
    }
  };

  shadeState("volatile", CHART_THEME.bgVolatile);
  shadeState("stress", CHART_THEME.bgStress);

  ctx.strokeStyle = CHART_THEME.shiftLine;
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  for (let i = 0; i < states.length; i += 1) {
    if (states[i] !== "shift") {
      continue;
    }
    const x = xToPx(i);
    ctx.beginPath();
    ctx.moveTo(x, dims.top);
    ctx.lineTo(x, dims.bottom);
    ctx.stroke();
  }
  ctx.setLineDash([]);
}

function methodPattern(methodId) {
  if (methodId === "naive") {
    return "horizontal";
  }
  if (methodId === "anchor") {
    return "diagonal";
  }
  return "vertical";
}

function shortMethod(label) {
  if (!label) {
    return "";
  }
  const lower = String(label).toLowerCase();
  if (lower.includes("naive")) {
    return "Naive";
  }
  if (lower.includes("hybrid") || lower.includes("constrained") || lower.includes("proj")) {
    return "Hybrid";
  }
  if (lower.includes("replay") || lower.includes("anchor")) {
    return "Replay";
  }
  return label;
}

export function drawEquity(canvas, series, regimeStates) {
  const { ctx, width, height } = setupCanvas(canvas);
  ctx.clearRect(0, 0, width, height);
  applyLineQuality(ctx);

  const dims = {
    left: 62,
    right: width - 20,
    top: 30,
    bottom: height - 48,
  };

  drawAxes(ctx, dims, "Time", "Portfolio value");

  const allY = series.flatMap((s) => s.values || []);
  const [yMin, yMax] = extent(allY, 0.08);
  const maxT = Math.max(...series.map((s) => (s.values?.length || 1) - 1), 1);

  function xToPx(t) {
    return dims.left + (t / Math.max(1, maxT)) * (dims.right - dims.left);
  }

  function yToPx(y) {
    return dims.bottom - ((y - yMin) / (yMax - yMin)) * (dims.bottom - dims.top);
  }

  drawRegimeBackdrop(ctx, dims, regimeStates, xToPx, maxT);

  for (const s of series) {
    ctx.save();
    ctx.strokeStyle = s.color;
    ctx.lineWidth = (s.lineWidth || 2) + 0.1;
    ctx.globalAlpha = s.alpha === undefined ? 1 : s.alpha;
    ctx.setLineDash(s.dash || []);
    ctx.beginPath();

    const values = s.values || [];
    for (let t = 0; t < values.length; t += 1) {
      const x = xToPx(t);
      const y = yToPx(values[t]);
      if (t === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }

    ctx.stroke();
    ctx.restore();
  }

  drawLegend(
    ctx,
    series.map((s) => ({ label: shortMethod(s.label), color: s.color, dash: s.dash, alpha: s.alpha })),
    10,
    width - 14,
  );
}

export function drawDrawdown(canvas, series, regimeStates) {
  const { ctx, width, height } = setupCanvas(canvas);
  ctx.clearRect(0, 0, width, height);
  applyLineQuality(ctx);

  const dims = {
    left: 62,
    right: width - 20,
    top: 30,
    bottom: height - 48,
  };

  drawAxes(ctx, dims, "Time", "Drawdown (%)");

  const drawdownSeries = series.map((s) => {
    let peak = -Infinity;
    const dd = (s.values || []).map((v) => {
      peak = Math.max(peak, v);
      return peak > 0 ? ((v / peak) - 1) * 100 : 0;
    });
    return {
      ...s,
      values: dd,
    };
  });

  const allY = drawdownSeries.flatMap((s) => s.values || []);
  const [rawMin] = extent(allY, 0.05);
  const yMin = Math.min(rawMin, -1);
  const yMax = 1;
  const maxT = Math.max(...drawdownSeries.map((s) => (s.values?.length || 1) - 1), 1);

  function xToPx(t) {
    return dims.left + (t / Math.max(1, maxT)) * (dims.right - dims.left);
  }

  function yToPx(y) {
    return dims.bottom - ((y - yMin) / (yMax - yMin)) * (dims.bottom - dims.top);
  }

  drawRegimeBackdrop(ctx, dims, regimeStates, xToPx, maxT);

  const zeroY = yToPx(0);
  ctx.strokeStyle = CHART_THEME.zero;
  ctx.lineWidth = 1.3;
  ctx.beginPath();
  ctx.moveTo(dims.left, zeroY);
  ctx.lineTo(dims.right, zeroY);
  ctx.stroke();

  for (const s of drawdownSeries) {
    ctx.save();
    ctx.strokeStyle = s.color;
    ctx.lineWidth = (s.lineWidth || 2) + 0.1;
    ctx.globalAlpha = s.alpha === undefined ? 1 : s.alpha;
    ctx.setLineDash(s.dash || []);
    ctx.beginPath();

    const values = s.values || [];
    for (let t = 0; t < values.length; t += 1) {
      const x = xToPx(t);
      const y = yToPx(values[t]);
      if (t === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }

    ctx.stroke();
    ctx.restore();
  }

  drawLegend(
    ctx,
    drawdownSeries.map((s) => ({ label: shortMethod(s.label), color: s.color, dash: s.dash, alpha: s.alpha })),
    10,
    width - 14,
  );
}

export function drawAllocationProfiles(canvas, profiles, regimeStates) {
  const { ctx, width, height } = setupCanvas(canvas);
  ctx.clearRect(0, 0, width, height);
  applyLineQuality(ctx);

  if (!profiles || profiles.length === 0) {
    return;
  }

  const frame = {
    left: 78,
    right: width - 24,
    top: 30,
    bottom: height - 40,
  };

  const panelGap = 22;
  const panelHeight = (frame.bottom - frame.top - panelGap) / 2;
  const allocPanel = {
    left: frame.left,
    right: frame.right,
    top: frame.top,
    bottom: frame.top + panelHeight,
  };
  const turnPanel = {
    left: frame.left,
    right: frame.right,
    top: allocPanel.bottom + panelGap,
    bottom: frame.bottom,
  };

  const maxT = Math.max(...profiles.map((p) => (p.values?.length || 1) - 1), 1);

  function xToPx(t) {
    return frame.left + (t / Math.max(1, maxT)) * (frame.right - frame.left);
  }

  function yAlloc(v) {
    return allocPanel.bottom - Math.max(0, Math.min(1, v)) * (allocPanel.bottom - allocPanel.top);
  }

  const maxTurn = Math.max(1e-6, ...profiles.flatMap((p) => (p.turnovers || []).map((v) => Number(v) || 0)));
  function yTurn(v) {
    return turnPanel.bottom - Math.max(0, Math.min(1, (Number(v) || 0) / maxTurn)) * (turnPanel.bottom - turnPanel.top);
  }

  drawRegimeBackdrop(ctx, allocPanel, regimeStates, xToPx, maxT);
  drawRegimeBackdrop(ctx, turnPanel, regimeStates, xToPx, maxT);

  ctx.strokeStyle = "rgba(124, 139, 158, 0.85)";
  ctx.lineWidth = 1.1;
  ctx.strokeRect(allocPanel.left, allocPanel.top, allocPanel.right - allocPanel.left, allocPanel.bottom - allocPanel.top);
  ctx.strokeRect(turnPanel.left, turnPanel.top, turnPanel.right - turnPanel.left, turnPanel.bottom - turnPanel.top);

  ctx.fillStyle = "#2f3e52";
  ctx.font = "600 11px 'IBM Plex Sans', sans-serif";
  ctx.textAlign = "left";
  ctx.fillText("risky allocation", frame.left + 6, allocPanel.top + 12);
  ctx.fillText("turnover", frame.left + 6, turnPanel.top + 12);

  const allocMid = yAlloc(0.5);
  ctx.strokeStyle = "rgba(104, 119, 138, 0.3)";
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 4]);
  ctx.beginPath();
  ctx.moveTo(allocPanel.left, allocMid);
  ctx.lineTo(allocPanel.right, allocMid);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.textAlign = "right";
  ctx.fillStyle = "#48586e";
  ctx.font = "500 10px 'IBM Plex Sans', sans-serif";
  ctx.fillText("100%", frame.left - 10, allocPanel.top + 2);
  ctx.fillText("50%", frame.left - 10, allocMid + 3);
  ctx.fillText("0%", frame.left - 10, allocPanel.bottom + 3);
  ctx.fillText(`${(maxTurn * 100).toFixed(0)}%`, frame.left - 10, turnPanel.top + 2);
  ctx.fillText("0%", frame.left - 10, turnPanel.bottom + 3);

  for (const p of profiles) {
    const values = p.values || [];
    const turns = p.turnovers || [];

    ctx.save();
    ctx.strokeStyle = p.color;
    ctx.lineWidth = (p.lineWidth || 2) + 0.3;
    ctx.globalAlpha = p.alpha === undefined ? 1 : p.alpha;
    ctx.setLineDash([]);
    ctx.beginPath();
    for (let t = 0; t < values.length; t += 1) {
      const x = xToPx(t);
      const y = yAlloc(values[t]);
      if (t === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
    ctx.restore();

    ctx.save();
    ctx.strokeStyle = p.color;
    ctx.lineWidth = 1.8;
    ctx.globalAlpha = p.alpha === undefined ? 1 : p.alpha;
    ctx.setLineDash([1.6, 4]);
    ctx.beginPath();
    for (let t = 0; t < turns.length; t += 1) {
      const x = xToPx(t);
      const y = yTurn(turns[t]);
      if (t === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
    ctx.restore();
  }

  ctx.strokeStyle = "rgba(124, 139, 158, 0.95)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(frame.left, allocPanel.top);
  ctx.lineTo(frame.left, turnPanel.bottom);
  ctx.lineTo(frame.right, turnPanel.bottom);
  ctx.stroke();

  ctx.fillStyle = "#3f4f64";
  ctx.font = "500 10px 'IBM Plex Sans', sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("time", (frame.left + frame.right) / 2, frame.bottom + 22);
}

export function drawRegimeRisk(canvas, regimes, rows) {
  const { ctx, width, height } = setupCanvas(canvas);
  ctx.clearRect(0, 0, width, height);
  applyLineQuality(ctx);

  if (!regimes || regimes.length === 0 || !rows || rows.length === 0) {
    return;
  }

  const dims = {
    left: 64,
    right: width - 16,
    top: 28,
    bottom: height - 52,
  };

  const allVals = [];
  for (const r of rows) {
    for (const regime of regimes) {
      allVals.push(r.sharpe?.[regime] || 0);
    }
  }

  const [rawMin, rawMax] = extent(allVals, 0.15);
  const yMin = Math.min(rawMin, -1);
  const yMax = Math.max(rawMax, 1);

  function yToPx(y) {
    return dims.bottom - ((y - yMin) / (yMax - yMin)) * (dims.bottom - dims.top);
  }

  function xCenter(i, n) {
    return dims.left + ((i + 0.5) / n) * (dims.right - dims.left);
  }

  drawGrid(ctx, dims, 4);
  drawAxes(ctx, dims, "Market regime", "Annualized Sharpe");

  const zeroY = yToPx(0);
  ctx.strokeStyle = CHART_THEME.zero;
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  ctx.moveTo(dims.left, zeroY);
  ctx.lineTo(dims.right, zeroY);
  ctx.stroke();

  const groupWidth = (dims.right - dims.left) / regimes.length;
  const barWidth = Math.min(16, groupWidth * 0.2);

  for (let g = 0; g < regimes.length; g += 1) {
    const cx = xCenter(g, regimes.length);
    const regime = regimes[g];

    for (let i = 0; i < rows.length; i += 1) {
      const row = rows[i];
      const value = row.sharpe?.[regime] || 0;
      const y = yToPx(value);
      const top = Math.min(y, zeroY);
      const h = Math.max(1, Math.abs(y - zeroY));
      const x = cx + (i - (rows.length - 1) / 2) * (barWidth * 1.25) - barWidth / 2;
      const alpha = row.alpha === undefined ? 1 : row.alpha;

      ctx.save();
      ctx.globalAlpha = alpha;
      drawPatternFill(ctx, x, top, barWidth, h, methodPattern(row.id), row.color);
      ctx.strokeStyle = row.color;
      ctx.lineWidth = 1.1;
      ctx.strokeRect(x, top, barWidth, h);
      ctx.restore();
    }

    ctx.fillStyle = "#222222";
    ctx.font = "11px 'IBM Plex Sans', sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(regime.charAt(0).toUpperCase() + regime.slice(1), cx, dims.bottom + 16);
  }

  drawLegend(
    ctx,
    rows.map((row) => ({
      label: shortMethod(row.label),
      color: row.color,
      kind: "pattern",
      pattern: methodPattern(row.id),
      alpha: row.alpha,
    })),
    10,
    width - 14,
  );
}

export function drawPortfolioState(canvas, rows) {
  const { ctx, width, height } = setupCanvas(canvas);
  ctx.clearRect(0, 0, width, height);
  applyLineQuality(ctx);

  if (!rows || rows.length === 0) {
    return;
  }

  const dims = {
    left: 58,
    right: width - 22,
    top: 42,
    bottom: height - 60,
  };

  ctx.strokeStyle = CHART_THEME.axis;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(dims.left, dims.top);
  ctx.lineTo(dims.left, dims.bottom);
  ctx.lineTo(dims.right, dims.bottom);
  ctx.stroke();

  ctx.save();
  ctx.fillStyle = CHART_THEME.axisText;
  ctx.font = "500 11px 'IBM Plex Sans', sans-serif";
  ctx.translate(dims.left - 40, (dims.top + dims.bottom) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillText("Risky allocation", 0, 0);
  ctx.restore();

  const ticks = [0, 0.25, 0.5, 0.75, 1];
  ctx.fillStyle = CHART_THEME.axisText;
  ctx.font = "500 10px 'IBM Plex Sans', sans-serif";
  ctx.textAlign = "right";
  for (const tick of ticks) {
    const y = dims.bottom - tick * (dims.bottom - dims.top);
    ctx.fillText(`${Math.round(tick * 100)}%`, dims.left - 8, y + 3);
  }

  const groupWidth = (dims.right - dims.left) / rows.length;
  const barGap = Math.max(8, Math.min(14, groupWidth * 0.09));
  const barWidth = Math.max(18, Math.min(34, groupWidth * 0.2));
  const chartHeight = dims.bottom - dims.top;

  const legendY = dims.top - 22;
  ctx.font = "500 10px 'IBM Plex Sans', sans-serif";
  ctx.textAlign = "left";
  ctx.fillStyle = "#3f4f63";
  ctx.fillText("calm sleeve", dims.left + 18, legendY + 5);
  ctx.fillText("stress sleeve", dims.left + 118, legendY + 5);
  ctx.strokeStyle = "#8ea0b4";
  ctx.lineWidth = 1;
  ctx.strokeRect(dims.left, legendY - 2, 12, 12);
  ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
  ctx.fillRect(dims.left + 1, legendY - 1, 10, 10);
  ctx.strokeStyle = "#8ea0b4";
  ctx.strokeRect(dims.left + 96, legendY - 2, 12, 12);
  ctx.fillStyle = "rgba(74, 95, 118, 0.22)";
  ctx.fillRect(dims.left + 97, legendY - 1, 10, 10);

  for (let i = 0; i < rows.length; i += 1) {
    const row = rows[i];
    const alpha = row.alpha === undefined ? 1 : row.alpha;
    const centerX = dims.left + groupWidth * (i + 0.5);
    const calm = Math.max(0, Math.min(1, row.calmWeight || 0));
    const stress = Math.max(0, Math.min(1, row.stressWeight || 0));

    drawStackBarVertical(
      ctx,
      centerX - barWidth - barGap * 0.5,
      dims.top,
      barWidth,
      chartHeight,
      calm,
      row.color,
      methodPattern(row.id),
      alpha * 0.66,
    );
    drawStackBarVertical(
      ctx,
      centerX + barGap * 0.5,
      dims.top,
      barWidth,
      chartHeight,
      stress,
      row.color,
      methodPattern(row.id),
      alpha,
    );

    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.fillStyle = "#223244";
    ctx.font = "600 11px 'IBM Plex Sans', sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(shortMethod(row.label), centerX, dims.bottom + 18);
    ctx.restore();
  }
}

function drawStackBarVertical(ctx, x, y, width, height, riskyWeight, color, pattern, alpha) {
  const riskyHeight = Math.max(0, Math.min(height, height * riskyWeight));
  const cashHeight = height - riskyHeight;

  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = "#9da8b6";
  ctx.lineWidth = 1;
  ctx.strokeRect(x, y, width, height);
  ctx.restore();

  if (cashHeight > 0) {
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(x, y, width, cashHeight);
    ctx.strokeStyle = "#c7cfdb";
    ctx.lineWidth = 1;
    ctx.strokeRect(x, y, width, cashHeight);
    ctx.restore();
  }

  if (riskyHeight > 0) {
    ctx.save();
    ctx.globalAlpha = alpha;
    drawPatternFill(ctx, x, y + cashHeight, width, riskyHeight, pattern, color);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.strokeRect(x, y + cashHeight, width, riskyHeight);
    ctx.restore();
  }
}
