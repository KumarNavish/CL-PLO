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

function drawAxes(ctx, dims, xLabel, yLabel) {
  const { left, top, right, bottom } = dims;

  ctx.strokeStyle = "#555555";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(left, top);
  ctx.lineTo(left, bottom);
  ctx.lineTo(right, bottom);
  ctx.stroke();

  ctx.fillStyle = "#222222";
  ctx.font = "12px 'IBM Plex Sans', 'Avenir Next', sans-serif";
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
  ctx.strokeStyle = "rgba(17, 17, 17, 0.12)";
  ctx.lineWidth = 1;

  for (let i = 1; i <= ticks; i += 1) {
    const x = left + ((right - left) * i) / (ticks + 1);
    const y = top + ((bottom - top) * i) / (ticks + 1);

    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, bottom);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(right, y);
    ctx.stroke();
  }
}

function drawLegend(ctx, legendItems, y = 10) {
  let x = 14;
  ctx.font = "12px 'IBM Plex Sans', 'Avenir Next', sans-serif";

  for (const item of legendItems) {
    if (item.kind === "pattern") {
      ctx.save();
      ctx.globalAlpha = item.alpha === undefined ? 1 : item.alpha;
      drawPatternFill(ctx, x, y - 2, 14, 6, item.pattern || "solid", item.color);
      ctx.strokeStyle = item.color;
      ctx.lineWidth = 1;
      ctx.strokeRect(x, y - 2, 14, 6);
      ctx.restore();
    } else {
      ctx.save();
      ctx.strokeStyle = item.color;
      ctx.lineWidth = 2;
      ctx.globalAlpha = item.alpha === undefined ? 1 : item.alpha;
      ctx.setLineDash(item.dash || []);
      ctx.beginPath();
      ctx.moveTo(x, y + 2);
      ctx.lineTo(x + 14, y + 2);
      ctx.stroke();
      ctx.restore();
    }

    ctx.fillStyle = "#222222";
    ctx.textAlign = "left";
    ctx.fillText(item.label, x + 18, y + 4);

    x += ctx.measureText(item.label).width + 46;
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

  shadeState("volatile", "rgba(17, 17, 17, 0.04)");
  shadeState("stress", "rgba(17, 17, 17, 0.1)");

  ctx.strokeStyle = "rgba(17, 17, 17, 0.26)";
  ctx.lineWidth = 1;
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
  if (label.includes("Naive")) {
    return "Naive";
  }
  if (label.includes("Constrained")) {
    return "Hybrid";
  }
  if (label.includes("Replay")) {
    return "Replay";
  }
  return label;
}

export function drawEquity(canvas, series, regimeStates) {
  const { ctx, width, height } = setupCanvas(canvas);
  ctx.clearRect(0, 0, width, height);

  const dims = {
    left: 62,
    right: width - 20,
    top: 24,
    bottom: height - 46,
  };

  drawGrid(ctx, dims, 4);
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
    ctx.lineWidth = s.lineWidth || 2;
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
  );
}

export function drawDrawdown(canvas, series, regimeStates) {
  const { ctx, width, height } = setupCanvas(canvas);
  ctx.clearRect(0, 0, width, height);

  const dims = {
    left: 62,
    right: width - 20,
    top: 24,
    bottom: height - 46,
  };

  drawGrid(ctx, dims, 4);
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
  ctx.strokeStyle = "#333333";
  ctx.lineWidth = 1.3;
  ctx.beginPath();
  ctx.moveTo(dims.left, zeroY);
  ctx.lineTo(dims.right, zeroY);
  ctx.stroke();

  for (const s of drawdownSeries) {
    ctx.save();
    ctx.strokeStyle = s.color;
    ctx.lineWidth = s.lineWidth || 2;
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
  );
}

export function drawAllocationProfiles(canvas, profiles, regimeStates) {
  const { ctx, width, height } = setupCanvas(canvas);
  ctx.clearRect(0, 0, width, height);

  if (!profiles || profiles.length === 0) {
    return;
  }

  const dims = {
    left: 92,
    right: width - 20,
    top: 24,
    bottom: height - 28,
  };

  const maxT = Math.max(...profiles.map((p) => (p.values?.length || 1) - 1), 1);

  function xToPx(t) {
    return dims.left + (t / Math.max(1, maxT)) * (dims.right - dims.left);
  }

  const turnoverScale = Math.max(
    1e-6,
    ...profiles.flatMap((p) => (p.turnovers || []).map((v) => Number(v) || 0)),
  );

  drawRegimeBackdrop(ctx, dims, regimeStates, xToPx, maxT);

  const laneGap = 14;
  const laneHeight = (dims.bottom - dims.top - laneGap * (profiles.length - 1)) / profiles.length;

  for (let i = 0; i < profiles.length; i += 1) {
    const p = profiles[i];
    const laneTop = dims.top + i * (laneHeight + laneGap);
    const laneBottom = laneTop + laneHeight;

    ctx.strokeStyle = "rgba(17, 17, 17, 0.16)";
    ctx.lineWidth = 1;
    ctx.strokeRect(dims.left, laneTop, dims.right - dims.left, laneHeight);

    ctx.fillStyle = "#2a2a2a";
    ctx.font = "12px 'IBM Plex Sans', 'Avenir Next', sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(shortMethod(p.label), 14, laneTop + laneHeight * 0.45);

    function yToPx(v) {
      return laneBottom - Math.max(0, Math.min(1, v)) * laneHeight;
    }

    const values = p.values || [];
    const turns = p.turnovers || [];

    // Cash/risky allocation fill: lower area is risky weight, upper area is cash.
    ctx.save();
    ctx.globalAlpha = (p.alpha === undefined ? 1 : p.alpha) * 0.08;
    ctx.fillStyle = "#111111";
    ctx.beginPath();
    ctx.moveTo(xToPx(0), laneTop);
    for (let t = 0; t < values.length; t += 1) {
      ctx.lineTo(xToPx(t), yToPx(values[t]));
    }
    ctx.lineTo(xToPx(Math.max(values.length - 1, 0)), laneTop);
    ctx.closePath();
    ctx.fill();
    ctx.restore();

    ctx.save();
    ctx.globalAlpha = (p.alpha === undefined ? 1 : p.alpha) * 0.15;
    ctx.fillStyle = p.color;
    ctx.beginPath();
    ctx.moveTo(xToPx(0), laneBottom);
    for (let t = 0; t < values.length; t += 1) {
      ctx.lineTo(xToPx(t), yToPx(values[t]));
    }
    ctx.lineTo(xToPx(Math.max(values.length - 1, 0)), laneBottom);
    ctx.closePath();
    ctx.fill();
    ctx.restore();

    ctx.fillStyle = "#5a646f";
    ctx.font = "10px 'IBM Plex Sans', 'Avenir Next', sans-serif";
    ctx.fillText("100%", 54, laneTop + 10);
    ctx.fillText("0%", 66, laneBottom - 2);
    ctx.fillText("TO", dims.right - 16, laneTop + 11);

    ctx.save();
    ctx.strokeStyle = p.color;
    ctx.lineWidth = p.lineWidth || 2;
    ctx.globalAlpha = p.alpha === undefined ? 1 : p.alpha;
    ctx.setLineDash(p.dash || []);
    ctx.beginPath();

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

    // Dashed turnover path inside each lane for stability-vs-responsiveness read.
    ctx.save();
    ctx.strokeStyle = "#111111";
    ctx.lineWidth = 1.2;
    ctx.globalAlpha = (p.alpha === undefined ? 1 : p.alpha) * 0.75;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    for (let t = 0; t < Math.min(values.length, turns.length); t += 1) {
      const x = xToPx(t);
      const normalized = Math.max(0, Math.min(1, (turns[t] || 0) / turnoverScale));
      const y = laneBottom - normalized * laneHeight * 0.42;
      if (t === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
    ctx.restore();

    ctx.fillStyle = "#3e4754";
    ctx.font = "10px 'IBM Plex Sans', 'Avenir Next', sans-serif";
    ctx.textAlign = "right";
    ctx.fillText(`avg TO ${Math.max(0, mean(turns) * 100).toFixed(1)}%`, dims.right - 8, laneBottom - 3);
  }

  drawLegend(
    ctx,
    profiles.map((p) => ({ label: shortMethod(p.label), color: p.color, dash: p.dash, alpha: p.alpha })),
    10,
  );

  ctx.fillStyle = "#2a2a2a";
  ctx.font = "11px 'IBM Plex Sans', 'Avenir Next', sans-serif";
  ctx.textAlign = "right";
  ctx.fillText("solid: risky allocation | dashed: turnover", dims.right, 14);
}

export function drawRegimeRisk(canvas, regimes, rows) {
  const { ctx, width, height } = setupCanvas(canvas);
  ctx.clearRect(0, 0, width, height);

  if (!regimes || regimes.length === 0 || !rows || rows.length === 0) {
    return;
  }

  const dims = {
    left: 64,
    right: width - 16,
    top: 24,
    bottom: height - 50,
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
  ctx.strokeStyle = "#333333";
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
    ctx.font = "11px 'IBM Plex Sans', 'Avenir Next', sans-serif";
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
  );
}
