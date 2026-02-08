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
  ctx.translate(left - 38, (top + bottom) / 2);
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

function drawLegend(ctx, legendItems, width, y) {
  let x = 14;
  ctx.font = "12px 'IBM Plex Sans', 'Avenir Next', sans-serif";

  for (const item of legendItems) {
    if (item.kind === "bar") {
      ctx.fillStyle = item.color;
      ctx.fillRect(x, y - 1, 14, 5);
    } else if (item.kind === "bar_hatch") {
      drawPatternFill(ctx, x, y - 2, 14, 6, "diagonal", item.color);
      ctx.strokeStyle = item.color;
      ctx.lineWidth = 1;
      ctx.strokeRect(x, y - 2, 14, 6);
    } else if (item.kind === "bar_outline") {
      ctx.strokeStyle = item.color;
      ctx.lineWidth = 1.4;
      ctx.strokeRect(x, y - 2, 14, 6);
    } else {
      ctx.save();
      ctx.strokeStyle = item.color;
      ctx.lineWidth = 2;
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

function pct(x) {
  return `${x.toFixed(1)}%`;
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

  for (let k = -h; k <= w + h; k += 4) {
    ctx.beginPath();
    ctx.moveTo(x + k, y + h);
    ctx.lineTo(x + k + h, y);
    ctx.stroke();
  }
  ctx.restore();
}

export function drawImpactBars(canvas, impact) {
  const { ctx, width, height } = setupCanvas(canvas);
  ctx.clearRect(0, 0, width, height);

  const dims = {
    left: 64,
    right: width - 18,
    top: 24,
    bottom: height - 46,
  };

  const values = impact.flatMap((g) => [g.anchor, g.proj, 0]);
  const [rawMin, rawMax] = extent(values, 0.15);
  const yMin = Math.min(rawMin, 0);
  const yMax = Math.max(rawMax, 0);

  function yToPx(y) {
    return dims.bottom - ((y - yMin) / (yMax - yMin)) * (dims.bottom - dims.top);
  }

  function xCenter(i, n) {
    return dims.left + ((i + 0.5) / n) * (dims.right - dims.left);
  }

  drawGrid(ctx, dims, 4);
  drawAxes(ctx, dims, "Metrics", "Delta vs naive (percentage points)");

  const zeroY = yToPx(0);
  ctx.strokeStyle = "#333333";
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  ctx.moveTo(dims.left, zeroY);
  ctx.lineTo(dims.right, zeroY);
  ctx.stroke();

  const groupWidth = (dims.right - dims.left) / impact.length;
  const barWidth = Math.min(22, groupWidth * 0.22);

  for (let i = 0; i < impact.length; i += 1) {
    const group = impact[i];
    const cx = xCenter(i, impact.length);

    drawBar(ctx, cx - barWidth * 0.7, barWidth, group.anchor, yToPx, zeroY, {
      color: "#0f5fbf",
      pattern: "diagonal",
      outlined: true,
    });
    drawBar(ctx, cx + barWidth * 0.7, barWidth, group.proj, yToPx, zeroY, {
      color: "#111111",
      pattern: "solid",
      outlined: false,
    });

    ctx.fillStyle = "#222222";
    ctx.font = "12px 'IBM Plex Sans', 'Avenir Next', sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(group.label, cx, dims.bottom + 16);

    labelValue(ctx, cx - barWidth * 0.7, yToPx(group.anchor), group.anchor, "#0f5fbf", zeroY);
    labelValue(ctx, cx + barWidth * 0.7, yToPx(group.proj), group.proj, "#111111", zeroY);
  }

  drawLegend(
    ctx,
    [
      { label: "Anchor", color: "#0f5fbf", kind: "bar_hatch" },
      { label: "Anchor + Projection", color: "#111111", kind: "bar" },
    ],
    width,
    10,
  );
}

function drawBar(ctx, centerX, width, value, yToPx, zeroY, style) {
  const y = yToPx(value);
  const top = Math.min(y, zeroY);
  const h = Math.max(1, Math.abs(y - zeroY));
  const left = centerX - width / 2;

  drawPatternFill(ctx, left, top, width, h, style.pattern || "solid", style.color || "#111111");
  if (style.outlined) {
    ctx.strokeStyle = style.color || "#111111";
    ctx.lineWidth = 1.1;
    ctx.strokeRect(left, top, width, h);
  }
}

function labelValue(ctx, x, y, value, color, zeroY) {
  ctx.fillStyle = color;
  ctx.font = "11px 'IBM Plex Sans', 'Avenir Next', sans-serif";
  ctx.textAlign = "center";

  if (y < zeroY) {
    ctx.fillText(pct(value), x, y - 6);
  } else {
    ctx.fillText(pct(value), x, y + 14);
  }
}

export function drawEquity(canvas, series, stressMarkers) {
  const { ctx, width, height } = setupCanvas(canvas);
  ctx.clearRect(0, 0, width, height);

  const dims = {
    left: 62,
    right: width - 20,
    top: 22,
    bottom: height - 46,
  };

  drawGrid(ctx, dims, 4);
  drawAxes(ctx, dims, "Time", "Equity");

  const allY = series.flatMap((s) => s.values);
  const [yMin, yMax] = extent(allY, 0.08);
  const maxT = Math.max(...series.map((s) => s.values.length - 1));

  function xToPx(t) {
    return dims.left + (t / Math.max(1, maxT)) * (dims.right - dims.left);
  }

  function yToPx(y) {
    return dims.bottom - ((y - yMin) / (yMax - yMin)) * (dims.bottom - dims.top);
  }

  // Shade contiguous stress windows so regime context is visible at a glance.
  if (Array.isArray(stressMarkers) && stressMarkers.length > 0) {
    const sorted = [...stressMarkers].sort((a, b) => a - b);
    const segments = [];
    let segStart = sorted[0];
    let segEnd = sorted[0];

    for (let i = 1; i < sorted.length; i += 1) {
      const idx = sorted[i];
      if (idx <= segEnd + 1) {
        segEnd = idx;
      } else {
        segments.push([segStart, segEnd]);
        segStart = idx;
        segEnd = idx;
      }
    }
    segments.push([segStart, segEnd]);

    const stepPx = maxT > 0 ? (dims.right - dims.left) / maxT : 8;
    ctx.fillStyle = "rgba(17, 17, 17, 0.08)";
    for (const [a, b] of segments) {
      const left = Math.max(dims.left, xToPx(a) - stepPx * 0.5);
      const right = Math.min(dims.right, xToPx(b) + stepPx * 0.5);
      ctx.fillRect(left, dims.top, Math.max(1, right - left), dims.bottom - dims.top);
    }

    ctx.strokeStyle = "rgba(17, 17, 17, 0.18)";
    ctx.lineWidth = 1;
    for (const [a, b] of segments) {
      const left = Math.max(dims.left, xToPx(a) - stepPx * 0.5);
      const right = Math.min(dims.right, xToPx(b) + stepPx * 0.5);
      ctx.beginPath();
      ctx.moveTo(left, dims.top);
      ctx.lineTo(left, dims.bottom);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(right, dims.top);
      ctx.lineTo(right, dims.bottom);
      ctx.stroke();
    }
  }

  for (const s of series) {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 2;
    ctx.setLineDash(s.dash || []);
    ctx.beginPath();

    for (let t = 0; t < s.values.length; t += 1) {
      const x = xToPx(t);
      const y = yToPx(s.values[t]);
      if (t === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }

    ctx.stroke();
    ctx.setLineDash([]);
  }

  drawLegend(ctx, series.map((s) => ({ label: s.label, color: s.color, dash: s.dash })), width, 10);
}

export function drawRegimeBars(canvas, rows) {
  const { ctx, width, height } = setupCanvas(canvas);
  ctx.clearRect(0, 0, width, height);

  if (!rows || rows.length === 0) {
    return;
  }

  const dims = {
    left: 70,
    right: width - 16,
    top: 24,
    bottom: height - 46,
  };

  const values = rows.flatMap((r) => [r.driftBp, r.stressBp, 0]);
  const [rawMin, rawMax] = extent(values, 0.15);
  const yMin = Math.min(rawMin, 0);
  const yMax = Math.max(rawMax, 0);

  function yToPx(y) {
    return dims.bottom - ((y - yMin) / (yMax - yMin)) * (dims.bottom - dims.top);
  }

  function xCenter(i, n) {
    return dims.left + ((i + 0.5) / n) * (dims.right - dims.left);
  }

  drawGrid(ctx, dims, 4);
  drawAxes(ctx, dims, "Method", "Mean return (basis points)");

  const zeroY = yToPx(0);
  ctx.strokeStyle = "#333333";
  ctx.lineWidth = 1.3;
  ctx.beginPath();
  ctx.moveTo(dims.left, zeroY);
  ctx.lineTo(dims.right, zeroY);
  ctx.stroke();

  const groupWidth = (dims.right - dims.left) / rows.length;
  const barWidth = Math.min(18, groupWidth * 0.18);

  for (let i = 0; i < rows.length; i += 1) {
    const row = rows[i];
    const cx = xCenter(i, rows.length);

    const driftY = yToPx(row.driftBp);
    const stressY = yToPx(row.stressBp);
    const driftTop = Math.min(driftY, zeroY);
    const driftH = Math.max(1, Math.abs(driftY - zeroY));
    const stressTop = Math.min(stressY, zeroY);
    const stressH = Math.max(1, Math.abs(stressY - zeroY));
    const driftX = cx - barWidth * 1.25;
    const stressX = cx + barWidth * 0.25;

    // Drift regime: outlined bar for quick contrast.
    ctx.strokeStyle = row.color;
    ctx.lineWidth = 1.5;
    ctx.setLineDash(methodDash(row.label));
    ctx.strokeRect(driftX, driftTop, barWidth, driftH);
    ctx.setLineDash([]);

    // Stress regime: pattern-filled for grayscale readability.
    drawPatternFill(ctx, stressX, stressTop, barWidth, stressH, methodPattern(row.label), row.color);
    ctx.strokeStyle = row.color;
    ctx.lineWidth = 1.1;
    ctx.strokeRect(stressX, stressTop, barWidth, stressH);

    ctx.fillStyle = "#222222";
    ctx.font = "11px 'IBM Plex Sans', 'Avenir Next', sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(shortLabel(row.label), cx, dims.bottom + 16);
  }

  drawLegend(
    ctx,
    [
      { label: "Drift regime", color: "#444444", kind: "bar_outline" },
      { label: "Stress regime", color: "#444444", kind: "bar" },
    ],
    width,
    10,
  );
}

function shortLabel(label) {
  if (!label) {
    return "";
  }
  if (label.includes("Naive")) {
    return "Naive";
  }
  if (label.includes("Constrained")) {
    return "Constrained";
  }
  if (label.includes("Replay")) {
    return "Replay";
  }
  return label;
}

function methodPattern(label) {
  if (!label) {
    return "solid";
  }
  if (label.includes("Naive")) {
    return "horizontal";
  }
  if (label.includes("Replay")) {
    return "diagonal";
  }
  return "solid";
}

function methodDash(label) {
  if (!label) {
    return [];
  }
  if (label.includes("Naive")) {
    return [5, 3];
  }
  if (label.includes("Replay")) {
    return [2, 2];
  }
  return [];
}
